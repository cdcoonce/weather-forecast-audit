"""Per-slice scoring and date-block bootstrap uncertainty.

Forecast errors are strongly correlated within an issuance date -- a bad NBM
cycle runs warm or cold at every station it covers together -- so treating
each verification row as an independent bootstrap unit understates
uncertainty: row-level resampling shuffles that shared-day noise away and
reports intervals that are falsely narrow. This module always resamples
calendar blocks of issuance dates (`BLOCK_DAYS` consecutive `run_date`s per
block; `BLOCK_DAYS = 1` is one issuance date per block), never individual
rows. See docs/methodology.md ("Scoring and uncertainty") for the measured
effect and for two documented limitations of the current default block
length.

No I/O, no DuckDB, no dbt: `score` takes a `polars.DataFrame` matching
`fct_forecast_verification`'s schema (dbt/models/marts) and returns a pure
per-slice summary.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
import polars as pl

N_BOOT = 2000
SEED = 20260927
CI_LEVEL = 0.95
MIN_SAMPLE_DATES = 30
BLOCK_DAYS = 1

_REQUIRED_COLUMNS = (
    "station",
    "run_date",
    "lead_day",
    "variable",
    "source",
    "target_date",
    "scorable",
    "error_f",
)
_ALLOWED_BY = frozenset({"station", "month", "season", "lead_day", "variable"})
_MATCH_KEY = ("station", "run_date", "lead_day", "variable")
_RAW_SOURCE = "raw_nbm"


def _slice_rng(seed: int, slice_key: tuple[object, ...]) -> np.random.Generator:
    """A `default_rng` seeded stably from `seed` and a slice's key.

    Never Python's `hash()`, which is salted per process: `sha256` of the
    key's `repr` is the same across processes and regardless of which other
    slices happen to be present in the same call, so a slice's CI is
    reproducible and unaffected by unrelated rows (e.g. another station).
    """
    digest = hashlib.sha256(repr(slice_key).encode()).digest()
    stable_int = int.from_bytes(digest[:8], byteorder="big")
    return np.random.default_rng([seed, stable_int])


def _block_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    ci_level: float,
) -> tuple[float | None, float | None]:
    """Percentile CI for a per-block ratio statistic, resampling blocks.

    `numerator`/`denominator` are row-level arrays that this function sums
    into per-block totals (e.g. signed error and a constant 1 for bias;
    matched `|error_source|` and `|error_raw|` for skill). `n_boot`
    multinomial draws of block weights `W` -- each drawing `n_blocks` blocks
    with replacement -- give `n_boot` replicate ratios
    `(W . numerator_blocks) / (W . denominator_blocks)`; the CI is the
    `(1-ci_level)/2, 1-(1-ci_level)/2` percentile of those replicates
    (numpy's default linear interpolation).

    Passing `block_ids = np.arange(len(numerator))` treats every row as its
    own block, i.e. ordinary row-level resampling -- used only by the
    coverage tests to measure how much narrower (and wrong) that is.

    Fewer than two distinct blocks cannot support a bootstrap: `(None,
    None)`. If every replicate lands on a zero denominator (all sampled
    blocks have `denominator` sum 0), the ratio is undefined everywhere and
    this also returns `(None, None)`.
    """
    unique_blocks, block_index = np.unique(block_ids, return_inverse=True)
    n_blocks = unique_blocks.size
    if n_blocks < 2:
        return None, None

    block_index = block_index.reshape(-1)
    block_numerator = np.zeros(n_blocks)
    block_denominator = np.zeros(n_blocks)
    np.add.at(block_numerator, block_index, numerator)
    np.add.at(block_denominator, block_index, denominator)

    weights = rng.multinomial(
        n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot
    )
    replicate_numerator = weights @ block_numerator
    replicate_denominator = weights @ block_denominator

    with np.errstate(invalid="ignore", divide="ignore"):
        replicates = replicate_numerator / replicate_denominator
    replicates = np.where(replicate_denominator == 0, np.nan, replicates)
    if np.all(np.isnan(replicates)):
        return None, None

    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.nanquantile(replicates, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _no_detectable_bias(lo: float | None, hi: float | None) -> bool:
    """True when the bias CI spans 0 (endpoints inclusive), or is undefined.

    An undefined CI (fewer than two blocks) is conservative: the site shows
    "no detectable bias" rather than manufacturing a false finding out of an
    interval that was never estimable.
    """
    if lo is None or hi is None:
        return True
    return lo <= 0.0 <= hi


def score(
    rows: pl.DataFrame,
    by: Sequence[str] = (),
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    ci_level: float = CI_LEVEL,
    min_sample_dates: int = MIN_SAMPLE_DATES,
    block_days: int = BLOCK_DAYS,
) -> pl.DataFrame:
    """Score `rows` per slice, with a date-block bootstrap CI on bias/mae/skill.

    `by` may combine `station`, `month`, `season`, `lead_day`, `variable`
    (month/season are derived from `target_date`, not `run_date` -- a
    lead-3 run issued 06-29 verifies 07-02, which belongs in July, the
    weather day being forecast). `source` is always an implicit grouping
    key -- pooling error across sources is meaningless -- and may also be
    named in `by` without being duplicated in the output.

    `min_sample_flag` counts distinct issuance dates (`n_dates`), not rows,
    because dates -- not verification rows -- are the bootstrap's
    independent resampling unit.

    `skill` (`1 - MAE_source / MAE_raw_nbm`) is computed on matched pairs
    only: a row of a non-`raw_nbm` source is matched when a scorable
    `raw_nbm` row exists for the same `(station, run_date, lead_day,
    variable)`. `raw_nbm` itself always scores exactly 0.0. A slice with no
    matched pairs, or whose matched `raw_nbm` MAE is 0, has `skill` (and its
    CI) null rather than a divide-by-zero.
    """
    for column in _REQUIRED_COLUMNS:
        if column not in rows.columns:
            msg = f"missing required column: {column}"
            raise ValueError(msg)

    unknown_dims = [dim for dim in by if dim not in _ALLOWED_BY and dim != "source"]
    if unknown_dims:
        msg = f"unknown by dimension(s): {unknown_dims}"
        raise ValueError(msg)
    if n_boot < 1:
        msg = f"n_boot must be >= 1, got {n_boot}"
        raise ValueError(msg)
    if not (0 < ci_level < 1):
        msg = f"ci_level must be in (0, 1), got {ci_level}"
        raise ValueError(msg)
    if block_days < 1:
        msg = f"block_days must be >= 1, got {block_days}"
        raise ValueError(msg)

    group_keys = list(dict.fromkeys([*by, "source"]))

    filtered = (
        rows.filter(pl.col("scorable") & pl.col("error_f").is_not_null())
        .with_columns(
            month=pl.col("target_date").dt.month(),
            season=(
                pl.when(pl.col("target_date").dt.month().is_in([12, 1, 2]))
                .then(pl.lit("DJF"))
                .when(pl.col("target_date").dt.month().is_in([3, 4, 5]))
                .then(pl.lit("MAM"))
                .when(pl.col("target_date").dt.month().is_in([6, 7, 8]))
                .then(pl.lit("JJA"))
                .otherwise(pl.lit("SON"))
            ),
            block_id=(pl.col("run_date").dt.epoch(time_unit="d") // block_days),
        )
    )

    raw_lookup = filtered.filter(pl.col("source") == _RAW_SOURCE).select(
        [*_MATCH_KEY, pl.col("error_f").alias("error_raw")]
    )
    matched = filtered.join(raw_lookup, on=list(_MATCH_KEY), how="inner")
    matched_non_raw = matched.filter(pl.col("source") != _RAW_SOURCE)

    if filtered.height == 0:
        columns = [
            *group_keys,
            "n",
            "n_dates",
            "bias",
            "bias_lo",
            "bias_hi",
            "mae",
            "mae_lo",
            "mae_hi",
            "skill",
            "skill_lo",
            "skill_hi",
            "min_sample_flag",
            "no_detectable_bias",
        ]
        return pl.DataFrame(schema=dict.fromkeys(columns, pl.Null))

    agg = (
        filtered.group_by(group_keys)
        .agg(
            n=pl.len(),
            n_dates=pl.col("run_date").n_unique(),
            bias=pl.col("error_f").mean(),
            mae=pl.col("error_f").abs().mean(),
        )
        .sort(group_keys)
    )

    if matched_non_raw.height > 0:
        skill_agg = matched_non_raw.group_by(group_keys).agg(
            n_matched=pl.len(),
            sum_abs_source=pl.col("error_f").abs().sum(),
            sum_abs_raw=pl.col("error_raw").abs().sum(),
        )
        agg = agg.join(skill_agg, on=group_keys, how="left")
    else:
        agg = agg.with_columns(
            n_matched=pl.lit(None, dtype=pl.UInt32),
            sum_abs_source=pl.lit(None, dtype=pl.Float64),
            sum_abs_raw=pl.lit(None, dtype=pl.Float64),
        )

    filtered_groups: dict[tuple[object, ...], pl.DataFrame] = {}
    for key, sub in filtered.group_by(group_keys, maintain_order=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        filtered_groups[key_tuple] = sub

    matched_groups: dict[tuple[object, ...], pl.DataFrame] = {}
    for key, sub in matched_non_raw.group_by(group_keys, maintain_order=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        matched_groups[key_tuple] = sub

    records: list[dict[str, object]] = []
    for row in agg.iter_rows(named=True):
        slice_key = tuple(row[k] for k in group_keys)
        sub = filtered_groups[slice_key]
        error_arr = sub["error_f"].to_numpy()
        block_arr = sub["block_id"].to_numpy()
        ones_arr = np.ones_like(error_arr)

        rng = _slice_rng(seed, slice_key)
        bias_lo, bias_hi = _block_bootstrap(
            error_arr, ones_arr, block_arr, rng, n_boot, ci_level
        )
        mae_lo, mae_hi = _block_bootstrap(
            np.abs(error_arr), ones_arr, block_arr, rng, n_boot, ci_level
        )

        source_value = row["source"]
        n_matched = row["n_matched"]
        sum_abs_source = row["sum_abs_source"]
        sum_abs_raw = row["sum_abs_raw"]

        if source_value == _RAW_SOURCE:
            skill, skill_lo, skill_hi = 0.0, 0.0, 0.0
        elif n_matched is None or sum_abs_raw == 0:
            skill, skill_lo, skill_hi = None, None, None
        else:
            skill = 1.0 - sum_abs_source / sum_abs_raw
            matched_sub = matched_groups[slice_key]
            matched_source_abs = np.abs(matched_sub["error_f"].to_numpy())
            matched_raw_abs = np.abs(matched_sub["error_raw"].to_numpy())
            matched_block = matched_sub["block_id"].to_numpy()
            ratio_lo, ratio_hi = _block_bootstrap(
                matched_source_abs,
                matched_raw_abs,
                matched_block,
                rng,
                n_boot,
                ci_level,
            )
            if ratio_lo is None:
                skill_lo, skill_hi = None, None
            else:
                skill_lo, skill_hi = 1.0 - ratio_hi, 1.0 - ratio_lo

        n_dates = row["n_dates"]
        record = {k: row[k] for k in group_keys}
        record.update(
            n=row["n"],
            n_dates=n_dates,
            bias=row["bias"],
            bias_lo=bias_lo,
            bias_hi=bias_hi,
            mae=row["mae"],
            mae_lo=mae_lo,
            mae_hi=mae_hi,
            skill=skill,
            skill_lo=skill_lo,
            skill_hi=skill_hi,
            min_sample_flag=n_dates < min_sample_dates,
            no_detectable_bias=_no_detectable_bias(bias_lo, bias_hi),
        )
        records.append(record)

    return pl.DataFrame(records)
