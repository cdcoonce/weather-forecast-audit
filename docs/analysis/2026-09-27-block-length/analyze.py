"""Measure the bootstrap block length, per PREREG.md (issue #31).

Two subcommands:

- `extract` connects READ-ONLY to `$WFA_DUCKDB_PATH` and writes the fixed
  12-station, 2025 sample to `data/verification_rows.parquet`, so the rest
  of the analysis re-runs from the committed extract without a warehouse.
- `analyze` (the default) reads that parquet and implements the
  pre-registered pooled/per-station block-length estimates, the ACF table,
  the deseasonalized estimates, the coverage replay, and the `BLOCK_DAYS`
  decision rule (including its cap and fallback), writing `results/*.csv`
  and `RESULTS.md`.

`analyze` never touches the network or DuckDB: `arch.bootstrap.
optimal_block_length` and the AR fit/simulate/replay machinery in
`weather_forecast_audit.block_length` are pure functions of the extracted
frame. Run as `uv run python docs/analysis/2026-09-27-block-length/analyze.py
<cmd>` from the repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from arch.bootstrap import optimal_block_length

from weather_forecast_audit.block_length import (
    ArFit,
    BlockChoice,
    acf,
    choose_block_days,
    deseasonalize,
    first_passing_block_days,
    fit_ar_yule_walker,
    pooled_series,
    replay_coverage,
    station_series,
    upper_median_index,
)

_HERE = Path(__file__).resolve().parent
_DEFAULT_DATA = _HERE / "data" / "verification_rows.parquet"
_DEFAULT_OUT = _HERE

# PREREG.md "Sample (fixed)": #6's twelve stations.
_STATIONS = (
    "KPHX",
    "KSFO",
    "KSEA",
    "KDEN",
    "KSLC",
    "KBIS",
    "KOKC",
    "KMSP",
    "KORD",
    "KATL",
    "KMIA",
    "KBOS",
)
_START_DATE = "2025-01-01"
_END_DATE = "2025-12-31"
_EXPECTED_N_DATES = (
    365  # calendar run_dates in [2025-01-01, 2025-12-31], not a leap year
)

_MIN_STATIONS = 6
_MIN_STATION_DATES = 300
_MISSING_FRACTION_FLAG = 0.10
_CAP = 14
_COVERAGE_THRESHOLD = 0.90
_N_SIMS = 400
_N_BOOT = 500
_ACF_MAX_LAG = 14
_DESEASON_WINDOW = 31
_REPLAY_SEED = 20260927

_EXTRACT_COLUMNS = (
    "station",
    "run_date",
    "lead_day",
    "variable",
    "source",
    "target_date",
    "scorable",
    "error_f",
    "nbm_version",
)


# -- shared helpers -----------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_HERE,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


# -- extract --------------------------------------------------------------


def _db_path() -> str:
    path = os.environ.get("WFA_DUCKDB_PATH")
    if not path:
        msg = "WFA_DUCKDB_PATH must be set to the DuckDB database file path"
        raise SystemExit(msg)
    return path


def cmd_extract(data_path: Path) -> None:
    """Extract the fixed sample from `fct_forecast_verification`, read-only."""
    db_path = _db_path()
    placeholders = ", ".join(f"'{station}'" for station in _STATIONS)
    columns = ", ".join(_EXTRACT_COLUMNS)
    query = f"""
        select {columns}
        from fct_forecast_verification
        where source = 'raw_nbm'
          and station in ({placeholders})
          and run_date >= date '{_START_DATE}'
          and run_date <= date '{_END_DATE}'
        order by station, run_date, lead_day, variable
    """
    with duckdb.connect(db_path, read_only=True) as conn:
        frame = conn.sql(query).pl()

    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(data_path)

    counts_by_station = dict(
        frame.group_by("station").len().sort("station").iter_rows()
    )
    for station in _STATIONS:
        n = counts_by_station.get(station, 0)
        if n == 0:
            print(
                f"{station}: 0 rows -- DROPPED (no data; PREREG.md drops, not replaces)"
            )
        else:
            print(f"{station}: {n} rows")

    print(f"wrote {data_path} ({frame.height} rows total)")
    print(f"sha256 {_sha256(data_path)}")


# -- analyze ---------------------------------------------------------------


def _missing(n_present: int) -> tuple[int, float]:
    n_missing = _EXPECTED_N_DATES - n_present
    return n_missing, n_missing / _EXPECTED_N_DATES


# PREREG.md's five (variable, lead) series, in its fixed order. The replay's
# seed streams are assigned by this order, so it is a constant, never read
# off the data: a missing key must stop the analysis, not shift every seed.
_SERIES_KEYS: tuple[tuple[str, int], ...] = (
    ("max", 1),
    ("max", 2),
    ("min", 1),
    ("min", 2),
    ("min", 3),
)


def _series_keys(pooled: pl.DataFrame) -> list[tuple[str, int]]:
    """PREREG.md's five keys; raise if the data's keys differ in any way."""
    present = set(pooled.select("variable", "lead_day").unique().rows())
    if present != set(_SERIES_KEYS):
        msg = (
            f"pooled series keys {sorted(present)} != PREREG keys "
            f"{list(_SERIES_KEYS)}; the pre-registration has no rule for this"
        )
        raise SystemExit(msg)
    return list(_SERIES_KEYS)


def _optimal_block_length(values: np.ndarray) -> float:
    if values.size < 2:
        msg = f"cannot estimate a block length from {values.size} value(s)"
        raise SystemExit(msg)
    return float(optimal_block_length(values)["circular"].iloc[0])


def cmd_analyze(data_path: Path, out_dir: Path) -> None:
    if not data_path.exists():
        msg = f"no extracted data at {data_path}; run the `extract` subcommand first"
        raise SystemExit(msg)

    rows = pl.read_parquet(data_path)
    pooled = pooled_series(rows, min_stations=_MIN_STATIONS)
    stations = station_series(rows)

    keys = _series_keys(pooled)
    if len(keys) != 5:
        print(
            f"WARNING: PREREG.md expects five (variable, lead_day) series; "
            f"found {len(keys)}: {keys}"
        )

    estimates_rows: list[dict[str, object]] = []
    station_estimates_rows: list[dict[str, object]] = []
    acf_rows: list[dict[str, object]] = []
    b_by_key: dict[tuple[str, int], tuple[float, float]] = {}
    median_station_by_key: dict[tuple[str, int], str | None] = {}
    pooled_values_by_key: dict[tuple[str, int], np.ndarray] = {}
    median_station_values_by_key: dict[tuple[str, int], np.ndarray | None] = {}

    for variable, lead_day in keys:
        pooled_sub = pooled.filter(
            (pl.col("variable") == variable) & (pl.col("lead_day") == lead_day)
        ).sort("run_date")
        pooled_values = pooled_sub["mean_error"].to_numpy()
        pooled_values_by_key[(variable, lead_day)] = pooled_values
        n_dates = pooled_values.size
        n_missing, missing_fraction = _missing(n_dates)
        b_pooled = _optimal_block_length(pooled_values)

        deseasoned = deseasonalize(pooled_values, window=_DESEASON_WINDOW)
        b_pooled_deseasonalized = _optimal_block_length(deseasoned)

        n = pooled_values.size
        band = 1.96 / math.sqrt(n) if n > 0 else float("nan")
        for lag, value in enumerate(acf(pooled_values, max_lag=_ACF_MAX_LAG), start=1):
            acf_rows.append(
                {
                    "variable": variable,
                    "lead_day": lead_day,
                    "lag": lag,
                    "acf": value,
                    "band": band,
                }
            )

        station_sub = stations.filter(
            (pl.col("variable") == variable) & (pl.col("lead_day") == lead_day)
        )
        station_names = sorted(station_sub["station"].unique().to_list())
        qualifying_b: list[float] = []
        qualifying_station: list[str] = []
        qualifying_values: list[np.ndarray] = []
        for station in station_names:
            values = (
                station_sub.filter(pl.col("station") == station)
                .sort("run_date")["error_f"]
                .to_numpy()
            )
            n_st_dates = values.size
            n_st_missing, _ = _missing(n_st_dates)
            excluded = n_st_dates < _MIN_STATION_DATES
            b_station = _optimal_block_length(values) if values.size >= 2 else None
            station_estimates_rows.append(
                {
                    "station": station,
                    "variable": variable,
                    "lead_day": lead_day,
                    "n_dates": n_st_dates,
                    "n_missing": n_st_missing,
                    "excluded_from_median": excluded,
                    "b_station": b_station,
                    "is_upper_median": False,
                }
            )
            if not excluded and b_station is not None:
                qualifying_b.append(b_station)
                qualifying_station.append(station)
                qualifying_values.append(values)

        if qualifying_b:
            idx = upper_median_index(qualifying_b)
            b_station_value = qualifying_b[idx]
            median_station = qualifying_station[idx]
            median_values = qualifying_values[idx]
            for row in station_estimates_rows:
                if (
                    row["variable"] == variable
                    and row["lead_day"] == lead_day
                    and row["station"] == median_station
                ):
                    row["is_upper_median"] = True
        else:
            msg = (
                f"no station series qualifies for the median at "
                f"(variable={variable}, lead_day={lead_day}); the "
                f"pre-registration has no rule for this"
            )
            raise SystemExit(msg)

        b_by_key[(variable, lead_day)] = (b_pooled, b_station_value)
        median_station_by_key[(variable, lead_day)] = median_station
        median_station_values_by_key[(variable, lead_day)] = median_values

        estimates_rows.append(
            {
                "variable": variable,
                "lead_day": lead_day,
                "n_dates": n_dates,
                "n_missing": n_missing,
                "missing_fraction": missing_fraction,
                "missing_flagged": missing_fraction > _MISSING_FRACTION_FLAG,
                "b_pooled": b_pooled,
                "b_station": b_station_value,
                "upper_median_station": median_station,
                "b_pooled_deseasonalized": b_pooled_deseasonalized,
            }
        )

    # -- coverage replay: five pooled series, then the upper-median station
    # series in the same key order (PREREG.md "Coverage replay").
    replay_series: list[tuple[str, tuple[str, int], np.ndarray]] = []
    for variable, lead_day in keys:
        replay_series.append(
            ("pooled", (variable, lead_day), pooled_values_by_key[(variable, lead_day)])
        )
    for variable, lead_day in keys:
        key = (variable, lead_day)
        values = median_station_values_by_key[key]
        label = f"station:{median_station_by_key[key]}"
        replay_series.append((label, key, values))

    seed_children = np.random.SeedSequence(_REPLAY_SEED).spawn(10)
    ar_fits: list[ArFit] = [
        fit_ar_yule_walker(values, max_p=7) for _, _, values in replay_series
    ]

    ar_fit_rows = [
        {
            "series": label,
            "variable": key[0],
            "lead_day": key[1],
            "p": fit.p,
            "phi": ";".join(f"{coef:.10g}" for coef in fit.phi),
            "sigma2": fit.sigma2,
            "mean": fit.mean,
        }
        for (label, key, _), fit in zip(replay_series, ar_fits, strict=True)
    ]

    coverage_rows: list[dict[str, object]] = []
    evaluated: dict[int, list[float]] = {}

    def coverage_at(block_days: int) -> list[float]:
        if block_days in evaluated:
            return evaluated[block_days]
        covs = []
        for i, (label, key, _) in enumerate(replay_series):
            rng = np.random.default_rng(seed_children[i])
            fit = ar_fits[i]
            n = replay_series[i][2].size
            cov = replay_coverage(
                fit, n=n, block_days=block_days, n_sims=_N_SIMS, rng=rng, n_boot=_N_BOOT
            )
            covs.append(cov)
            coverage_rows.append(
                {
                    "block_days": block_days,
                    "series": label,
                    "variable": key[0],
                    "lead_day": key[1],
                    "coverage": cov,
                }
            )
        evaluated[block_days] = covs
        return covs

    choice = choose_block_days(b_by_key, cap=_CAP)

    outcome: str
    fallback_from: int | None = None
    if choice.capped:
        chosen_block_days = 1
        outcome = "capped"
    else:
        covs = coverage_at(choice.value)
        if all(c >= _COVERAGE_THRESHOLD for c in covs):
            chosen_block_days = choice.value
            outcome = "rule_passes"
        else:
            fallback = first_passing_block_days(
                coverage_at,
                start=choice.value + 1,
                cap=_CAP,
                threshold=_COVERAGE_THRESHOLD,
            )
            fallback_from = choice.value
            if fallback is None:
                chosen_block_days = 1
                outcome = "fallback_exhausted_capped"
            else:
                chosen_block_days = fallback
                outcome = "fallback"

    coverage_at(1)  # PREREG.md: always report the current default's coverage too.

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(estimates_rows).write_csv(results_dir / "estimates.csv")
    pl.DataFrame(station_estimates_rows).write_csv(
        results_dir / "station_estimates.csv"
    )
    pl.DataFrame(acf_rows).write_csv(results_dir / "acf.csv")
    pl.DataFrame(ar_fit_rows).write_csv(results_dir / "ar_fits.csv")
    pl.DataFrame(coverage_rows).write_csv(results_dir / "coverage.csv")

    _write_results_md(
        out_dir=out_dir,
        data_path=data_path,
        estimates_rows=estimates_rows,
        station_estimates_rows=station_estimates_rows,
        acf_rows=acf_rows,
        coverage_rows=coverage_rows,
        choice=choice,
        outcome=outcome,
        chosen_block_days=chosen_block_days,
        fallback_from=fallback_from,
    )

    print(f"BLOCK_DAYS decision: {outcome}, chosen={chosen_block_days}")
    print(f"wrote {results_dir} and {out_dir / 'RESULTS.md'}")


def _write_results_md(
    *,
    out_dir: Path,
    data_path: Path,
    estimates_rows: list[dict[str, object]],
    station_estimates_rows: list[dict[str, object]],
    acf_rows: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
    choice: BlockChoice,
    outcome: str,
    chosen_block_days: int,
    fallback_from: int | None,
) -> None:
    lines: list[str] = []
    lines.append("# Block-length measurement results (#31)")
    lines.append("")
    lines.append("Generated by `docs/analysis/2026-09-27-block-length/analyze.py`.")
    lines.append("Do not hand-edit; re-run the script to regenerate.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Extract (`{data_path.name}`) SHA-256: `{_sha256(data_path)}`")
    lines.append(f"- `analyze.py` SHA-256: `{_sha256(Path(__file__).resolve())}`")
    lines.append(f"- Git HEAD: `{_git_head()}`")
    lines.append("")
    lines.append("## Per-series estimates")
    lines.append("")
    lines.append(
        "| variable | lead_day | n_dates | n_missing | missing_flagged | "
        "b_pooled | b_station | upper_median_station | b_pooled_deseasonalized |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in estimates_rows:
        lines.append(
            f"| {row['variable']} | {row['lead_day']} | {row['n_dates']} | "
            f"{row['n_missing']} | {row['missing_flagged']} | "
            f"{row['b_pooled']} | {row['b_station']} | "
            f"{row['upper_median_station']} | {row['b_pooled_deseasonalized']} |"
        )
    lines.append("")
    lines.append("## Per-station estimates")
    lines.append("")
    lines.append(
        "| station | variable | lead_day | n_dates | excluded_from_median | "
        "b_station | is_upper_median |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in station_estimates_rows:
        lines.append(
            f"| {row['station']} | {row['variable']} | {row['lead_day']} | "
            f"{row['n_dates']} | {row['excluded_from_median']} | "
            f"{row['b_station']} | {row['is_upper_median']} |"
        )
    lines.append("")
    lines.append("## Autocorrelation of the pooled series (secondary, not binding)")
    lines.append("")
    lines.append(
        "Lag-k sample autocorrelation of each pooled (variable, lead) series; "
        "the band is +/-1.96/sqrt(n)."
    )
    lines.append("")
    acf_keys = list(dict.fromkeys((r["variable"], r["lead_day"]) for r in acf_rows))
    lines.append(
        "| lag | " + " | ".join(f"{v} {lead}" for v, lead in acf_keys) + " | band |"
    )
    lines.append("| --- " * (len(acf_keys) + 2) + "|")
    lags = sorted({int(r["lag"]) for r in acf_rows})
    by_key_lag = {(r["variable"], r["lead_day"], r["lag"]): r for r in acf_rows}
    for lag in lags:
        cells = [f"{by_key_lag[(v, lead, lag)]['acf']:.3f}" for v, lead in acf_keys]
        band = by_key_lag[(*acf_keys[0], lag)]["band"]
        lines.append(f"| {lag} | " + " | ".join(cells) + f" | {band:.3f} |")
    lines.append("")
    lines.append("## Coverage replay")
    lines.append("")
    lines.append("| block_days | series | variable | lead_day | coverage |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in coverage_rows:
        lines.append(
            f"| {row['block_days']} | {row['series']} | {row['variable']} | "
            f"{row['lead_day']} | {row['coverage']:.4f} |"
        )
    lines.append("")
    lines.append("## Decision rule walkthrough")
    lines.append("")
    lines.append(f"- Rule 2 raw ceiling: `{choice.raw_ceiling}`")
    lines.append(f"- Cap (rule 3): `{choice.capped}`")
    if fallback_from is not None:
        lines.append(f"- Rule value tried first: `{fallback_from}`")
    lines.append(f"- Outcome: `{outcome}`")
    lines.append("")
    if outcome in ("capped", "fallback_exhausted_capped"):
        lines.append(
            "## BLOCK_DAYS\n\n"
            "The rule's ceiling was not accepted by the coverage replay "
            f"(outcome `{outcome}`). Per PREREG.md rule 3, `BLOCK_DAYS` is "
            "left at 1, #16 stays blocked, and the choice goes back to "
            "Charles. The deseasonalized estimates above inform that "
            "revisit but do not decide it."
        )
    else:
        lines.append(f"## BLOCK_DAYS\n\n`BLOCK_DAYS = {chosen_block_days}`.")
    lines.append("")
    # The interpretation is written by hand in INTERPRETATION.md beside this
    # script and included verbatim, so regenerating RESULTS.md keeps it.
    interpretation = Path(__file__).resolve().parent / "INTERPRETATION.md"
    lines.append("## Interpretation")
    lines.append("")
    if interpretation.exists():
        lines.append(interpretation.read_text().rstrip())
    else:
        lines.append("(not yet written: add INTERPRETATION.md and re-run)")
    lines.append("")

    (out_dir / "RESULTS.md").write_text("\n".join(lines))


# -- CLI ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cmd",
        nargs="?",
        default="analyze",
        choices=("extract", "analyze"),
        help=(
            "'extract' pulls the sample from DuckDB; "
            "'analyze' (default) measures block length"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DEFAULT_DATA,
        help="verification_rows.parquet path (extract's output, analyze's input)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="directory analyze writes results/ and RESULTS.md into",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.cmd == "extract":
        cmd_extract(args.data)
    else:
        cmd_analyze(args.data, args.out)


if __name__ == "__main__":
    main()
