"""Block-length selection and coverage-replay helpers (issue #31).

Pure functions only: no `arch`, no DuckDB, no I/O. `arch.bootstrap.
optimal_block_length` -- the Politis & White (2004) / Patton, Politis & White
(2009) circular-block estimator the pre-registration calls for -- is used
only by `docs/analysis/2026-09-27-block-length/analyze.py`, which imports
this module alongside `arch`. Keeping `arch` out of this module means the
AR fitting, simulation and decision-rule machinery stays unit-testable
without that dependency.

See `docs/analysis/2026-09-27-block-length/PREREG.md` for the fixed
specification this implements: which rows count toward the pooled and
per-station series, the Yule-Walker AR fit and AIC order selection, the
coverage replay that is the acceptance check for a candidate `BLOCK_DAYS`,
and the decision rule (with its cap) that turns per-series estimates into
one value.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from weather_forecast_audit import scoring

_RAW_SOURCE = "raw_nbm"

# The coverage replay's synthetic frame (PREREG "Coverage replay", step 3):
# one row per simulated value, consecutive `run_date`s starting 2025-01-01,
# so `scoring.score`'s calendar blocks align exactly as they would on real
# data. Station/lead/variable are placeholders -- `by=()` never groups on
# them -- fixed so the frame always matches `scoring`'s required schema.
_SIM_START_DATE = date(2025, 1, 1)
_SIM_STATION = "SIM"
_SIM_LEAD_DAY = 1
_SIM_VARIABLE = "max"


def pooled_series(rows: pl.DataFrame, *, min_stations: int = 6) -> pl.DataFrame:
    """Per (variable, lead_day, run_date) cross-station mean `error_f`.

    Filters to `source == 'raw_nbm'`, `scorable`, non-null `error_f`
    (PREREG "Series", item 1), then drops dates with fewer than
    `min_stations` distinct stations scorable that date. `n_stations`
    counts distinct stations, not rows, so a duplicated row for the same
    station and date does not inflate it.
    """
    filtered = rows.filter(
        (pl.col("source") == _RAW_SOURCE)
        & pl.col("scorable")
        & pl.col("error_f").is_not_null()
    )
    grouped = filtered.group_by(["variable", "lead_day", "run_date"]).agg(
        mean_error=pl.col("error_f").mean(),
        n_stations=pl.col("station").n_unique(),
    )
    return grouped.filter(pl.col("n_stations") >= min_stations).sort(
        ["variable", "lead_day", "run_date"]
    )


def station_series(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (station, variable, lead_day, run_date) `error_f`.

    Same filter as `pooled_series` (PREREG "Series", item 2): `source ==
    'raw_nbm'`, `scorable`, non-null `error_f`.
    """
    filtered = rows.filter(
        (pl.col("source") == _RAW_SOURCE)
        & pl.col("scorable")
        & pl.col("error_f").is_not_null()
    )
    columns = ["station", "variable", "lead_day", "run_date", "error_f"]
    return filtered.select(columns).sort(columns[:-1])


def upper_median_index(values: Sequence[float]) -> int:
    """Index into `values` of the element at sorted position `len // 2`.

    An even count's "upper median" (PREREG "Decision rule", step 1) is the
    higher of the two middle values -- e.g. 12 values gives the 7th
    smallest. Ties are broken by a stable sort, so the earlier-indexed of
    equal values sorts first.
    """
    if len(values) == 0:
        msg = "upper_median_index requires at least one value"
        raise ValueError(msg)
    order = np.argsort(np.asarray(values), kind="stable")
    return int(order[len(values) // 2])


def acf(x: np.ndarray, max_lag: int = 14) -> np.ndarray:
    """Biased sample autocorrelation at lags `1..max_lag`, mean-removed.

    `rho(k) = gamma(k) / gamma(0)` with `gamma(k)` the lag-`k` autocovariance
    computed from the mean-removed series (the `1/n` normalization common to
    every `gamma(k)` cancels in the ratio).
    """
    x = np.asarray(x, dtype=float)
    demeaned = x - x.mean()
    gamma0 = np.sum(demeaned**2)
    result = np.empty(max_lag, dtype=float)
    for i in range(1, max_lag + 1):
        gamma_lag = np.sum(demeaned[:-i] * demeaned[i:])
        result[i - 1] = gamma_lag / gamma0
    return result


def deseasonalize(x: np.ndarray, window: int = 31) -> np.ndarray:
    """`x` minus its centered rolling mean, partial windows at the edges.

    Matches a centered rolling mean with `min_periods=1`: at index `i` the
    window is `[i - window//2, i + window//2]` clipped to the array's
    bounds, so the edges average over fewer observations rather than
    producing nulls.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    half = window // 2
    cumsum = np.concatenate([[0.0], np.cumsum(x)])
    result = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_mean = (cumsum[hi] - cumsum[lo]) / (hi - lo)
        result[i] = x[i] - window_mean
    return result


@dataclass(frozen=True)
class ArFit:
    """A fitted AR(p) model: `x_t = mean + sum_j phi_j (x_{t-j} - mean) + e_t`."""

    p: int
    phi: np.ndarray
    sigma2: float
    mean: float


def fit_ar_yule_walker(x: np.ndarray, max_p: int = 7) -> ArFit:
    """Fit AR(p) by Yule-Walker (Levinson-Durbin), choosing `p` by AIC.

    Computes the biased autocovariances of the demeaned series out to lag
    `max_p`, runs the Levinson-Durbin recursion to get each order's
    coefficients and innovation variance `sigma2_p`, then picks
    `p = argmin_p (n * ln(sigma2_p) + 2p)` over `p` in `0..max_p`; ties go to
    the smaller `p` (numpy's `argmin` returns the first minimum, and orders
    are evaluated smallest-first). `p = 0` is white noise: `sigma2` is the
    series' own (biased) variance.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    max_p = min(max_p, n - 1)
    mean = float(x.mean())
    demeaned = x - mean

    gamma = np.array(
        [np.sum(demeaned[: n - k] * demeaned[k:]) / n for k in range(max_p + 1)]
    )

    sigma2_by_order: list[float] = [float(gamma[0])]
    phi_by_order: list[np.ndarray] = [np.array([], dtype=float)]

    phi_prev = np.array([], dtype=float)
    sigma2_prev = gamma[0]
    for k in range(1, max_p + 1):
        if k == 1:
            reflection = gamma[1] / gamma[0]
            phi_k = np.array([reflection])
        else:
            numerator = gamma[k] - np.sum(phi_prev * gamma[k - 1 : 0 : -1])
            reflection = numerator / sigma2_prev
            phi_k = np.concatenate(
                [phi_prev - reflection * phi_prev[::-1], [reflection]]
            )
        sigma2_k = sigma2_prev * (1.0 - reflection**2)
        sigma2_by_order.append(float(sigma2_k))
        phi_by_order.append(phi_k)
        phi_prev = phi_k
        sigma2_prev = sigma2_k

    aic = [n * math.log(s) + 2 * p for p, s in enumerate(sigma2_by_order)]
    best_p = int(np.argmin(aic))

    return ArFit(
        p=best_p,
        phi=phi_by_order[best_p],
        sigma2=sigma2_by_order[best_p],
        mean=mean,
    )


def simulate_ar(
    fit: ArFit, n: int, rng: np.random.Generator, burn_in: int = 100
) -> np.ndarray:
    """Simulate `n` values from `fit`'s Gaussian-innovation AR(p), plus `fit.mean`.

    Draws `n + burn_in` innovations, runs the AR(p) recursion (missing
    early lags treated as 0, washed out by discarding `burn_in`), and
    returns the last `n` values shifted by `fit.mean`.
    """
    total = n + burn_in
    innovations = rng.normal(0.0, math.sqrt(fit.sigma2), size=total)
    if fit.p == 0:
        series = innovations
    else:
        series = np.zeros(total, dtype=float)
        for t in range(total):
            ar_part = 0.0
            for j in range(fit.p):
                lag_idx = t - j - 1
                if lag_idx >= 0:
                    ar_part += fit.phi[j] * series[lag_idx]
            series[t] = ar_part + innovations[t]
    return series[burn_in:] + fit.mean


def _replay_frame(values: np.ndarray) -> pl.DataFrame:
    n = values.size
    run_dates = [_SIM_START_DATE + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "station": [_SIM_STATION] * n,
            "run_date": run_dates,
            "lead_day": [_SIM_LEAD_DAY] * n,
            "variable": [_SIM_VARIABLE] * n,
            "source": [_RAW_SOURCE] * n,
            "target_date": run_dates,
            "scorable": [True] * n,
            "error_f": values.tolist(),
        }
    )


def replay_coverage(
    fit: ArFit,
    n: int,
    block_days: int,
    *,
    n_sims: int,
    rng: np.random.Generator,
    n_boot: int,
) -> float:
    """Share of `n_sims` simulated bootstrap CIs that contain `fit.mean`.

    Each simulation draws a length-`n` series from `fit` (PREREG "Coverage
    replay", step 2), lays it out as one row per consecutive `run_date`
    starting 2025-01-01, and scores it with `scoring.score(by=(), n_boot=
    n_boot, seed=scoring.SEED, block_days=block_days)` -- the module's own
    `CI_LEVEL` and `min_sample_dates` defaults apply.
    """
    hits = 0
    for _ in range(n_sims):
        values = simulate_ar(fit, n, rng)
        frame = _replay_frame(values)
        result = scoring.score(
            frame,
            by=(),
            n_boot=n_boot,
            seed=scoring.SEED,
            block_days=block_days,
        )
        row = result.row(0, named=True)
        lo, hi = row["bias_lo"], row["bias_hi"]
        if lo is not None and hi is not None and lo <= fit.mean <= hi:
            hits += 1
    return hits / n_sims


@dataclass(frozen=True)
class BlockChoice:
    """The decision rule's output: the chosen `BLOCK_DAYS`, or the cap outcome."""

    value: int | None
    raw_ceiling: int
    capped: bool


def choose_block_days(
    b: Mapping[tuple[str, int], tuple[float, float]], cap: int = 14
) -> BlockChoice:
    """PREREG "Decision rule": `ceil(max over keys of max(b_pooled, b_station))`.

    `b` maps each (variable, lead_day) key to `(b_pooled, b_station)`. The
    ceiling is floored at 1. If it exceeds `cap`, the rule does not choose a
    value: `value` is `None` and `capped` is `True` (rule 3's "goes back to
    Charles" outcome).
    """
    per_key_max = [max(b_pooled, b_station) for b_pooled, b_station in b.values()]
    raw_ceiling = max(1, math.ceil(max(per_key_max)))
    if raw_ceiling > cap:
        return BlockChoice(value=None, raw_ceiling=raw_ceiling, capped=True)
    return BlockChoice(value=raw_ceiling, raw_ceiling=raw_ceiling, capped=False)


def first_passing_block_days(
    coverage_at: Callable[[int], Sequence[float]],
    start: int,
    cap: int = 14,
    threshold: float = 0.90,
) -> int | None:
    """The fallback rule: first `block_days` in `[start, cap]` passing everywhere.

    `coverage_at(block_days)` returns the coverage of each replay series at
    that block length; a block length "passes" when every value is `>=
    threshold`. Returns `None` if nothing up to `cap` passes (PREREG
    "Coverage replay", fallback).
    """
    for block_days in range(start, cap + 1):
        if all(c >= threshold for c in coverage_at(block_days)):
            return block_days
    return None
