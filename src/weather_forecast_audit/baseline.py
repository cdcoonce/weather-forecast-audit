"""Rolling-bias baseline corrector (issue #18): the transparent floor an ML
challenger must beat (PRD module 7).

`BaselineModel` estimates, per (station, lead_day, variable), a rolling mean
of signed error (`forecast_f - observed_f`) over the last `window_days` of
scorable training pairs the walk-forward evaluator's leakage guard (#17)
allows it to see, and subtracts that estimate from the raw forecast. A group
with fewer than `min_pairs` of history falls back to the raw forecast,
flagged.

No I/O, no DuckDB, no dbt: a plain `weather_forecast_audit.walkforward.Model`
implementation, driven only through `walk_forward` in tests
(`tests/unit/test_baseline.py`). See docs/methodology.md ("Baseline
correction (rolling bias)") for the W and k justification.
"""

import numpy as np
import polars as pl

# Pre-registered before any scoring; never tuned on backfill skill (that
# would fit the evaluation itself). See the docstrings below and
# docs/methodology.md for the justification. Sensitivity at W=14 and W=60 is
# reported in the PR as exploratory only, never used to change these.
WINDOW_DAYS = 30
"""~1 month: short enough to follow seasonal drift (biases differ by season
at the same station, which is exactly what the audit exists to show), and
long enough that the standard error of the mean error is about
2.5°F/sqrt(30) ≈ 0.45°F under independence -- below the 0.5-1.5°F biases
being corrected."""

MIN_PAIRS = 15
"""Half of WINDOW_DAYS: a station with patchy data is corrected only when its
estimate has SE ≲ 0.65°F; otherwise it falls back to raw."""

_GROUP_KEYS = ("station", "lead_day", "variable")

_STATS_SCHEMA = {
    "station": pl.Utf8,
    "lead_day": pl.Int64,
    "variable": pl.Utf8,
    "bias_estimate_f": pl.Float64,
    "n_pairs": pl.Int64,
}


class BaselineModel:
    """Rolling per-(station, lead_day, variable) bias correction, with fallback."""

    def __init__(
        self, window_days: int = WINDOW_DAYS, min_pairs: int = MIN_PAIRS
    ) -> None:
        self.window_days = window_days
        self.min_pairs = min_pairs
        self.name = "baseline"
        self._stats: pl.DataFrame = pl.DataFrame(schema=_STATS_SCHEMA)

    def fit(self, training_pairs: pl.DataFrame) -> None:
        """Estimate each group's rolling bias over the trailing `window_days`.

        `training_pairs` is sorted ascending by `window_end_utc` (the
        evaluator's contract, build spec #18 D2), so the window start is a
        single `searchsorted` on the already-sorted array, not a full scan.
        `anchor` is the latest `window_end_utc` in the whole training set
        (across every station/lead/variable); the kept pairs are those with
        `window_end_utc > anchor - window_days` (strictly greater). An empty
        training set yields no estimates, so everything falls back.
        """
        if training_pairs.height == 0:
            self._stats = pl.DataFrame(schema=_STATS_SCHEMA)
            return

        window_end_arr = training_pairs.get_column("window_end_utc").to_numpy()
        anchor = window_end_arr[-1]
        threshold = anchor - np.timedelta64(self.window_days, "D")
        start_idx = int(np.searchsorted(window_end_arr, threshold, side="right"))
        windowed = training_pairs.slice(start_idx, training_pairs.height - start_idx)

        self._stats = (
            windowed.group_by(list(_GROUP_KEYS))
            .agg(
                bias_estimate_f=(pl.col("forecast_f") - pl.col("observed_f")).mean(),
                n_pairs=pl.len().cast(pl.Int64),
            )
        )

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        """Corrected `forecast_f`, `fallback`, `bias_estimate_f`, `n_pairs`.

        `forecast_f` is `raw - bias` when the group has `n_pairs >=
        min_pairs`, else the raw `forecast_f` unchanged. `fallback` is true
        whenever the group has no estimate or fewer than `min_pairs`.
        """
        joined = run_rows.join(self._stats, on=list(_GROUP_KEYS), how="left")
        has_estimate = pl.col("n_pairs").is_not_null()
        meets_min = has_estimate & (pl.col("n_pairs") >= self.min_pairs)

        return joined.select(
            pl.when(meets_min)
            .then(pl.col("forecast_f") - pl.col("bias_estimate_f"))
            .otherwise(pl.col("forecast_f"))
            .alias("forecast_f"),
            (~meets_min).alias("fallback"),
            pl.col("bias_estimate_f"),
            pl.col("n_pairs").fill_null(0).alias("n_pairs"),
        )

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return self.predict_detail(run_rows)["forecast_f"]
