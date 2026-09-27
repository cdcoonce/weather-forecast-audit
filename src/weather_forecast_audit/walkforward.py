"""Walk-forward evaluator with a lead-aware leakage guard (issue #17).

A correction model's measured skill means nothing if it ever sees the
future (PRD module 6). There is a subtle trap: at run *t*, a lead-3
forecast's most recently *verified* error comes from a run at least 3-4
days older, so cutting off training data by `run_date` alone leaks -- run
*t-1*'s lead-3 target is the same weather day as run *t*'s lead-2 (or
lead-1) target. The correct cutoff is the verification window's own close
time, never `run_date` or `target_date`: for the run issued at time *T*,
the training set is exactly the pairs with `scorable = true` and
`window_end_utc < T` (strict), across every lead, variable and station.
See docs/methodology.md ("Walk-forward evaluation") for the measured leak
this guards against.

No I/O, no DuckDB, no dbt: `walk_forward` takes a `polars.DataFrame`
matching `fct_forecast_verification`'s schema (dbt/models/marts) and a
`Model`, and returns a pure per-run-row prediction frame. The evaluator
itself has no randomness; determinism is inherited from the model and
polars' own stable sorts.
"""

from datetime import date, datetime
from typing import Literal, Protocol

import numpy as np
import polars as pl

Cadence = Literal["daily", "monthly"]

_REQUIRED_COLUMNS = (
    "station",
    "run_date",
    "runtime_utc",
    "lead_day",
    "variable",
    "target_date",
    "forecast_f",
    "spread_f",
    "window_start_utc",
    "window_end_utc",
    "observed_f",
    "scorable",
)

# What `predict` is allowed to see: guidance-side columns only. Never
# observed_f, error_f, hourly_observed_f, cli_f, n_obs, hours_covered,
# scorable, extreme_source or periods_found -- those are the answers.
# cycle_hour is included only when the input frame carries it.
_PREDICT_COLUMNS = (
    "station",
    "run_date",
    "runtime_utc",
    "cycle_hour",
    "lead_day",
    "variable",
    "target_date",
    "forecast_f",
    "spread_f",
    "window_start_utc",
    "window_end_utc",
)

# Sorting the scorable pairs once by this key turns the leakage cutoff into
# a single prefix slice per run (searchsorted on window_end_utc), rather
# than an O(dates x N) re-filter.
_CUTOFF_SORT_KEYS = ("window_end_utc", "station", "runtime_utc", "lead_day", "variable")
_PREDICT_ROW_SORT_KEYS = ("station", "lead_day", "variable")
_OUTPUT_SORT_KEYS = ("run_date", "station", "lead_day", "variable")
_RAW_SOURCE = "raw_nbm"

_OUTPUT_SCHEMA = {
    "station": pl.Utf8,
    "run_date": pl.Date,
    "runtime_utc": pl.Datetime("us", "UTC"),
    "lead_day": pl.Int64,
    "variable": pl.Utf8,
    "target_date": pl.Date,
    "source": pl.Utf8,
    "forecast_f": pl.Float64,
    "raw_forecast_f": pl.Float64,
    "retrained_on": pl.Date,
    "trained_through": pl.Datetime("us", "UTC"),
}


class Model(Protocol):
    """A correction model the evaluator fits and queries at each retrain."""

    name: str

    def fit(self, training_pairs: pl.DataFrame) -> None:
        """Fit on pairs with `scorable = true` and `window_end_utc < T`."""
        ...

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        """Corrected `forecast_f`, aligned to `run_rows` (same length, no nulls)."""
        ...


class RawNbmModel:
    """Reference model: passes NBM's own guidance through unmodified."""

    name = "raw_nbm"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["forecast_f"]


def _validate_columns(pairs: pl.DataFrame) -> None:
    for column in _REQUIRED_COLUMNS:
        if column not in pairs.columns:
            msg = f"missing required column: {column}"
            raise ValueError(msg)


def _issuance_times(pairs: pl.DataFrame, run_dates: list[date]) -> dict[date, datetime]:
    """Each evaluated run date's issuance time T, or raise on mixed cycles."""
    grouped = (
        pairs.filter(pl.col("run_date").is_in(run_dates))
        .group_by("run_date")
        .agg(
            pl.col("runtime_utc").n_unique().alias("_n_runtimes"),
            pl.col("runtime_utc").first().alias("_runtime"),
        )
    )
    times: dict[date, datetime] = {}
    for row in grouped.iter_rows(named=True):
        if row["_n_runtimes"] != 1:
            msg = (
                f"run_date {row['run_date']} has rows with disagreeing "
                "runtime_utc values (mixed cycles)"
            )
            raise ValueError(msg)
        times[row["run_date"]] = row["_runtime"]
    return times


def _retrain_dates(run_dates: list[date], retrain: Cadence) -> set[date]:
    if retrain == "daily":
        return set(run_dates)
    if retrain == "monthly":
        seen_months: set[tuple[int, int]] = set()
        dates: set[date] = set()
        for run_date_ in run_dates:
            month_key = (run_date_.year, run_date_.month)
            if month_key not in seen_months:
                seen_months.add(month_key)
                dates.add(run_date_)
        return dates
    msg = f"retrain must be 'daily' or 'monthly', got {retrain!r}"
    raise ValueError(msg)


def _empty_result() -> pl.DataFrame:
    return pl.DataFrame(schema=_OUTPUT_SCHEMA)


def walk_forward(
    pairs: pl.DataFrame,
    model: Model,
    start: date,
    end: date,
    retrain: Cadence = "monthly",
) -> pl.DataFrame:
    """Walk forward through issuance dates in `[start, end]`, guarding leakage.

    For the run issued at `T`, `model.fit` sees exactly the pairs with
    `scorable = true` and `window_end_utc < T` (strict), across all leads,
    variables and stations -- computed as a prefix slice of the scorable
    pairs sorted once by (window_end_utc, station, runtime_utc, lead_day,
    variable), via `searchsorted`. `model.predict` sees every run row
    (scorable or not), restricted to an allow-list of guidance-side
    columns, sorted by (station, lead_day, variable).

    Retraining: `"daily"` refits before every evaluated run date;
    `"monthly"` refits on the first evaluated run date of each calendar
    month (including the first evaluated date overall). Between refits,
    the model fitted at the last retrain predicts.

    Raises `ValueError` for a missing required column, a run date whose
    rows disagree on `runtime_utc`, `start > end`, or a `predict` result
    that is the wrong length or contains nulls.
    """
    if start > end:
        msg = f"start must be <= end, got start={start} end={end}"
        raise ValueError(msg)

    _validate_columns(pairs)

    if "source" in pairs.columns:
        pairs = pairs.filter(pl.col("source") == _RAW_SOURCE)

    run_dates = sorted(
        pairs.filter((pl.col("run_date") >= start) & (pl.col("run_date") <= end))
        .get_column("run_date")
        .unique()
        .to_list()
    )
    if not run_dates:
        return _empty_result()

    issuance_t = _issuance_times(pairs, run_dates)
    retrain_dates = _retrain_dates(run_dates, retrain)

    scorable_sorted = pairs.filter(pl.col("scorable")).sort(list(_CUTOFF_SORT_KEYS))
    window_end_arr = scorable_sorted.get_column("window_end_utc").to_numpy()
    predict_columns = [c for c in _PREDICT_COLUMNS if c in pairs.columns]

    last_retrain_date: date | None = None
    trained_through: datetime | None = None
    outputs: list[pl.DataFrame] = []

    for run_date_ in run_dates:
        t = issuance_t[run_date_]

        if run_date_ in retrain_dates:
            cutoff_idx = int(
                np.searchsorted(window_end_arr, np.datetime64(t.replace(tzinfo=None)))
            )
            training_pairs = scorable_sorted.slice(0, cutoff_idx)
            model.fit(training_pairs)
            last_retrain_date = run_date_
            trained_through = (
                training_pairs.get_column("window_end_utc")[-1]
                if cutoff_idx > 0
                else None
            )

        run_rows = pairs.filter(pl.col("run_date") == run_date_).sort(
            list(_PREDICT_ROW_SORT_KEYS)
        )
        predictions = model.predict(run_rows.select(predict_columns))

        if predictions.len() != run_rows.height:
            msg = (
                f"predict returned {predictions.len()} rows for run_date "
                f"{run_date_}, expected {run_rows.height}"
            )
            raise ValueError(msg)
        if predictions.null_count() > 0:
            msg = f"predict returned null values for run_date {run_date_}"
            raise ValueError(msg)

        outputs.append(
            run_rows.select(
                "station",
                "run_date",
                "runtime_utc",
                "lead_day",
                "variable",
                "target_date",
                pl.col("forecast_f").alias("raw_forecast_f"),
            ).with_columns(
                pl.Series("forecast_f", predictions, dtype=pl.Float64),
                source=pl.lit(model.name),
                retrained_on=pl.lit(last_retrain_date),
                trained_through=pl.lit(trained_through, dtype=pl.Datetime("us", "UTC")),
            )
        )

    result = pl.concat(outputs).select(list(_OUTPUT_SCHEMA))
    return result.sort(list(_OUTPUT_SORT_KEYS))
