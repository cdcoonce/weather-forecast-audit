"""Tests for the walk-forward evaluator and its lead-aware leakage guard.

Test 1 is the headline test: it proves a cheating model (memorizing the
observed value for a target it has already seen) gains no advantage over
raw NBM under the correct `window_end_utc < T` cutoff, and -- as a reverse
check built independently in this file -- that the same cheat *does*
reduce MAE under a naive `run_date < current run_date` cutoff on the same
data. That is the leak the evaluator exists to close: a lead-3 target from
run *t-1* is the same weather day as a lead-2 (or lead-1) target from run
*t*, so cutting training data off by issuance date alone hands a memorizing
model tomorrow's answer.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from weather_forecast_audit.walkforward import (
    Model,
    RawNbmModel,
    walk_forward,
)

pytestmark = pytest.mark.unit

_PREDICT_COLUMNS = frozenset(
    {
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
    }
)

_LEADS_BY_VARIABLE = {"max": (1, 2), "min": (1, 2, 3)}


# -- synthetic data helper ----------------------------------------------


def _window_max(target: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time(12, 0), tzinfo=UTC)
    end = datetime.combine(target + timedelta(days=1), time(6, 0), tzinfo=UTC)
    return start, end


def _window_min(target: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time(0, 0), tzinfo=UTC)
    end = datetime.combine(target, time(18, 0), tzinfo=UTC)
    return start, end


def _consecutive_run_dates(start_date: date, n: int) -> list[date]:
    return [start_date + timedelta(days=i) for i in range(n)]


def _synthetic_pairs(
    run_dates: Sequence[date],
    stations: Sequence[str] = ("KPHX", "KDEN", "KORD"),
    *,
    seed: int = 0,
    cycle_hour: int | Callable[[date], int] = 13,
    unscorable_frac: float = 0.0,
) -> pl.DataFrame:
    """Realistic `fct_forecast_verification` pairs, max lead 1-2, min lead 1-3.

    `observed_f` for a given (station, target_date, variable) is drawn once
    from a seeded RNG and shared across every run/lead that verifies that
    same weather day -- a lead-3 row from run t-1 and a lead-2 row from run
    t verify the *same* target, so they must agree on what actually
    happened. `forecast_f` adds noise plus a fixed per-station bias on top
    of that shared truth. Some rows are marked unscorable (`scorable =
    false`, `observed_f` null) per `unscorable_frac`; `forecast_f` is still
    populated, since NBM guidance exists whether or not the verification
    window could later be tiled.
    """
    rng = np.random.default_rng(seed)
    station_bias = {
        station: (i - (len(stations) - 1) / 2) * 1.5
        for i, station in enumerate(stations)
    }

    target_dates = sorted(
        {
            run_date_ + timedelta(days=lead)
            for run_date_ in run_dates
            for leads in _LEADS_BY_VARIABLE.values()
            for lead in leads
        }
    )
    true_observed = {
        (station, target, variable): float(rng.normal(60.0, 10.0))
        for station in stations
        for target in target_dates
        for variable in _LEADS_BY_VARIABLE
    }

    records: list[dict[str, object]] = []
    for run_date_ in run_dates:
        hour = cycle_hour(run_date_) if callable(cycle_hour) else cycle_hour
        runtime = datetime.combine(run_date_, time(hour, 0), tzinfo=UTC)
        for station in stations:
            for variable, leads in _LEADS_BY_VARIABLE.items():
                for lead in leads:
                    target = run_date_ + timedelta(days=lead)
                    window_start, window_end = (
                        _window_max(target)
                        if variable == "max"
                        else _window_min(target)
                    )
                    observed = true_observed[(station, target, variable)]
                    noise = float(rng.normal(0.0, 1.0))
                    forecast = observed + noise + station_bias[station]
                    scorable = bool(rng.random() >= unscorable_frac)
                    records.append(
                        {
                            "station": station,
                            "run_date": run_date_,
                            "runtime_utc": runtime,
                            "cycle_hour": hour,
                            "lead_day": lead,
                            "variable": variable,
                            "target_date": target,
                            "forecast_f": forecast,
                            "spread_f": 3.0,
                            "window_start_utc": window_start,
                            "window_end_utc": window_end,
                            "observed_f": observed if scorable else None,
                            "scorable": scorable,
                            "source": "raw_nbm",
                        }
                    )
    return pl.DataFrame(records)


def _mae(predictions: pl.DataFrame, scorable_pairs: pl.DataFrame) -> float:
    joined = predictions.join(
        scorable_pairs.select(
            ["station", "run_date", "lead_day", "variable", "observed_f"]
        ),
        on=["station", "run_date", "lead_day", "variable"],
        how="inner",
    )
    errors = (joined["forecast_f"] - joined["observed_f"]).abs()
    return float(errors.mean())


# -- test-only models -----------------------------------------------------


class CheatingModel:
    """Memorizes observed_f by (station, target_date, variable); else raw."""

    name = "cheating"

    def __init__(self) -> None:
        self._memo: dict[tuple[str, date, str], float] = {}

    def fit(self, training_pairs: pl.DataFrame) -> None:
        self._memo = {
            (row["station"], row["target_date"], row["variable"]): row["observed_f"]
            for row in training_pairs.filter(
                pl.col("observed_f").is_not_null()
            ).iter_rows(named=True)
        }

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        values = [
            self._memo.get(
                (row["station"], row["target_date"], row["variable"]),
                row["forecast_f"],
            )
            for row in run_rows.iter_rows(named=True)
        ]
        return pl.Series(values, dtype=pl.Float64)


def _leaky_walk_forward(
    pairs: pl.DataFrame, model: Model, start: date, end: date
) -> pl.DataFrame:
    """Reverse check for test 1: cutoff by `run_date`, not `window_end_utc < T`.

    A deliberately wrong evaluator, built independently of
    `weather_forecast_audit.walkforward`, to prove the cheat has teeth: it
    slices training data by `run_date < current run_date`, which lets a
    lead-3 target from run t-1 leak into training for run t (the same
    weather day as run t's own lead-2 target).
    """
    scorable = pairs.filter(pl.col("scorable"))
    run_dates = sorted(
        pairs.filter((pl.col("run_date") >= start) & (pl.col("run_date") <= end))
        .get_column("run_date")
        .unique()
        .to_list()
    )
    outputs = []
    for run_date_ in run_dates:
        training = scorable.filter(pl.col("run_date") < run_date_)
        model.fit(training)
        run_rows = pairs.filter(pl.col("run_date") == run_date_).sort(
            ["station", "lead_day", "variable"]
        )
        predictions = model.predict(run_rows)
        outputs.append(
            run_rows.select(
                "station", "run_date", "lead_day", "variable", "target_date"
            ).with_columns(pl.Series("forecast_f", predictions))
        )
    return pl.concat(outputs)


class SpyModel:
    """Records every training set it is given, for closure assertions."""

    name = "spy"

    def __init__(self) -> None:
        self.fit_calls: list[pl.DataFrame] = []

    def fit(self, training_pairs: pl.DataFrame) -> None:
        self.fit_calls.append(training_pairs)

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["forecast_f"]


class _CountingRawModel:
    name = "raw_nbm"

    def __init__(self) -> None:
        self.fit_count = 0

    def fit(self, training_pairs: pl.DataFrame) -> None:
        self.fit_count += 1

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["forecast_f"]


class ColumnRecordingModel:
    name = "recorder"

    def __init__(self) -> None:
        self.seen_columns: frozenset[str] | None = None

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        self.seen_columns = frozenset(run_rows.columns)
        return run_rows["forecast_f"]


class LeakyPredictModel:
    name = "leaky_predict"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["observed_f"]


class ShortOutputModel:
    name = "short_output"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["forecast_f"].head(1)


class NullOutputModel:
    name = "null_output"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        values = run_rows["forecast_f"].to_list()
        values[0] = None
        return pl.Series(values, dtype=pl.Float64)


# -- 1. cheating model gains nothing; leaky cutoff has teeth ---------------


def test_cheating_model_gains_nothing_but_leaky_cutoff_helps() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 60)
    pairs = _synthetic_pairs(run_dates, seed=42)
    scorable = pairs.filter(pl.col("scorable"))

    raw_result = walk_forward(
        pairs, RawNbmModel(), start=run_dates[10], end=run_dates[-1], retrain="daily"
    )
    cheat_result = walk_forward(
        pairs, CheatingModel(), start=run_dates[10], end=run_dates[-1], retrain="daily"
    )

    sort_keys = ["run_date", "station", "lead_day", "variable"]
    assert cheat_result.sort(sort_keys)["forecast_f"].equals(
        raw_result.sort(sort_keys)["forecast_f"]
    )

    raw_mae = _mae(raw_result, scorable)
    cheat_mae = _mae(cheat_result, scorable)
    assert cheat_mae == pytest.approx(raw_mae, abs=1e-9)

    # Reverse check: the same cheat, under a run_date cutoff, leaks and
    # measurably reduces MAE below raw's.
    leaky_result = _leaky_walk_forward(
        pairs, CheatingModel(), start=run_dates[10], end=run_dates[-1]
    )
    leaky_mae = _mae(leaky_result, scorable)
    assert leaky_mae < raw_mae - 0.5, (
        f"leaky run_date cutoff should let the cheat reduce MAE: "
        f"leaky={leaky_mae:.4f} raw={raw_mae:.4f}"
    )


# -- 2. lead-3 closure and the exact-boundary edge case ---------------------


def test_lead3_closure_and_exact_boundary_excluded() -> None:
    run_dates = _consecutive_run_dates(date(2026, 2, 1), 40)
    pairs = _synthetic_pairs(run_dates, stations=("KPHX", "KDEN"), seed=1)

    t = date(2026, 2, 20)
    t_index = run_dates.index(t)
    t_time = datetime.combine(t, time(13, 0), tzinfo=UTC)

    # A hand-added pair whose window closes exactly at t's own T. Real
    # windows (06Z/18Z close, 13Z issuance) never coincide like this, but
    # it is a legal input, and the cutoff must still exclude it (strict <).
    boundary_row = {
        "station": "KPHX",
        "run_date": date(2026, 2, 10),
        "runtime_utc": datetime.combine(date(2026, 2, 10), time(13, 0), tzinfo=UTC),
        "cycle_hour": 13,
        "lead_day": 1,
        "variable": "min",
        "target_date": t,
        "forecast_f": 54.0,
        "spread_f": 3.0,
        "window_start_utc": t_time - timedelta(hours=18),
        "window_end_utc": t_time,
        "observed_f": 55.0,
        "scorable": True,
        "source": "raw_nbm",
    }
    pairs = pl.concat([pairs, pl.DataFrame([boundary_row])], how="vertical")

    spy = SpyModel()
    walk_forward(pairs, spy, start=run_dates[0], end=run_dates[-1], retrain="daily")

    training_at_t = spy.fit_calls[t_index]
    training_at_t_plus_1 = spy.fit_calls[t_index + 1]

    # Boundary row excluded exactly at T, included once T has passed it.
    assert training_at_t.filter(pl.col("window_end_utc") == t_time).height == 0
    assert (
        training_at_t_plus_1.filter(pl.col("window_end_utc") == t_time).height == 1
    )

    # Lead-3 min pair from run t-1 (target t+2, window closing t+2 18Z):
    # absent from training at t.
    lead3_leak = training_at_t.filter(
        (pl.col("run_date") == t - timedelta(days=1))
        & (pl.col("lead_day") == 3)
        & (pl.col("variable") == "min")
    )
    assert lead3_leak.height == 0

    # Lead-2 max pair from run t-1 (target t+1): also absent.
    lead2_max_leak = training_at_t.filter(
        (pl.col("run_date") == t - timedelta(days=1))
        & (pl.col("lead_day") == 2)
        & (pl.col("variable") == "max")
    )
    assert lead2_max_leak.height == 0

    # A pair whose window closed at t 06Z (max, target t-1, e.g. run t-2
    # lead 1 or run t-3 lead 2) is present.
    closed_at_t_06z = training_at_t.filter(
        pl.col("window_end_utc") == datetime.combine(t, time(6, 0), tzinfo=UTC)
    )
    assert closed_at_t_06z.height > 0

    # max(window_end_utc) < T holds for every recorded fit.
    for fit_index, training_pairs in enumerate(spy.fit_calls):
        run_date_ = run_dates[fit_index]
        max_window_end = training_pairs.get_column("window_end_utc").max()
        if max_window_end is not None:
            assert max_window_end < datetime.combine(
                run_date_, time(13, 0), tzinfo=UTC
            )


# -- 3. retrain cadence -----------------------------------------------------


def test_retrain_cadence_monthly_and_daily_with_gap() -> None:
    run_dates = [
        date(2026, 1, 28),
        date(2026, 1, 29),
        date(2026, 1, 30),
        date(2026, 1, 31),
        # 2026-02-01 missing: simply not evaluated.
        date(2026, 2, 2),
        date(2026, 2, 3),
        date(2026, 2, 4),
    ]
    pairs = _synthetic_pairs(run_dates, seed=2)

    monthly_model = _CountingRawModel()
    monthly_result = walk_forward(
        pairs,
        monthly_model,
        start=run_dates[0],
        end=run_dates[-1],
        retrain="monthly",
    )
    assert monthly_model.fit_count == 2
    for row in monthly_result.iter_rows(named=True):
        expected = date(2026, 1, 28) if row["run_date"].month == 1 else date(2026, 2, 2)
        assert row["retrained_on"] == expected

    daily_model = _CountingRawModel()
    daily_result = walk_forward(
        pairs, daily_model, start=run_dates[0], end=run_dates[-1], retrain="daily"
    )
    assert daily_model.fit_count == len(run_dates)
    for row in daily_result.iter_rows(named=True):
        assert row["retrained_on"] == row["run_date"]


# -- 4. property: trained_through < runtime_utc -----------------------------


def test_property_trained_through_before_runtime_utc() -> None:
    rng = np.random.default_rng(999)
    checked_any = False
    for _trial in range(25):
        n_stations = int(rng.integers(1, 6))
        stations = tuple(f"S{i:02d}" for i in range(n_stations))
        n_dates = int(rng.integers(20, 70))
        all_dates = _consecutive_run_dates(date(2026, 1, 1), n_dates)
        keep_mask = rng.random(n_dates) >= 0.15
        run_dates = [d for d, keep in zip(all_dates, keep_mask, strict=True) if keep]
        if len(run_dates) < 2:
            continue
        unscorable_frac = float(rng.uniform(0.0, 0.3))
        cadence = "daily" if rng.random() < 0.5 else "monthly"
        pairs = _synthetic_pairs(
            run_dates,
            stations=stations,
            seed=int(rng.integers(0, 1_000_000)),
            unscorable_frac=unscorable_frac,
        )

        result = walk_forward(
            pairs,
            RawNbmModel(),
            start=run_dates[0],
            end=run_dates[-1],
            retrain=cadence,
        )
        checked_any = True

        non_null = result.filter(pl.col("trained_through").is_not_null())
        assert (non_null["trained_through"] < non_null["runtime_utc"]).all()

        first_date_rows = result.filter(pl.col("run_date") == run_dates[0])
        assert first_date_rows["trained_through"].is_null().all()
    assert checked_any


# -- 5. determinism ----------------------------------------------------------


def test_determinism_bit_identical_across_runs() -> None:
    run_dates = _consecutive_run_dates(date(2026, 3, 1), 45)
    pairs = _synthetic_pairs(run_dates, seed=5)

    first = walk_forward(
        pairs, RawNbmModel(), start=run_dates[0], end=run_dates[-1], retrain="monthly"
    )
    second = walk_forward(
        pairs, RawNbmModel(), start=run_dates[0], end=run_dates[-1], retrain="monthly"
    )
    assert first.equals(second)


# -- 6. predict never sees answer columns ------------------------------------


def test_predict_receives_exactly_the_allowed_columns() -> None:
    run_dates = _consecutive_run_dates(date(2026, 3, 1), 10)
    pairs = _synthetic_pairs(run_dates, seed=3)

    recorder = ColumnRecordingModel()
    walk_forward(pairs, recorder, start=run_dates[0], end=run_dates[-1])

    assert recorder.seen_columns == _PREDICT_COLUMNS


def test_predict_reading_observed_f_raises_column_not_found() -> None:
    run_dates = _consecutive_run_dates(date(2026, 3, 1), 10)
    pairs = _synthetic_pairs(run_dates, seed=3)

    with pytest.raises(ColumnNotFoundError):
        walk_forward(pairs, LeakyPredictModel(), start=run_dates[0], end=run_dates[-1])


# -- 7. validation -------------------------------------------------------------


def test_missing_required_column_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=4).drop("spread_f")
    with pytest.raises(ValueError, match="spread_f"):
        walk_forward(pairs, RawNbmModel(), start=run_dates[0], end=run_dates[-1])


def test_mixed_runtime_utc_within_run_date_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=5)
    bad_run_date = run_dates[2]
    bad = pairs.with_columns(
        pl.when(
            (pl.col("run_date") == bad_run_date)
            & (pl.col("station") == "KPHX")
            & (pl.col("lead_day") == 1)
            & (pl.col("variable") == "max")
        )
        .then(pl.col("runtime_utc") + pl.duration(hours=1))
        .otherwise(pl.col("runtime_utc"))
        .alias("runtime_utc")
    )
    with pytest.raises(ValueError, match="runtime_utc"):
        walk_forward(bad, RawNbmModel(), start=run_dates[0], end=run_dates[-1])


def test_predict_wrong_length_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=6)
    with pytest.raises(ValueError):
        walk_forward(pairs, ShortOutputModel(), start=run_dates[0], end=run_dates[-1])


def test_predict_null_output_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=7)
    with pytest.raises(ValueError):
        walk_forward(pairs, NullOutputModel(), start=run_dates[0], end=run_dates[-1])


def test_start_after_end_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=8)
    with pytest.raises(ValueError):
        walk_forward(
            pairs, RawNbmModel(), start=run_dates[-1], end=run_dates[0]
        )


# -- 8. canonical-cycle changeover --------------------------------------------


def _canonical_cycle_hour(run_date_: date) -> int:
    return 13 if run_date_ <= date(2026, 4, 29) else 12


def test_training_pairs_are_sorted_by_window_end_utc() -> None:
    """Contract (build spec #18 D2): `fit` always sees an ascending prefix."""
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 20)
    pairs = _synthetic_pairs(run_dates, seed=11)

    spy = SpyModel()
    walk_forward(pairs, spy, start=run_dates[5], end=run_dates[-1], retrain="daily")

    assert spy.fit_calls
    for training_pairs in spy.fit_calls:
        window_ends = training_pairs.get_column("window_end_utc").to_list()
        assert window_ends == sorted(window_ends)


# -- 9. predict_detail protocol (build spec #18 D2) --------------------------


class DetailModel:
    """Uses `predict_detail`; flags every other row as a fallback."""

    name = "detail"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        n = run_rows.height
        fallback = [i % 2 == 0 for i in range(n)]
        return pl.DataFrame(
            {
                "forecast_f": run_rows["forecast_f"],
                "fallback": pl.Series(fallback, dtype=pl.Boolean),
                "bias_estimate_f": pl.Series([1.23] * n, dtype=pl.Float64),
                "n_pairs": pl.Series([5] * n, dtype=pl.Int64),
            }
        )


class RecordingDetailModel:
    """Records whether `predict` or `predict_detail` was actually called."""

    name = "recording_detail"

    def __init__(self) -> None:
        self.predict_called = False
        self.predict_detail_called = False

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        self.predict_called = True
        return run_rows["forecast_f"]

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        self.predict_detail_called = True
        return pl.DataFrame(
            {
                "forecast_f": run_rows["forecast_f"],
                "fallback": pl.Series([False] * run_rows.height, dtype=pl.Boolean),
            }
        )


class DetailWrongLengthModel:
    name = "detail_wrong_length"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "forecast_f": run_rows["forecast_f"].head(1),
                "fallback": pl.Series([False], dtype=pl.Boolean),
            }
        )


class DetailNullForecastModel:
    name = "detail_null_forecast"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        values = run_rows["forecast_f"].to_list()
        values[0] = None
        return pl.DataFrame(
            {
                "forecast_f": pl.Series(values, dtype=pl.Float64),
                "fallback": pl.Series([False] * run_rows.height, dtype=pl.Boolean),
            }
        )


class DetailNullFallbackModel:
    name = "detail_null_fallback"

    def fit(self, training_pairs: pl.DataFrame) -> None:
        return None

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        fallback = [False] * run_rows.height
        fallback[0] = None
        return pl.DataFrame(
            {
                "forecast_f": run_rows["forecast_f"],
                "fallback": pl.Series(fallback, dtype=pl.Boolean),
            }
        )


def test_output_carries_fallback_column_raw_nbm_all_false() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 10)
    pairs = _synthetic_pairs(run_dates, seed=12)

    result = walk_forward(
        pairs, RawNbmModel(), start=run_dates[0], end=run_dates[-1]
    )

    assert "fallback" in result.columns
    assert result["fallback"].dtype == pl.Boolean
    assert result["fallback"].null_count() == 0
    assert (~result["fallback"]).all()


def test_predict_detail_is_used_instead_of_predict_when_present() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 10)
    pairs = _synthetic_pairs(run_dates, seed=13)

    model = RecordingDetailModel()
    result = walk_forward(pairs, model, start=run_dates[0], end=run_dates[-1])

    assert model.predict_detail_called
    assert not model.predict_called
    assert (~result["fallback"]).all()


def test_predict_detail_fallback_values_flow_through() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 10)
    pairs = _synthetic_pairs(run_dates, seed=14, stations=("KPHX",))

    result = walk_forward(
        pairs, DetailModel(), start=run_dates[0], end=run_dates[-1]
    )

    assert set(result["fallback"].to_list()) == {True, False}
    assert "bias_estimate_f" not in result.columns
    assert "n_pairs" not in result.columns


def test_predict_detail_wrong_length_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=15)
    with pytest.raises(ValueError):
        walk_forward(
            pairs, DetailWrongLengthModel(), start=run_dates[0], end=run_dates[-1]
        )


def test_predict_detail_null_forecast_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=16)
    with pytest.raises(ValueError):
        walk_forward(
            pairs, DetailNullForecastModel(), start=run_dates[0], end=run_dates[-1]
        )


def test_predict_detail_null_fallback_raises() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _synthetic_pairs(run_dates, seed=17)
    with pytest.raises(ValueError):
        walk_forward(
            pairs, DetailNullFallbackModel(), start=run_dates[0], end=run_dates[-1]
        )


# -- 10. canonical-cycle changeover --------------------------------------------


def test_canonical_cycle_changeover() -> None:
    run_dates = _consecutive_run_dates(date(2026, 4, 25), 10)
    pairs = _synthetic_pairs(run_dates, seed=9, cycle_hour=_canonical_cycle_hour)

    spy = SpyModel()
    walk_forward(pairs, spy, start=run_dates[0], end=run_dates[-1], retrain="daily")

    changeover_date = date(2026, 4, 30)
    idx = run_dates.index(changeover_date)
    training_at_changeover = spy.fit_calls[idx]

    # Max window closing 04-30 06Z: included (06Z < 12Z issuance).
    included = training_at_changeover.filter(
        pl.col("window_end_utc")
        == datetime.combine(changeover_date, time(6, 0), tzinfo=UTC)
    )
    assert included.height > 0

    # Min window closing 04-30 18Z: excluded (18Z is after 12Z issuance).
    excluded = training_at_changeover.filter(
        pl.col("window_end_utc")
        == datetime.combine(changeover_date, time(18, 0), tzinfo=UTC)
    )
    assert excluded.height == 0

    # The 04-29 run (13Z) still used its own T.
    idx_before = run_dates.index(date(2026, 4, 29))
    training_before = spy.fit_calls[idx_before]
    max_window_end = training_before.get_column("window_end_utc").max()
    if max_window_end is not None:
        assert max_window_end < datetime.combine(
            date(2026, 4, 29), time(13, 0), tzinfo=UTC
        )
