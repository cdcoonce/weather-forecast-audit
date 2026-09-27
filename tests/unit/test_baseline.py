"""Tests for the rolling-bias baseline corrector (build spec #18).

`BaselineModel` is a `Model` (walk-forward Protocol) that removes a
per-(station, lead_day, variable) rolling mean of signed error, estimated
from the last `window_days` of scorable training pairs, falling back to raw
guidance when a group has fewer than `min_pairs`. Every test here drives it
through the real `walk_forward` evaluator (issue #17), never calling `fit`/
`predict_detail` directly, so the leakage guard and the rolling window are
both exercised together.
"""

from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import polars as pl
import pytest

from weather_forecast_audit.baseline import MIN_PAIRS, WINDOW_DAYS, BaselineModel
from weather_forecast_audit.walkforward import walk_forward

pytestmark = pytest.mark.unit

_LEADS_BY_VARIABLE = {"max": (1, 2), "min": (1, 2, 3)}


# -- synthetic data helper (mirrors test_walkforward.py's, noise-free) -------


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


def _noise_free_pairs(
    run_dates: list[date],
    stations: tuple[str, ...] = ("KPHX",),
    *,
    bias_f: float | dict[date, float] = 2.0,
    cycle_hour: int = 13,
) -> pl.DataFrame:
    """Realistic verification pairs, `forecast_f = observed_f + bias_f`, no noise.

    `bias_f` may be a constant, or a mapping from `run_date` to a step-change
    bias in effect for that run's forecasts (test 2, the rolling-window
    guard). `observed_f` for a given (station, target_date, variable) is
    shared across every run/lead that verifies that target, exactly as in
    `test_walkforward.py`'s helper.
    """
    rng = np.random.default_rng(0)
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
        runtime = datetime.combine(run_date_, time(cycle_hour, 0), tzinfo=UTC)
        bias = bias_f[run_date_] if isinstance(bias_f, dict) else bias_f
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
                    forecast = observed + bias
                    records.append(
                        {
                            "station": station,
                            "run_date": run_date_,
                            "runtime_utc": runtime,
                            "cycle_hour": cycle_hour,
                            "lead_day": lead,
                            "variable": variable,
                            "target_date": target,
                            "forecast_f": forecast,
                            "spread_f": 3.0,
                            "window_start_utc": window_start,
                            "window_end_utc": window_end,
                            "observed_f": observed,
                            "scorable": True,
                            "source": "raw_nbm",
                        }
                    )
    return pl.DataFrame(records)


class SpyModel:
    """Records every training set it is given (mirrors test_walkforward.py's)."""

    name = "spy"

    def __init__(self) -> None:
        self.fit_calls: list[pl.DataFrame] = []

    def fit(self, training_pairs: pl.DataFrame) -> None:
        self.fit_calls.append(training_pairs)

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return run_rows["forecast_f"]


def _corrected_errors(result: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    return result.join(
        pairs.select(["station", "run_date", "lead_day", "variable", "observed_f"]),
        on=["station", "run_date", "lead_day", "variable"],
        how="inner",
    ).with_columns(error_f=(pl.col("forecast_f") - pl.col("observed_f")))


# -- 1. constant bias removed -------------------------------------------------


def test_constant_bias_removed_once_enough_history() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 90)
    pairs = _noise_free_pairs(run_dates, bias_f=2.0)

    result = walk_forward(
        pairs,
        BaselineModel(),
        start=run_dates[0],
        end=run_dates[-1],
        retrain="daily",
    )

    non_fallback = result.filter(~pl.col("fallback"))
    assert non_fallback.height > 0

    errors = _corrected_errors(non_fallback, pairs)["error_f"].to_numpy()
    assert errors == pytest.approx(np.zeros_like(errors), abs=1e-9)

    # Once a group has reached min_pairs, every later prediction for that
    # group is non-fallback: a rolling window over a constant-rate,
    # noise-free daily series only gains pairs for a fixed group, so once
    # it clears k it never drops back below it.
    for _key, group in result.group_by(
        ["station", "lead_day", "variable"], maintain_order=True
    ):
        rows = group.sort("run_date")
        fallback_flags = rows["fallback"].to_list()
        first_non_fallback = next(
            (i for i, f in enumerate(fallback_flags) if not f), None
        )
        if first_non_fallback is not None:
            assert not any(fallback_flags[first_non_fallback:])


# -- 2. rolling, not expanding ------------------------------------------------


def test_rolling_window_tracks_a_step_change_not_the_expanding_mean() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 90)
    step_date = date(2026, 2, 15)
    assert step_date in run_dates

    bias_by_date = {
        run_date_: (2.0 if run_date_ < step_date else -1.0) for run_date_ in run_dates
    }
    pairs = _noise_free_pairs(run_dates, bias_f=bias_by_date)

    result = walk_forward(
        pairs,
        BaselineModel(),
        start=run_dates[0],
        end=run_dates[-1],
        retrain="daily",
    )

    # A parallel spy run over the identical pairs/schedule captures the
    # exact training_pairs BaselineModel.fit received at each retrain date,
    # so the expected bias can be derived independently (plain arithmetic
    # over that frame), without calling BaselineModel's own code.
    spy = SpyModel()
    walk_forward(
        pairs, spy, start=run_dates[0], end=run_dates[-1], retrain="daily"
    )

    def _group_expected_bias(fit_call_index: int) -> float:
        training = spy.fit_calls[fit_call_index]
        anchor = training["window_end_utc"].max()
        threshold = anchor - timedelta(days=WINDOW_DAYS)
        windowed = training.filter(
            (pl.col("station") == "KPHX")
            & (pl.col("variable") == "min")
            & (pl.col("lead_day") == 1)
            & (pl.col("window_end_utc") > threshold)
        )
        return float((windowed["forecast_f"] - windowed["observed_f"]).mean())

    # A run whose training window contains only post-step pairs corrects by
    # (applies a bias correction of) exactly -1.0, the pure post-step bias --
    # not some blend with the +2.0 pre-step history, which an *expanding*
    # mean would still be carrying at this point.
    late_run = run_dates[-1]
    late_idx = run_dates.index(late_run)
    late_rows = result.filter(
        (pl.col("run_date") == late_run)
        & (pl.col("station") == "KPHX")
        & (pl.col("variable") == "min")
        & (pl.col("lead_day") == 1)
        & (~pl.col("fallback"))
    )
    assert late_rows.height == 1
    late_correction = float(
        late_rows["raw_forecast_f"][0] - late_rows["forecast_f"][0]
    )
    assert late_correction == pytest.approx(-1.0, abs=1e-9)
    assert _group_expected_bias(late_idx) == pytest.approx(-1.0, abs=1e-9)

    # A run whose window straddles the step corrects by (applies a bias
    # correction of) the exact mixed mean, computed independently above from
    # the raw training pairs -- not the pure post-step bias alone, which a
    # correctly-windowed rolling estimate should not yet have converged to.
    straddle_run = step_date + timedelta(days=2)
    straddle_idx = run_dates.index(straddle_run)
    straddle_rows = result.filter(
        (pl.col("run_date") == straddle_run)
        & (pl.col("station") == "KPHX")
        & (pl.col("variable") == "min")
        & (pl.col("lead_day") == 1)
        & (~pl.col("fallback"))
    )
    assert straddle_rows.height == 1
    straddle_correction = float(
        straddle_rows["raw_forecast_f"][0] - straddle_rows["forecast_f"][0]
    )
    expected_bias = _group_expected_bias(straddle_idx)
    assert straddle_correction == pytest.approx(expected_bias, abs=1e-9)
    # The mixed mean must actually differ from both pure regimes, or this
    # test would not distinguish a rolling window from an expanding one.
    assert expected_bias != pytest.approx(-1.0, abs=1e-6)
    assert expected_bias != pytest.approx(2.0, abs=1e-6)


# -- 3. fallback boundary ------------------------------------------------------


def _build_group_history(n_history: int, eval_date: date) -> pl.DataFrame:
    """`n_history` scorable (KPHX, min, lead 1) training pairs, all safely
    inside `eval_date`'s issuance cutoff (and within `WINDOW_DAYS`), plus the
    evaluated run's own (unscorable, not-yet-verified) row for that group.
    """
    eval_runtime = datetime.combine(eval_date, time(13, 0), tzinfo=UTC)
    records: list[dict[str, object]] = []
    for i in range(n_history):
        target = eval_date - timedelta(days=i + 1)
        window_start, window_end = _window_min(target)
        observed = 60.0 + i
        records.append(
            {
                "station": "KPHX",
                "run_date": target - timedelta(days=1),
                "runtime_utc": datetime.combine(
                    target - timedelta(days=1), time(13, 0), tzinfo=UTC
                ),
                "cycle_hour": 13,
                "lead_day": 1,
                "variable": "min",
                "target_date": target,
                "forecast_f": observed + 2.0,
                "spread_f": 3.0,
                "window_start_utc": window_start,
                "window_end_utc": window_end,
                "observed_f": observed,
                "scorable": True,
                "source": "raw_nbm",
            }
        )
    eval_target = eval_date + timedelta(days=1)
    window_start, window_end = _window_min(eval_target)
    records.append(
        {
            "station": "KPHX",
            "run_date": eval_date,
            "runtime_utc": eval_runtime,
            "cycle_hour": 13,
            "lead_day": 1,
            "variable": "min",
            "target_date": eval_target,
            "forecast_f": 65.0,
            "spread_f": 3.0,
            "window_start_utc": window_start,
            "window_end_utc": window_end,
            "observed_f": None,
            "scorable": False,
            "source": "raw_nbm",
        }
    )
    return pl.DataFrame(records)


def test_fallback_boundary_exact_min_pairs() -> None:
    """A group with exactly min_pairs is corrected; min_pairs - 1 falls back."""
    eval_date = date(2026, 3, 1)

    pairs_exact = _build_group_history(MIN_PAIRS, eval_date)
    result_exact = walk_forward(
        pairs_exact, BaselineModel(), start=eval_date, end=eval_date, retrain="daily"
    )
    assert result_exact.height == 1
    assert result_exact["fallback"][0] is False

    pairs_short = _build_group_history(MIN_PAIRS - 1, eval_date)
    result_short = walk_forward(
        pairs_short, BaselineModel(), start=eval_date, end=eval_date, retrain="daily"
    )
    assert result_short.height == 1
    assert result_short["fallback"][0] is True
    assert result_short["forecast_f"][0] == pytest.approx(
        result_short["raw_forecast_f"][0]
    )


def test_group_absent_from_training_falls_back() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 5)
    pairs = _noise_free_pairs(run_dates, stations=("KPHX",))

    # KORD never appears anywhere except the evaluated run's own row.
    extra_row = pairs.filter(pl.col("run_date") == run_dates[-1]).with_columns(
        station=pl.lit("KORD")
    )
    pairs_with_kord = pl.concat([pairs, extra_row], how="vertical")

    result = walk_forward(
        pairs_with_kord,
        BaselineModel(),
        start=run_dates[-1],
        end=run_dates[-1],
        retrain="daily",
    )
    kord_rows = result.filter(pl.col("station") == "KORD")
    assert kord_rows.height > 0
    assert kord_rows["fallback"].all()
    assert kord_rows["forecast_f"].equals(kord_rows["raw_forecast_f"])


def test_empty_training_set_means_everything_falls_back() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 1)
    pairs = _noise_free_pairs(run_dates, stations=("KPHX", "KORD"))

    result = walk_forward(
        pairs, BaselineModel(), start=run_dates[0], end=run_dates[0], retrain="daily"
    )

    assert result.height > 0
    assert result["fallback"].all()
    assert result["forecast_f"].equals(result["raw_forecast_f"])


def test_window_shorter_than_min_pairs_is_rejected() -> None:
    # One pair per group per day, so a window shorter than min_pairs can never
    # reach min_pairs: every group would fall back forever, silently (the W=14,
    # k=15 sensitivity run was 100% fallback). Refuse the configuration.
    with pytest.raises(ValueError, match="window_days"):
        BaselineModel(window_days=14, min_pairs=15)
    BaselineModel(window_days=15, min_pairs=15)  # the boundary is allowed


def test_predict_detail_rows_stay_aligned_to_run_rows() -> None:
    # The evaluator checks length, not alignment: if a join reordered rows,
    # corrected forecasts would land on the wrong station with no error.
    # Groups get distinct biases, run rows arrive in a scrambled order, and
    # every output row must carry its own group's correction.
    end = datetime(2025, 1, 31, 6, tzinfo=UTC)
    biases = {
        ("KAAA", 1, "max"): 1.0,
        ("KBBB", 2, "min"): 5.0,
        ("KCCC", 1, "min"): -3.0,
    }
    rows = []
    for (station, lead, variable), bias in biases.items():
        for day in range(20):
            rows.append(
                {
                    "station": station,
                    "lead_day": lead,
                    "variable": variable,
                    "forecast_f": 50.0 + bias,
                    "observed_f": 50.0,
                    "window_end_utc": end - timedelta(days=day),
                }
            )
    training = pl.DataFrame(rows).sort("window_end_utc")
    model = BaselineModel(window_days=30, min_pairs=15)
    model.fit(training)

    run_rows = pl.DataFrame(
        {
            "station": ["KCCC", "KAAA", "KBBB", "KAAA", "KCCC", "KBBB"],
            "lead_day": [1, 1, 2, 1, 1, 2],
            "variable": ["min", "max", "min", "max", "min", "min"],
            "forecast_f": [10.0, 20.0, 30.0, 40.0, 60.0, 70.0],
        }
    )
    detail = model.predict_detail(run_rows)

    expected = [
        forecast - biases[(station, lead, variable)]
        for station, lead, variable, forecast in run_rows.select(
            "station", "lead_day", "variable", "forecast_f"
        ).iter_rows()
    ]
    assert detail["forecast_f"].to_list() == expected
    assert not detail["fallback"].any()
