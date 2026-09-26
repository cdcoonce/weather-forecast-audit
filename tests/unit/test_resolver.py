"""Pure resolver tests: TXN classification, windows, coverage, known answers.

The known-answer table (13Z run 2023-07-14, KPHX) comes from the build spec
and was independently recomputed against the ASOS fixture before this file
was written; it is the hand-checked oracle for the whole module.
"""

import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from weather_forecast_audit.resolver import (
    MIN_HOUR_COVERAGE,
    Window,
    classify_txn,
    lead_day,
    observed_extreme,
    resolve_window,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ASOS_FIXTURE = (
    REPO / "tests" / "fixtures" / "iem" / "asos_kphx_2023-07-13_2023-07-17.csv"
)


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)


def _load_asos_observations() -> list[tuple[datetime, float | None]]:
    rows: list[tuple[datetime, float | None]] = []
    with ASOS_FIXTURE.open(newline="") as handle:
        for record in csv.DictReader(handle):
            tmpf = None if record["tmpf"] in ("M", "") else float(record["tmpf"])
            rows.append((_utc(record["valid"]), tmpf))
    return rows


# -- classify_txn ------------------------------------------------------------


def test_classify_txn_00z_is_max_for_previous_day() -> None:
    variable, target = classify_txn(_utc("2023-07-16 00:00"))
    assert variable == "max"
    assert target == date(2023, 7, 15)


def test_classify_txn_12z_is_min_for_same_day() -> None:
    variable, target = classify_txn(_utc("2023-07-15 12:00"))
    assert variable == "min"
    assert target == date(2023, 7, 15)


@pytest.mark.parametrize("hour", [1, 6, 13, 18, 23])
def test_classify_txn_other_hours_raise(hour: int) -> None:
    ftime = _utc("2023-07-15 00:00").replace(hour=hour)
    with pytest.raises(ValueError, match="hour"):
        classify_txn(ftime)


def test_classify_txn_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="naive"):
        classify_txn(datetime(2023, 7, 15, 0, 0))  # noqa: DTZ001 (naive on purpose)


# -- resolve_window ------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [date(2023, 7, 15), date(2026, 3, 8), date(2026, 11, 1)],
    ids=["normal", "us-dst-start", "us-dst-end"],
)
def test_resolve_window_max_is_always_18h_fixed_utc(target: date) -> None:
    window = resolve_window(target, "max")
    assert window.start_utc == datetime(
        target.year, target.month, target.day, 12, 0, tzinfo=UTC
    )
    assert window.end_utc - window.start_utc == timedelta(hours=18)


@pytest.mark.parametrize(
    "target",
    [date(2023, 7, 15), date(2026, 3, 8), date(2026, 11, 1)],
    ids=["normal", "us-dst-start", "us-dst-end"],
)
def test_resolve_window_min_is_always_18h_fixed_utc(target: date) -> None:
    window = resolve_window(target, "min")
    assert window.start_utc == datetime(
        target.year, target.month, target.day, 0, 0, tzinfo=UTC
    )
    assert window.end_utc - window.start_utc == timedelta(hours=18)


# -- lead_day ------------------------------------------------------------


def test_lead_day_13z_run_leads_1_through_3() -> None:
    runtime = _utc("2023-07-14 13:00")
    assert lead_day(runtime, date(2023, 7, 15)) == 1
    assert lead_day(runtime, date(2023, 7, 16)) == 2
    assert lead_day(runtime, date(2023, 7, 17)) == 3


def test_lead_day_12z_run_includes_lead_0() -> None:
    runtime = _utc("2026-05-06 12:00")
    assert lead_day(runtime, date(2026, 5, 6)) == 0
    assert lead_day(runtime, date(2026, 5, 7)) == 1


# -- observed_extreme: boundary + coverage ------------------------------------


def test_observed_extreme_boundary_start_included_end_excluded() -> None:
    # A tiny window can never clear the fixed 14-of-18-hour threshold, so
    # boundary handling is checked on the raw counts, not the gated value_f.
    window = Window(_utc("2023-07-15 00:00"), _utc("2023-07-15 01:00"))
    obs = [
        (_utc("2023-07-15 00:00"), 50.0),  # exactly start: in
        (_utc("2023-07-15 01:00"), 999.0),  # exactly end: out
    ]
    result = observed_extreme(window, "max", obs)
    assert result.n_obs == 1
    assert result.hours_covered == 1


def test_observed_extreme_one_minute_before_end_included() -> None:
    window = Window(_utc("2023-07-15 00:00"), _utc("2023-07-15 01:00"))
    obs = [(_utc("2023-07-15 00:00") + timedelta(minutes=59), 77.0)]
    result = observed_extreme(window, "max", obs)
    assert result.n_obs == 1
    assert result.hours_covered == 1


def test_observed_extreme_missing_tmpf_does_not_count_toward_coverage() -> None:
    window = Window(_utc("2023-07-15 00:00"), _utc("2023-07-15 03:00"))
    obs = [
        (_utc("2023-07-15 00:00"), None),
        (_utc("2023-07-15 01:00"), None),
    ]
    result = observed_extreme(window, "max", obs)
    assert result.hours_covered == 0
    assert result.n_obs == 0
    assert result.value_f is None
    assert result.scorable is False


@pytest.mark.parametrize(
    ("hours_present", "expected_scorable"),
    [(14, True), (13, False), (15, True)],
)
def test_observed_extreme_completeness_threshold(
    hours_present: int, expected_scorable: bool
) -> None:
    window = Window(_utc("2023-07-15 00:00"), _utc("2023-07-16 06:00"))
    obs = [
        (_utc("2023-07-15 00:00") + timedelta(hours=h), 70.0 + h)
        for h in range(hours_present)
    ]
    result = observed_extreme(window, "max", obs, MIN_HOUR_COVERAGE)
    assert result.hours_covered == hours_present
    assert result.scorable is expected_scorable
    if not expected_scorable:
        assert result.value_f is None


def test_observed_extreme_rejects_naive_valid_time() -> None:
    window = Window(_utc("2023-07-15 00:00"), _utc("2023-07-15 03:00"))
    obs = [(datetime(2023, 7, 15, 0, 30), 70.0)]  # noqa: DTZ001 (naive on purpose)
    with pytest.raises(ValueError, match="naive"):
        observed_extreme(window, "max", obs)


# -- known-answer table (hand-checked KPHX day) -------------------------------


@pytest.mark.parametrize(
    ("target", "variable", "expected_value", "expected_hours"),
    [
        (date(2023, 7, 15), "min", 93.0, 18),
        (date(2023, 7, 15), "max", 117.0, 18),
        (date(2023, 7, 16), "min", 94.0, 18),
        (date(2023, 7, 16), "max", 113.0, 18),
        (date(2023, 7, 17), "min", None, 12),
        (date(2023, 7, 14), "max", 115.0, 18),
        (date(2023, 7, 14), "min", 94.0, 18),
    ],
)
def test_known_answer_table_kphx_2023_07_14_run(
    target: date,
    variable: str,
    expected_value: float | None,
    expected_hours: int,
) -> None:
    observations = _load_asos_observations()
    window = resolve_window(target, variable)
    result = observed_extreme(window, variable, observations)
    assert result.value_f == expected_value
    assert result.hours_covered == expected_hours
    if expected_value is None:
        assert result.scorable is False
    else:
        assert result.scorable is True
