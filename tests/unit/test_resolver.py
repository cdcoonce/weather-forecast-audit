"""Pure resolver tests: TXN classification, windows, coverage, known answers.

The known-answer table (13Z run 2023-07-14, KPHX) comes from the build spec
and was independently recomputed against the ASOS fixture before this file
was written; it is the hand-checked oracle for the whole module.
"""

import csv
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from weather_forecast_audit.iem.metar import parse_six_hour_groups
from weather_forecast_audit.resolver import (
    MIN_HOUR_COVERAGE,
    Window,
    classify_txn,
    lead_day,
    observed_extreme,
    resolve_observed,
    resolve_window,
    six_hour_extreme,
    six_hour_periods,
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


def _load_six_hour_reports() -> list[tuple[datetime, float | None, float | None]]:
    reports: list[tuple[datetime, float | None, float | None]] = []
    with ASOS_FIXTURE.open(newline="") as handle:
        for record in csv.DictReader(handle):
            groups = parse_six_hour_groups(record["metar"])
            max_f = groups.max_c * 9 / 5 + 32 if groups.max_c is not None else None
            min_f = groups.min_c * 9 / 5 + 32 if groups.min_c is not None else None
            reports.append((_utc(record["valid"]), max_f, min_f))
    return reports


def _load_resolve_observed_reports() -> (
    list[tuple[datetime, float | None, float | None, float | None]]
):
    reports: list[tuple[datetime, float | None, float | None, float | None]] = []
    with ASOS_FIXTURE.open(newline="") as handle:
        for record in csv.DictReader(handle):
            tmpf = None if record["tmpf"] in ("M", "") else float(record["tmpf"])
            groups = parse_six_hour_groups(record["metar"])
            max_f = groups.max_c * 9 / 5 + 32 if groups.max_c is not None else None
            min_f = groups.min_c * 9 / 5 + 32 if groups.min_c is not None else None
            reports.append((_utc(record["valid"]), tmpf, max_f, min_f))
    return reports


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


# -- six_hour_periods ----------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [date(2023, 7, 15), date(2026, 3, 8), date(2026, 11, 1)],
    ids=["normal", "us-dst-start", "us-dst-end"],
)
def test_six_hour_periods_max_window(target: date) -> None:
    window = resolve_window(target, "max")
    start = datetime(target.year, target.month, target.day, 12, 0, tzinfo=UTC)
    assert six_hour_periods(window) == (
        start + timedelta(hours=6),
        start + timedelta(hours=12),
        start + timedelta(hours=18),
    )


@pytest.mark.parametrize(
    "target",
    [date(2023, 7, 15), date(2026, 3, 8), date(2026, 11, 1)],
    ids=["normal", "us-dst-start", "us-dst-end"],
)
def test_six_hour_periods_min_window(target: date) -> None:
    window = resolve_window(target, "min")
    start = datetime(target.year, target.month, target.day, 0, 0, tzinfo=UTC)
    assert six_hour_periods(window) == (
        start + timedelta(hours=6),
        start + timedelta(hours=12),
        start + timedelta(hours=18),
    )


# -- six_hour_extreme: known answers (hand-checked KPHX day) ------------------


@pytest.mark.parametrize(
    ("variable", "expected_value_f"),
    [("max", 118.04), ("min", 91.94)],
)
def test_six_hour_extreme_known_answer_kphx_2023_07_15(
    variable: str, expected_value_f: float
) -> None:
    reports = _load_six_hour_reports()
    window = resolve_window(date(2023, 7, 15), variable)
    result = six_hour_extreme(window, variable, reports)
    assert result.tiled is True
    assert result.periods_found == 3
    assert result.value_f == pytest.approx(expected_value_f)


# -- six_hour_extreme: boundary -----------------------------------------------


def test_six_hour_extreme_boundary_at_h_excluded() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    (h1, _h2, _h3) = six_hour_periods(window)
    result = six_hour_extreme(window, "max", [(h1, 999.0, None)])
    assert result.periods_found == 0


def test_six_hour_extreme_boundary_h_minus_60_included() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    (h1, _h2, _h3) = six_hour_periods(window)
    reports = [(h1 - timedelta(minutes=60), 999.0, None)]
    result = six_hour_extreme(window, "max", reports)
    assert result.periods_found == 1


def test_six_hour_extreme_boundary_h_minus_61_excluded() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    (h1, _h2, _h3) = six_hour_periods(window)
    reports = [(h1 - timedelta(minutes=61), 999.0, None)]
    result = six_hour_extreme(window, "max", reports)
    assert result.periods_found == 0


# -- six_hour_extreme: latest report wins, missing period ---------------------


def test_six_hour_extreme_latest_report_wins() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    h1, h2, h3 = six_hour_periods(window)
    reports = [
        (h1 - timedelta(minutes=50), 200.0, None),  # earlier, higher value
        (h1 - timedelta(minutes=10), 105.0, None),  # later report: should win
        (h2 - timedelta(minutes=9), 100.0, None),
        (h3 - timedelta(minutes=9), 100.0, None),
    ]
    result = six_hour_extreme(window, "max", reports)
    assert result.tiled is True
    assert result.value_f == 105.0


def test_six_hour_extreme_one_missing_period_not_tiled() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    h1, h2, _h3 = six_hour_periods(window)
    reports = [
        (h1 - timedelta(minutes=9), 100.0, None),
        (h2 - timedelta(minutes=9), 100.0, None),
    ]
    result = six_hour_extreme(window, "max", reports)
    assert result.tiled is False
    assert result.value_f is None
    assert result.periods_found == 2


# -- synthetic: an extreme between hourly readings (issue #6's core case) ----


def test_synthetic_extreme_between_hourly_readings() -> None:
    """The tracer's core #6 evidence: an actual peak between hourly readings.

    Hourly tmpf peaks at 117, but the METAR 6-hour max group -- sampled
    continuously by the station's own sensor, not just at :51 past the hour
    -- caught a warmer moment the hourly cadence missed. six_hour_extreme
    surfaces it; observed_extreme, built only from the hourly readings,
    cannot.
    """
    window = resolve_window(date(2024, 1, 1), "max")
    hourly_obs = [
        (window.start_utc + timedelta(hours=h), 110.0 + h) for h in range(7)
    ] + [
        (window.start_utc + timedelta(hours=h), 117.0 - (h - 7)) for h in range(7, 18)
    ]
    hourly_result = observed_extreme(window, "max", hourly_obs)
    assert hourly_result.value_f == 117.0

    h1, h2, h3 = six_hour_periods(window)
    six_hour_reports = [
        (h1 - timedelta(minutes=9), 110.0, None),
        (h2 - timedelta(minutes=9), 118.2, None),  # the missed peak
        (h3 - timedelta(minutes=9), 112.0, None),
    ]
    six_hour_result = six_hour_extreme(window, "max", six_hour_reports)
    assert six_hour_result.tiled is True
    assert six_hour_result.value_f == 118.2
    assert six_hour_result.value_f > hourly_result.value_f


# -- resolve_observed: known answers (five 13Z-2023-07-14 targets) -----------


@pytest.mark.parametrize(
    ("target", "variable", "expected_value_f", "expected_hourly_f"),
    [
        (date(2023, 7, 15), "min", 91.94, 93.0),
        (date(2023, 7, 15), "max", 118.04, 117.0),
        (date(2023, 7, 16), "min", 93.92, 94.0),
        (date(2023, 7, 16), "max", 114.08, 113.0),
    ],
)
def test_resolve_observed_tiled_targets_are_metar_6h_and_scorable(
    target: date,
    variable: str,
    expected_value_f: float,
    expected_hourly_f: float,
) -> None:
    reports = _load_resolve_observed_reports()
    window = resolve_window(target, variable)
    result = resolve_observed(window, variable, reports)

    assert result.extreme_source == "metar_6h"
    assert result.scorable is True
    assert result.periods_found == 3
    assert result.value_f == pytest.approx(expected_value_f)
    assert result.hourly_value_f == pytest.approx(expected_hourly_f)


def test_resolve_observed_untiled_target_is_none_and_unscorable() -> None:
    """2023-07-17's min window: the 17:51 report the fixture drops leaves the

    18Z synoptic period without a qualifying report, so the window cannot be
    tiled. hourly_value_f still follows its own coverage threshold (12 of 18
    hours here, below MIN_HOUR_COVERAGE), independent of extreme_source.
    """
    reports = _load_resolve_observed_reports()
    window = resolve_window(date(2023, 7, 17), "min")
    result = resolve_observed(window, "min", reports)

    assert result.extreme_source == "none"
    assert result.scorable is False
    assert result.value_f is None
    assert result.periods_found == 2
    assert result.hourly_value_f is None
    assert result.hours_covered == 12


# -- resolve_observed: synthetic fallback and recovery cases ------------------


def test_resolve_observed_one_missing_synoptic_report_falls_back_to_none() -> None:
    window = resolve_window(date(2023, 7, 15), "max")
    h1, h2, _h3 = six_hour_periods(window)
    reports = [
        (h1 - timedelta(minutes=9), None, 100.0, None),
        (h2 - timedelta(minutes=9), None, 100.0, None),
        # third synoptic report (h3) is missing entirely
    ]
    result = resolve_observed(window, "max", reports)

    assert result.extreme_source == "none"
    assert result.scorable is False
    assert result.value_f is None
    assert result.periods_found == 2


# The two sources' coverage must be tested independently. Every other case
# varies them together (both complete or both empty), which leaves a hourly
# fallback, or an extra hourly-coverage condition, invisible to the suite.


def test_resolve_observed_untiled_stays_unscorable_despite_full_hourly() -> None:
    window = resolve_window(date(2024, 1, 1), "max")
    hourly = [
        (window.start_utc + timedelta(hours=h, minutes=51), 100.0 + h, None, None)
        for h in range(18)
    ]
    h1, h2, _h3 = six_hour_periods(window)
    six_hour = [
        (h1 - timedelta(minutes=9), None, 106.0, None),
        (h2 - timedelta(minutes=9), None, 112.0, None),
    ]
    result = resolve_observed(window, "max", hourly + six_hour)

    assert result.hourly_value_f == 117.0  # the hourly side is complete
    assert result.extreme_source == "none"
    assert result.scorable is False
    assert result.value_f is None


def test_resolve_observed_tiled_is_scorable_despite_sparse_hourly() -> None:
    window = resolve_window(date(2024, 1, 1), "min")
    h1, h2, h3 = six_hour_periods(window)
    # Only the three synoptic reports: 3 of 18 hours covered, far below the
    # hourly threshold, but the 6-hour groups tile the window.
    reports = [
        (h1 - timedelta(minutes=9), 40.0, None, 35.0),
        (h2 - timedelta(minutes=9), 38.0, None, 31.5),
        (h3 - timedelta(minutes=9), 45.0, None, 33.0),
    ]
    result = resolve_observed(window, "min", reports)

    assert result.hourly_value_f is None  # the hourly side fails its threshold
    assert result.extreme_source == "metar_6h"
    assert result.scorable is True
    assert result.value_f == 31.5


def test_resolve_observed_recovers_peak_hourly_misses() -> None:
    """issue #6's core case: metar_6h finds a peak between hourly readings."""
    window = resolve_window(date(2024, 1, 1), "max")
    hourly_reports = [
        (window.start_utc + timedelta(hours=h), 110.0 + h, None, None)
        for h in range(7)
    ] + [
        (window.start_utc + timedelta(hours=h), 117.0 - (h - 7), None, None)
        for h in range(7, 18)
    ]
    h1, h2, h3 = six_hour_periods(window)
    six_hour_reports = [
        (h1 - timedelta(minutes=9), None, 110.0, None),
        (h2 - timedelta(minutes=9), None, 118.2, None),  # the missed peak
        (h3 - timedelta(minutes=9), None, 112.0, None),
    ]
    result = resolve_observed(window, "max", hourly_reports + six_hour_reports)

    assert result.extreme_source == "metar_6h"
    assert result.scorable is True
    assert result.value_f == 118.2
    assert result.hourly_value_f == 117.0
    assert result.value_f > result.hourly_value_f
