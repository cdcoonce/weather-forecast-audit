"""Pure completeness-summary computation (issue #11 D-completeness).

Exact rates on a hand-built fixture: every station is expected to have data
for the same archive window (station_registry carries no per-station
commissioning date), so `expected_days` only depends on the calendar, not
the station.
"""

from datetime import date

import polars as pl
import pytest

from weather_forecast_audit.completeness import (
    completeness_by_station_year,
    gap_reason_counts,
    stations_over_threshold,
)

pytestmark = pytest.mark.unit

STATIONS = ["KAAA", "KBBB"]
ARCHIVE_START = date(2020, 12, 30)
DATA_THROUGH = date(2021, 1, 2)  # 4 days: 12-30, 12-31 (2020), 01-01, 01-02 (2021)

# KAAA: two reasons on file for the SAME day (2020-12-30) -- must count as
# one gap-day, not two -- plus one more gap-day in 2021. KBBB: no gaps.
GAPS = pl.DataFrame(
    {
        "station": ["KAAA", "KAAA", "KAAA"],
        "gap_date": [date(2020, 12, 30), date(2020, 12, 30), date(2021, 1, 1)],
        "reason": ["missing_run", "missing_observations", "missing_run"],
    }
)


def test_completeness_by_station_year_exact_rates() -> None:
    result = completeness_by_station_year(GAPS, STATIONS, ARCHIVE_START, DATA_THROUGH)

    rows = {
        (row["station"], row["year"]): (
            row["expected_days"],
            row["gap_days"],
            row["gap_rate"],
        )
        for row in result.to_dicts()
    }

    assert rows[("KAAA", 2020)] == (2, 1, pytest.approx(0.5))
    assert rows[("KAAA", 2021)] == (2, 1, pytest.approx(0.5))
    assert rows[("KBBB", 2020)] == (2, 0, pytest.approx(0.0))
    assert rows[("KBBB", 2021)] == (2, 0, pytest.approx(0.0))
    assert set(rows) == {("KAAA", 2020), ("KAAA", 2021), ("KBBB", 2020), ("KBBB", 2021)}


def test_completeness_rejects_data_through_before_archive_start() -> None:
    with pytest.raises(ValueError, match="archive_start"):
        completeness_by_station_year(GAPS, STATIONS, date(2021, 1, 1), date(2020, 1, 1))


def test_gap_reason_counts_are_not_deduplicated_by_day() -> None:
    result = gap_reason_counts(GAPS)

    rows = {
        (row["station"], row["year"], row["reason"]): row["count"]
        for row in result.to_dicts()
    }

    assert rows == {
        ("KAAA", 2020, "missing_run"): 1,
        ("KAAA", 2020, "missing_observations"): 1,
        ("KAAA", 2021, "missing_run"): 1,
    }


def test_stations_over_threshold_excludes_stations_under_it() -> None:
    result = stations_over_threshold(
        GAPS, STATIONS, ARCHIVE_START, DATA_THROUGH, threshold=0.10
    )

    rows = result.to_dicts()
    assert len(rows) == 1
    assert rows[0]["station"] == "KAAA"
    assert rows[0]["gap_rate"] == pytest.approx(0.5)
    # missing_run appears twice overall (2020 + 2021), missing_observations once.
    assert rows[0]["dominant_reason"] == "missing_run"


def test_stations_over_threshold_empty_when_nothing_exceeds_it() -> None:
    result = stations_over_threshold(
        GAPS, STATIONS, ARCHIVE_START, DATA_THROUGH, threshold=0.9
    )

    assert result.to_dicts() == []


def test_stations_over_threshold_default_is_ten_percent() -> None:
    # KAAA's overall gap_rate is 2/4 = 0.5, well past the 0.10 default.
    result = stations_over_threshold(GAPS, STATIONS, ARCHIVE_START, DATA_THROUGH)

    assert [row["station"] for row in result.to_dicts()] == ["KAAA"]
