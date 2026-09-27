"""Cycle-regime lookup: the seed CSV is the only source of changeover dates."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from weather_forecast_audit.regimes import (
    canonical_cycle_hour,
    load_archive_start,
    load_cycle_regimes,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "dbt" / "seeds" / "nbs_cycle_regimes.csv"
ARCHIVE_SEED = REPO / "dbt" / "seeds" / "nbs_archive.csv"


def test_load_archive_start_returns_the_pinned_date() -> None:
    archive_start = load_archive_start(ARCHIVE_SEED)
    assert isinstance(archive_start, date)
    # Pinned by scripts/registry/probe_archive_start.py (issue #7 decision 6,
    # orchestrator redirect): the first date every registry station has a
    # non-null txn, which is the NBM v4.0 rollout, not the earlier
    # 2020-02-25/26 cycle-schedule change (that date used a differently
    # named n_x column for the daily max/min, which this project never
    # reads -- see the nbs_cycle_regimes.csv note).
    assert archive_start == date(2020, 9, 29)


def test_load_cycle_regimes_reads_seed_csv() -> None:
    regimes = load_cycle_regimes(SEED)
    archive_start = load_archive_start(ARCHIVE_SEED)

    assert len(regimes) == 2
    assert regimes[0].canonical_cycle_hour == 13
    assert regimes[0].valid_from == archive_start
    assert regimes[0].valid_to == date(2026, 4, 29)
    assert regimes[1].canonical_cycle_hour == 12
    assert regimes[1].valid_from == date(2026, 4, 30)
    assert regimes[1].valid_to is None


def test_canonical_cycle_hour_by_date() -> None:
    regimes = load_cycle_regimes(SEED)
    archive_start = load_archive_start(ARCHIVE_SEED)

    cases = [
        (archive_start, 13),
        (date(2023, 7, 14), 13),
        (date(2026, 4, 29), 13),
        (date(2026, 4, 30), 12),
        (date(2026, 5, 6), 12),
        (date(2030, 1, 1), 12),
    ]
    for run_date, expected_hour in cases:
        assert canonical_cycle_hour(run_date, regimes) == expected_hour


def test_canonical_cycle_hour_day_before_archive_start_raises() -> None:
    regimes = load_cycle_regimes(SEED)
    archive_start = load_archive_start(ARCHIVE_SEED)
    day_before = archive_start - timedelta(days=1)
    with pytest.raises(ValueError, match="no regime"):
        canonical_cycle_hour(day_before, regimes)


def test_canonical_cycle_hour_2020_02_25_raises_before_archive_start() -> None:
    # The archive was already running its 1/7/13/19 UTC cycle schedule on
    # this date (build spec #7 decision 5's original cycle-schedule fact),
    # but its daily max/min was not yet readable as `txn` until the archive
    # start pinned above -- so this date is still before any regime.
    regimes = load_cycle_regimes(SEED)
    with pytest.raises(ValueError, match="no regime"):
        canonical_cycle_hour(date(2020, 2, 25), regimes)


def test_canonical_cycle_hour_before_any_regime_raises() -> None:
    regimes = load_cycle_regimes(SEED)
    with pytest.raises(ValueError, match="no regime"):
        canonical_cycle_hour(date(2000, 1, 1), regimes)
