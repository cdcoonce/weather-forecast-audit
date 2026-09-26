"""Cycle-regime lookup: the seed CSV is the only source of changeover dates."""

from datetime import date
from pathlib import Path

import pytest

from weather_forecast_audit.regimes import canonical_cycle_hour, load_cycle_regimes

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "dbt" / "seeds" / "nbs_cycle_regimes.csv"


def test_load_cycle_regimes_reads_seed_csv() -> None:
    regimes = load_cycle_regimes(SEED)

    assert len(regimes) == 2
    assert regimes[0].canonical_cycle_hour == 13
    assert regimes[0].valid_from == date(2020, 2, 25)
    assert regimes[0].valid_to == date(2026, 4, 29)
    assert regimes[1].canonical_cycle_hour == 12
    assert regimes[1].valid_from == date(2026, 4, 30)
    assert regimes[1].valid_to is None


@pytest.mark.parametrize(
    ("run_date", "expected_hour"),
    [
        (date(2020, 2, 25), 13),
        (date(2023, 7, 14), 13),
        (date(2026, 4, 29), 13),
        (date(2026, 4, 30), 12),
        (date(2026, 5, 6), 12),
        (date(2030, 1, 1), 12),
    ],
)
def test_canonical_cycle_hour_by_date(run_date: date, expected_hour: int) -> None:
    regimes = load_cycle_regimes(SEED)
    assert canonical_cycle_hour(run_date, regimes) == expected_hour


def test_canonical_cycle_hour_before_any_regime_raises() -> None:
    regimes = load_cycle_regimes(SEED)
    with pytest.raises(ValueError, match="no regime"):
        canonical_cycle_hour(date(2000, 1, 1), regimes)
