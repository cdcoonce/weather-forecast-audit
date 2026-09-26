"""Live smoke tests: one request per IEM client, deselected in CI.

`network`-marked, so `pytest -m "not network"` (the CI/gate invocation)
skips these; they exist to catch a live endpoint drifting out from under
the fixtures, not to pin any specific day's values.
"""

from datetime import UTC, datetime, timedelta

import pytest

from weather_forecast_audit.iem.guidance import fetch_guidance
from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.iem.observations import fetch_cli, fetch_hourly
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry

# A few days back: the archive can lag "today" by a day or two.
RECENT_DAY = (datetime.now(tz=UTC) - timedelta(days=3)).date()


@pytest.mark.network
def test_fetch_guidance_live_one_day_smoke() -> None:
    fetcher = UrllibFetcher()
    regimes = load_cycle_regimes()

    result = fetch_guidance("KPHX", RECENT_DAY, RECENT_DAY, fetcher, regimes)

    assert isinstance(result.rows, list)
    if result.rows:
        assert all(row.station == "KPHX" for row in result.rows)


@pytest.mark.network
def test_fetch_hourly_live_one_day_smoke() -> None:
    fetcher = UrllibFetcher()
    station = load_registry()["KPHX"]

    result = fetch_hourly(station, RECENT_DAY, RECENT_DAY, fetcher)

    assert isinstance(result.rows, list)
    if result.rows:
        assert all(row.station == "KPHX" for row in result.rows)


@pytest.mark.network
def test_fetch_cli_live_one_day_smoke() -> None:
    fetcher = UrllibFetcher()
    station = load_registry()["KPHX"]

    result = fetch_cli(station, RECENT_DAY, RECENT_DAY, fetcher)

    assert isinstance(result.rows, list) or isinstance(result.gaps, list)
