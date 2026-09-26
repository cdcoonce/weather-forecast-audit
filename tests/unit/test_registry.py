"""Station registry: the seed CSV keyed by ICAO."""

from pathlib import Path

import pytest

from weather_forecast_audit.registry import load_registry

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "dbt" / "seeds" / "station_registry.csv"


def test_load_registry_keys_by_icao() -> None:
    registry = load_registry(SEED)

    assert set(registry) == {"KPHX"}
    station = registry["KPHX"]
    assert station.cli_station == "KPHX"
    assert station.nbm_station_id == "KPHX"
    assert station.iana_tz == "America/Phoenix"
    assert station.climate_region == "Southwest"
    assert station.coastal_flag is False
    assert station.elevation_m == 337
    assert station.lat == pytest.approx(33.43428)
    assert station.lon == pytest.approx(-112.01158)
