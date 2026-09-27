"""Station registry: the seed CSV keyed by ICAO.

The full CONUS registry (issue #7) is built by
`scripts/registry/build_registry.py`; `tests/unit/test_registry_seed.py`
validates every row. This file keeps a hand-checked spot check on KPHX,
the station every other pre-#7 test and fixture is built around.
"""

from pathlib import Path

import pytest

from weather_forecast_audit.registry import load_registry

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "dbt" / "seeds" / "station_registry.csv"


def test_load_registry_keys_by_icao() -> None:
    registry = load_registry(SEED)

    assert "KPHX" in registry
    assert len(registry) > 1
    station = registry["KPHX"]
    assert station.cli_station == "KPHX"
    assert station.nbm_station_id == "KPHX"
    assert station.iana_tz == "America/Phoenix"
    assert station.climate_region == "Southwest"
    assert station.coastal_flag is False
    assert station.elevation_m == 337
    assert station.lat == pytest.approx(33.4343)
    assert station.lon == pytest.approx(-112.0116)
