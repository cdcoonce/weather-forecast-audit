"""Dagster resources: swappable in tests (build spec #10 D5)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.resources import (
    ClockResource,
    IemResource,
    StationsResource,
    WarehouseResource,
)

pytestmark = pytest.mark.dagster


@pytest.mark.io
def test_warehouse_resource_connect_initializes_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "warehouse.duckdb"
    resource = WarehouseResource(duckdb_path=str(db_path))

    with resource.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'raw'"
            ).fetchall()
        }

    assert "nbs_guidance" in tables
    assert "ingest_gaps" in tables


def test_iem_resource_fetcher_defaults_match_urllib_fetcher() -> None:
    resource = IemResource()
    fetcher = resource.fetcher()

    assert isinstance(fetcher, UrllibFetcher)
    assert fetcher.min_interval_s == 1.0


def test_iem_resource_fetcher_honors_overridden_config() -> None:
    resource = IemResource(
        min_interval_s=0.0, max_retries=1, backoff_s=0.0, timeout_s=5
    )
    fetcher = resource.fetcher()

    assert isinstance(fetcher, UrllibFetcher)
    assert fetcher.min_interval_s == 0.0
    assert fetcher.max_retries == 1
    assert fetcher.timeout_s == 5


def test_stations_resource_default_returns_whole_registry_sorted() -> None:
    resource = StationsResource()
    stations = resource.stations()

    icaos = [s.icao for s in stations]
    assert icaos == sorted(icaos)
    assert "KPHX" in icaos
    assert "KORD" in icaos


def test_stations_resource_only_filters_and_sorts() -> None:
    resource = StationsResource(only=["KORD", "KPHX"])
    stations = resource.stations()

    assert [s.icao for s in stations] == ["KORD", "KPHX"]


def test_stations_resource_unknown_icao_raises() -> None:
    resource = StationsResource(only=["KPHX", "KBOGUS"])

    with pytest.raises(ValueError, match="KBOGUS"):
        resource.stations()


def test_clock_resource_now_override_parses_iso_utc() -> None:
    resource = ClockResource(now_override="2023-07-16T00:00:00Z")

    assert resource.now() == datetime(2023, 7, 16, tzinfo=UTC).replace(tzinfo=None)


def test_clock_resource_default_now_is_close_to_wall_clock() -> None:
    resource = ClockResource()

    now = resource.now()

    assert now.tzinfo is None
    assert abs(now - datetime.now(UTC).replace(tzinfo=None)) < timedelta(seconds=30)
