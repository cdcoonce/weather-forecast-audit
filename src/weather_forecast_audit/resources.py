"""Dagster resources: swappable in tests (build spec #10 D5).

Each resource wraps an existing plain-Python entry point (`warehouse.init_db`,
`UrllibFetcher`, `registry.load_registry`) rather than re-implementing it, so
production and test code share the same underlying behavior; only the
resource layer differs (a fixture-serving `Fetcher`, a fixed `ClockResource`).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from dagster import ConfigurableResource, EnvVar

from weather_forecast_audit import warehouse
from weather_forecast_audit.iem.http import Fetcher, UrllibFetcher
from weather_forecast_audit.registry import DEFAULT_SEED_PATH, Station, load_registry


class WarehouseResource(ConfigurableResource):
    """Connects to the DuckDB warehouse, running the idempotent `init_db` first."""

    duckdb_path: str = EnvVar("WFA_DUCKDB_PATH")

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with duckdb.connect(self.duckdb_path) as conn:
            warehouse.init_db(conn)
            yield conn


class IemResource(ConfigurableResource):
    """Vends a `Fetcher`. Tests subclass this to return a fixture fetcher."""

    min_interval_s: float = 1.0
    max_retries: int = 3
    backoff_s: float = 2.0
    timeout_s: float = 60

    def fetcher(self) -> Fetcher:
        return UrllibFetcher(
            min_interval_s=self.min_interval_s,
            max_retries=self.max_retries,
            backoff_s=self.backoff_s,
            timeout_s=self.timeout_s,
        )


class StationsResource(ConfigurableResource):
    """The station registry, optionally filtered to `only` (empty = everything)."""

    registry_path: str = str(DEFAULT_SEED_PATH)
    only: list[str] = []

    def stations(self) -> list[Station]:
        registry = load_registry(Path(self.registry_path))
        if not self.only:
            return sorted(registry.values(), key=lambda station: station.icao)

        unknown = sorted(icao for icao in self.only if icao not in registry)
        if unknown:
            msg = f"unknown station(s) in StationsResource.only: {', '.join(unknown)}"
            raise ValueError(msg)
        return sorted(
            (registry[icao] for icao in self.only), key=lambda station: station.icao
        )


class ClockResource(ConfigurableResource):
    """A fake-clock resource; only the checks use it (build spec D5)."""

    now_override: str | None = None

    def now(self) -> datetime:
        """The current time, as a naive UTC datetime (repo-wide convention)."""
        if self.now_override is None:
            return datetime.now(UTC).replace(tzinfo=None)
        parsed = datetime.fromisoformat(self.now_override)
        if parsed.tzinfo is None:
            return parsed
        return parsed.astimezone(UTC).replace(tzinfo=None)
