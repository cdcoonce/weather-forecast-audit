"""Re-ingesting the same station/range twice must not duplicate rows
(issue #11 acceptance: a killed, re-run backfill unit must be safe to
retry).

`warehouse.py`'s loaders are documented as delete-then-insert per
station/date-range (see its module docstring and `warehouse._replace`);
this test proves that end to end through `pipeline.ingest_station`, the
same entry point the Dagster ingest assets and `wfa ingest` use, with the
`FixtureFetcher` recipe `tests/integration/test_tracer_pipeline.py` and
`test_export_tracer.py` already share.
"""

from datetime import date
from pathlib import Path

import duckdb
import pytest
from support.fixture_fetcher import FixtureFetcher

from weather_forecast_audit import warehouse
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry

pytestmark = [pytest.mark.integration, pytest.mark.io]

RAW_TABLES = [
    "nbs_guidance",
    "asos_hourly",
    "cli_daily",
    "resolved_windows",
    "ingest_gaps",
]


def _table_counts(db_path: Path) -> dict[str, int]:
    with duckdb.connect(str(db_path)) as conn:
        return {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()[0]  # noqa: S608
            for table in RAW_TABLES
        }


def test_reingesting_the_same_station_and_range_leaves_row_counts_unchanged(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "idempotence.duckdb"
    station = load_registry()["KPHX"]
    regimes = load_cycle_regimes()
    fetcher = FixtureFetcher()
    start, end = date(2023, 7, 14), date(2023, 7, 14)

    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        first = ingest_station(conn, station, start, end, fetcher, regimes)
    counts_after_first = _table_counts(db_path)

    # A second ingest of the exact same station/range, as a killed backfill
    # unit's retry would do: a fresh connection, same inputs.
    with duckdb.connect(str(db_path)) as conn:
        second = ingest_station(conn, station, start, end, fetcher, regimes)
    counts_after_second = _table_counts(db_path)

    assert counts_after_first == counts_after_second
    assert counts_after_first["nbs_guidance"] > 0
    assert counts_after_first["asos_hourly"] > 0
    assert first.guidance_rows == second.guidance_rows
    assert first.asos_rows == second.asos_rows
    assert first.cli_rows == second.cli_rows
    assert first.resolved_rows == second.resolved_rows
    assert first.gaps_by_reason == second.gaps_by_reason
