"""Raw warehouse DDL and idempotent loaders."""

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from weather_forecast_audit import warehouse
from weather_forecast_audit.gaps import GapRecord
from weather_forecast_audit.iem.guidance import GuidanceRow
from weather_forecast_audit.iem.observations import CliDaily, HourlyObservation
from weather_forecast_audit.warehouse import ResolvedWindowRow

pytestmark = [pytest.mark.unit, pytest.mark.io]


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


@pytest.fixture
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    warehouse.init_db(connection)
    return connection


def test_init_db_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    warehouse.init_db(conn)  # second call must not raise
    tables = {
        row[0]
        for row in conn.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'raw'"
        ).fetchall()
    }
    assert tables == {
        "nbs_guidance",
        "asos_hourly",
        "cli_daily",
        "ingest_gaps",
        "resolved_windows",
    }


def test_load_guidance_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [
        GuidanceRow(
            station="KPHX",
            runtime=_utc("2023-07-14 13:00:00"),
            ftime=_utc("2023-07-15 00:00:00"),
            cycle_hour=13,
            txn=118.0,
            xnd=1.0,
            tmp=100.0,
        )
    ]
    warehouse.load_guidance(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)
    warehouse.load_guidance(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)

    count = conn.execute("select count(*) from raw.nbs_guidance").fetchone()
    assert count == (1,)


def test_load_guidance_only_replaces_its_own_date_range(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    day1 = [
        GuidanceRow(
            "KPHX",
            _utc("2023-07-14 13:00:00"),
            _utc("2023-07-15 00:00:00"),
            13,
            118.0,
            1.0,
            100.0,
        )
    ]
    day2 = [
        GuidanceRow(
            "KPHX",
            _utc("2023-07-15 13:00:00"),
            _utc("2023-07-16 00:00:00"),
            13,
            113.0,
            1.0,
            99.0,
        )
    ]
    warehouse.load_guidance(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), day1)
    warehouse.load_guidance(conn, "KPHX", date(2023, 7, 15), date(2023, 7, 15), day2)

    count = conn.execute("select count(*) from raw.nbs_guidance").fetchone()
    assert count == (2,)


def test_load_hourly_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [HourlyObservation("KPHX", _utc("2023-07-14 00:00:00"), 100.0)]
    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)
    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)

    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (1,)


def test_load_cli_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [CliDaily("KPHX", date(2023, 7, 14), 116, 93)]
    warehouse.load_cli(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)
    warehouse.load_cli(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)

    assert conn.execute("select count(*) from raw.cli_daily").fetchone() == (1,)


def test_load_gaps_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    gaps = [GapRecord("KPHX", "nbs", "2023-07-14T13:00Z", "missing_run")]
    warehouse.load_gaps(conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps)
    warehouse.load_gaps(conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps)

    assert conn.execute("select count(*) from raw.ingest_gaps").fetchone() == (1,)


def test_load_resolved_windows_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [
        ResolvedWindowRow(
            station="KPHX",
            runtime_utc=_utc("2023-07-14 13:00:00"),
            ftime_utc=_utc("2023-07-15 00:00:00"),
            variable="max",
            target_date=date(2023, 7, 15),
            lead_day=1,
            window_start_utc=_utc("2023-07-15 12:00:00"),
            window_end_utc=_utc("2023-07-16 06:00:00"),
            observed_f=117.0,
            n_obs=18,
            hours_covered=18,
            hours_expected=18,
            scorable=True,
        )
    ]
    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows
    )
    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows
    )

    assert conn.execute("select count(*) from raw.resolved_windows").fetchone() == (1,)
