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


def _naive_now(text: str) -> datetime:
    """A naive-UTC `now` value, as `load_gaps` requires (build spec D3)."""
    return _utc(text).replace(tzinfo=None)


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
    rows = [
        HourlyObservation(
            "KPHX", _utc("2023-07-14 00:00:00"), 100.0, max_6h_f=None, min_6h_f=None
        )
    ]
    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)
    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)

    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (1,)


def test_load_hourly_writes_six_hour_group_columns(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    rows = [
        HourlyObservation(
            "KPHX",
            _utc("2023-07-15 23:51:00"),
            115.0,
            max_6h_f=118.04,
            min_6h_f=107.06,
        )
    ]
    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 15), date(2023, 7, 15), rows)

    result = conn.execute(
        "select max_6h_f, min_6h_f from raw.asos_hourly where station = 'KPHX'"
    ).fetchone()
    assert result == (118.04, 107.06)


def test_init_db_upgrades_pre_existing_old_schema_table(
    tmp_path: Path,
) -> None:
    """A DuckDB file created before this migration only has the old columns.

    init_db must upgrade it in place (add the new columns, keep the rows)
    rather than requiring a fresh database, because the file is a long-lived
    cache (build spec fact 3).
    """
    old_db = tmp_path / "old_warehouse.duckdb"
    connection = duckdb.connect(str(old_db))
    connection.execute("create schema if not exists raw")
    connection.execute(
        "create table raw.asos_hourly ("
        "station varchar not null, valid_utc timestamp not null, tmpf double)"
    )
    connection.execute(
        "insert into raw.asos_hourly values ('KPHX', '2023-07-14 00:00:00', 100.0)"
    )

    warehouse.init_db(connection)

    columns = {
        row[0]
        for row in connection.execute(
            "select column_name from information_schema.columns "
            "where table_schema = 'raw' and table_name = 'asos_hourly'"
        ).fetchall()
    }
    assert {"station", "valid_utc", "tmpf", "max_6h_f", "min_6h_f"} <= columns
    assert connection.execute("select count(*) from raw.asos_hourly").fetchone() == (
        1,
    )
    row = connection.execute(
        "select station, tmpf, max_6h_f, min_6h_f from raw.asos_hourly"
    ).fetchone()
    assert row == ("KPHX", 100.0, None, None)


def test_load_cli_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = [CliDaily("KPHX", date(2023, 7, 14), 116, 93)]
    warehouse.load_cli(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)
    warehouse.load_cli(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows)

    assert conn.execute("select count(*) from raw.cli_daily").fetchone() == (1,)


def test_load_gaps_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    gaps = [GapRecord("KPHX", "nbs", "2023-07-14T13:00Z", "missing_run")]
    now = _naive_now("2024-01-01 00:00:00")
    warehouse.load_gaps(
        conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps, now=now
    )
    warehouse.load_gaps(
        conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps, now=now
    )

    assert conn.execute("select count(*) from raw.ingest_gaps").fetchone() == (1,)


def test_load_gaps_keeps_first_seen_for_persisting_gap(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    gaps = [GapRecord("KPHX", "nbs", "2023-07-14T13:00Z", "missing_run")]
    first_run = _naive_now("2024-01-01 00:00:00")
    later_run = _naive_now("2024-06-01 00:00:00")

    warehouse.load_gaps(
        conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps, now=first_run
    )
    warehouse.load_gaps(
        conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps, now=later_run
    )

    first_seen = conn.execute(
        "select first_seen from raw.ingest_gaps where station = 'KPHX'"
    ).fetchone()
    assert first_seen == (first_run,)


def test_load_gaps_assigns_first_seen_to_new_gap(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    gaps = [GapRecord("KPHX", "nbs", "2023-07-14T13:00Z", "missing_run")]
    now = _naive_now("2024-03-01 00:00:00")

    warehouse.load_gaps(
        conn, "KPHX", "nbs", date(2023, 7, 14), date(2023, 7, 14), gaps, now=now
    )

    first_seen = conn.execute(
        "select first_seen from raw.ingest_gaps where station = 'KPHX'"
    ).fetchone()
    assert first_seen == (now,)


def test_load_gaps_deletes_vanished_gap(conn: duckdb.DuckDBPyConnection) -> None:
    gap_a = GapRecord("KPHX", "nbs", "2023-07-14T13:00Z", "missing_run")
    gap_b = GapRecord("KPHX", "nbs", "2023-07-15T13:00Z", "missing_run")
    first_run = _naive_now("2024-01-01 00:00:00")
    later_run = _naive_now("2024-06-01 00:00:00")

    warehouse.load_gaps(
        conn,
        "KPHX",
        "nbs",
        date(2023, 7, 14),
        date(2023, 7, 15),
        [gap_a, gap_b],
        now=first_run,
    )
    warehouse.load_gaps(
        conn,
        "KPHX",
        "nbs",
        date(2023, 7, 14),
        date(2023, 7, 15),
        [gap_a],
        now=later_run,
    )

    rows = conn.execute(
        "select expected, first_seen from raw.ingest_gaps where station = 'KPHX'"
    ).fetchall()
    assert rows == [("2023-07-14T13:00Z", first_run)]


def test_init_db_upgrades_ingest_gaps_old_schema(tmp_path: Path) -> None:
    """A pre-#10 database lacks `first_seen`; init_db must add it in place."""
    old_db = tmp_path / "old_warehouse.duckdb"
    connection = duckdb.connect(str(old_db))
    connection.execute("create schema if not exists raw")
    connection.execute(
        "create table raw.ingest_gaps ("
        "station varchar not null, source varchar not null, "
        "expected varchar not null, reason varchar not null)"
    )
    connection.execute(
        "insert into raw.ingest_gaps values ('KPHX', 'nbs', "
        "'2023-07-14T13:00Z', 'missing_run')"
    )

    warehouse.init_db(connection)

    columns = {
        row[0]
        for row in connection.execute(
            "select column_name from information_schema.columns "
            "where table_schema = 'raw' and table_name = 'ingest_gaps'"
        ).fetchall()
    }
    assert "first_seen" in columns
    assert connection.execute("select count(*) from raw.ingest_gaps").fetchone() == (
        1,
    )


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
            observed_f=118.04,
            n_obs=18,
            hours_covered=18,
            hours_expected=18,
            scorable=True,
            extreme_source="metar_6h",
            periods_found=3,
            hourly_observed_f=117.0,
        )
    ]
    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows
    )
    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 14), date(2023, 7, 14), rows
    )

    assert conn.execute("select count(*) from raw.resolved_windows").fetchone() == (1,)


def test_load_resolved_windows_writes_extreme_source_columns(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    rows = [
        ResolvedWindowRow(
            station="KPHX",
            runtime_utc=_utc("2023-07-16 13:00:00"),
            ftime_utc=_utc("2023-07-17 12:00:00"),
            variable="min",
            target_date=date(2023, 7, 17),
            lead_day=1,
            window_start_utc=_utc("2023-07-17 00:00:00"),
            window_end_utc=_utc("2023-07-17 18:00:00"),
            observed_f=None,
            n_obs=12,
            hours_covered=12,
            hours_expected=18,
            scorable=False,
            extreme_source="none",
            periods_found=2,
            hourly_observed_f=None,
        )
    ]
    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 16), date(2023, 7, 16), rows
    )

    result = conn.execute(
        "select observed_f, extreme_source, periods_found, hourly_observed_f "
        "from raw.resolved_windows where station = 'KPHX'"
    ).fetchone()
    assert result == (None, "none", 2, None)


def test_init_db_upgrades_resolved_windows_old_schema(tmp_path: Path) -> None:
    """A pre-#6 database lacks extreme_source/periods_found/hourly_observed_f.

    init_db must add them in place, matching the max_6h_f/min_6h_f upgrade
    already done for raw.asos_hourly in phase 1 (build spec item 2).
    """
    old_db = tmp_path / "old_warehouse.duckdb"
    connection = duckdb.connect(str(old_db))
    connection.execute("create schema if not exists raw")
    connection.execute(
        "create table raw.resolved_windows ("
        "station varchar not null, runtime_utc timestamp not null, "
        "ftime_utc timestamp not null, variable varchar not null, "
        "target_date date not null, lead_day integer not null, "
        "window_start_utc timestamp not null, window_end_utc timestamp not null, "
        "observed_f double, n_obs integer not null, hours_covered integer not null, "
        "hours_expected integer not null, scorable boolean not null)"
    )
    connection.execute(
        "insert into raw.resolved_windows values ("
        "'KPHX', '2023-07-14 13:00:00', '2023-07-15 00:00:00', 'max', "
        "'2023-07-15', 1, '2023-07-15 12:00:00', '2023-07-16 06:00:00', "
        "117.0, 18, 18, 18, true)"
    )

    warehouse.init_db(connection)

    columns = {
        row[0]
        for row in connection.execute(
            "select column_name from information_schema.columns "
            "where table_schema = 'raw' and table_name = 'resolved_windows'"
        ).fetchall()
    }
    assert {"extreme_source", "periods_found", "hourly_observed_f"} <= columns
    assert connection.execute(
        "select count(*) from raw.resolved_windows"
    ).fetchone() == (1,)
    row = connection.execute(
        "select station, observed_f, extreme_source, periods_found, "
        "hourly_observed_f from raw.resolved_windows"
    ).fetchone()
    assert row == ("KPHX", 117.0, None, None, None)
