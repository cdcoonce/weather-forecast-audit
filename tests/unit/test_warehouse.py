"""Raw warehouse DDL and idempotent loaders."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
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
        "model_predictions",
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


@pytest.mark.parametrize(
    "gap",
    [
        # Outside the window: would be inserted but never deleted by a re-run
        # of this window, so it would accumulate duplicates silently.
        GapRecord("KPHX", "nbs", "2023-07-15T13:00Z", "missing_run"),
        GapRecord("KORD", "nbs", "2023-07-14T13:00Z", "missing_run"),
        GapRecord("KPHX", "asos", "2023-07-14T13:00Z", "missing_observations"),
    ],
    ids=["outside-window", "other-station", "other-source"],
)
def test_load_gaps_rejects_gap_it_could_never_replace(
    conn: duckdb.DuckDBPyConnection, gap: GapRecord
) -> None:
    with pytest.raises(ValueError, match="outside the replaced slice"):
        warehouse.load_gaps(
            conn,
            "KPHX",
            "nbs",
            date(2023, 7, 14),
            date(2023, 7, 14),
            [gap],
            now=_naive_now("2024-01-01 00:00:00"),
        )
    assert conn.execute("select count(*) from raw.ingest_gaps").fetchone() == (0,)


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


# -- load_predictions (build spec #18 D3.1) -----------------------------------


def _prediction_rows(
    run_dates: list[date], *, source: str = "baseline", forecast_f: float = 60.0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "station": ["KPHX"] * len(run_dates),
            "run_date": run_dates,
            "runtime_utc": [_utc("2024-01-01 13:00:00")] * len(run_dates),
            "lead_day": [1] * len(run_dates),
            "variable": ["max"] * len(run_dates),
            "target_date": [
                d + (date(2024, 1, 2) - date(2024, 1, 1)) for d in run_dates
            ],
            "source": [source] * len(run_dates),
            "forecast_f": [forecast_f] * len(run_dates),
            "raw_forecast_f": [forecast_f + 2.0] * len(run_dates),
            "retrained_on": run_dates,
            "trained_through": [_utc("2024-01-01 06:00:00")] * len(run_dates),
            "fallback": [False] * len(run_dates),
            "params": ['{"window_days": 30, "min_pairs": 15}'] * len(run_dates),
        }
    )


def test_load_predictions_is_idempotent(conn: duckdb.DuckDBPyConnection) -> None:
    rows = _prediction_rows([date(2024, 1, 1), date(2024, 1, 2)])
    now = _naive_now("2024-06-01 00:00:00")

    warehouse.load_predictions(
        conn, "baseline", date(2024, 1, 1), date(2024, 1, 2), rows, now=now
    )
    warehouse.load_predictions(
        conn, "baseline", date(2024, 1, 1), date(2024, 1, 2), rows, now=now
    )

    count = conn.execute(
        "select count(*) from raw.model_predictions where source = 'baseline'"
    ).fetchone()
    assert count == (2,)


def test_load_predictions_replaces_only_its_own_run_date_range_for_source(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _naive_now("2024-06-01 00:00:00")
    all_days = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
    warehouse.load_predictions(
        conn,
        "baseline",
        date(2024, 1, 1),
        date(2024, 1, 3),
        _prediction_rows(all_days, forecast_f=60.0),
        now=now,
    )

    # A narrower re-load of just 01-02 must replace only that date, and
    # leave the other source untouched.
    warehouse.load_predictions(
        conn,
        "challenger",
        date(2024, 1, 1),
        date(2024, 1, 3),
        _prediction_rows(all_days, source="challenger", forecast_f=70.0),
        now=now,
    )
    warehouse.load_predictions(
        conn,
        "baseline",
        date(2024, 1, 2),
        date(2024, 1, 2),
        _prediction_rows([date(2024, 1, 2)], forecast_f=99.0),
        now=now,
    )

    baseline_rows = conn.execute(
        "select run_date, forecast_f from raw.model_predictions "
        "where source = 'baseline' order by run_date"
    ).fetchall()
    assert baseline_rows == [
        (date(2024, 1, 1), 60.0),
        (date(2024, 1, 2), 99.0),
        (date(2024, 1, 3), 60.0),
    ]

    challenger_count = conn.execute(
        "select count(*) from raw.model_predictions where source = 'challenger'"
    ).fetchone()
    assert challenger_count == (3,)


def test_load_predictions_stamps_generated_at(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _naive_now("2024-06-01 12:00:00")
    warehouse.load_predictions(
        conn,
        "baseline",
        date(2024, 1, 1),
        date(2024, 1, 1),
        _prediction_rows([date(2024, 1, 1)]),
        now=now,
    )

    generated_at = conn.execute(
        "select generated_at from raw.model_predictions"
    ).fetchone()
    assert generated_at == (now,)


# -- explicit-schema loader frames (issue #41) --------------------------------
#
# `pl.DataFrame(records)` infers each column's dtype from (by default) its
# first 100 rows. A column that is whole-valued across that sample infers as
# Int64/Null; a later row that doesn't fit that dtype either gets silently
# truncated (Int64 <- float) or raises a `ComputeError` "could not append
# value ... to the builder" (Null <- non-null), depending on the transition.
# Both are reproduced here against the unmodified loaders before `_frame_for`
# is introduced.


def test_load_hourly_accepts_float_after_100_integral_readings(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """100 whole-degree tmpf readings as Python int must not poison the
    101st, fractional reading's dtype (#41)."""
    base = _utc("2023-07-14 00:00:00")
    rows = [
        HourlyObservation(
            "KPHX", base + timedelta(hours=i), 62, max_6h_f=None, min_6h_f=None
        )
        for i in range(100)
    ]
    rows.append(
        HourlyObservation(
            "KPHX", base + timedelta(hours=100), 62.96, max_6h_f=None, min_6h_f=None
        )
    )

    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 19), rows)

    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (101,)
    last_tmpf = conn.execute(
        "select tmpf from raw.asos_hourly order by valid_utc desc limit 1"
    ).fetchone()
    assert last_tmpf == (62.96,)


def _resolved_window_row(
    i: int, observed_f: float, base: datetime
) -> ResolvedWindowRow:
    runtime = base + timedelta(hours=i)
    return ResolvedWindowRow(
        station="KPHX",
        runtime_utc=runtime,
        ftime_utc=runtime + timedelta(hours=12),
        variable="max",
        target_date=date(2023, 7, 15),
        lead_day=1,
        window_start_utc=runtime,
        window_end_utc=runtime + timedelta(hours=6),
        observed_f=observed_f,
        n_obs=1,
        hours_covered=1,
        hours_expected=1,
        scorable=True,
        extreme_source="metar_6h",
        periods_found=1,
        hourly_observed_f=observed_f,
    )


def test_load_resolved_windows_accepts_float_after_100_integral_readings(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Same shape as the tmpf case, on `observed_f` (#41)."""
    base = _utc("2023-07-14 00:00:00")
    rows = [_resolved_window_row(i, 100, base) for i in range(100)]
    rows.append(_resolved_window_row(100, 100.5, base))

    warehouse.load_resolved_windows(
        conn, "KPHX", date(2023, 7, 14), date(2023, 7, 19), rows
    )

    count = conn.execute("select count(*) from raw.resolved_windows").fetchone()
    assert count == (101,)
    last_observed_f = conn.execute(
        "select observed_f from raw.resolved_windows order by runtime_utc desc limit 1"
    ).fetchone()
    assert last_observed_f == (100.5,)


def test_load_gaps_accepts_new_first_seen_after_many_null_first_seen(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A pre-#10 database has 100+ gaps on file with a NULL `first_seen`
    (predating that column). Re-running `load_gaps` over that slice plus one
    brand-new gap must not choke on the None-inferred-Null column meeting a
    real datetime for the new gap's `first_seen` (#41)."""
    start = date(2023, 7, 1)
    existing_gaps = [
        GapRecord(
            "KPHX",
            "nbs",
            f"{(start + timedelta(days=i)).isoformat()}T00:00Z",
            "missing_run",
        )
        for i in range(100)
    ]
    new_date = start + timedelta(days=100)
    new_gap = GapRecord("KPHX", "nbs", f"{new_date.isoformat()}T00:00Z", "missing_run")
    end = start + timedelta(days=110)

    conn.executemany(
        "insert into raw.ingest_gaps (station, source, expected, reason, first_seen) "
        "values (?, ?, ?, ?, NULL)",
        [(g.station, g.source, g.expected, g.reason) for g in existing_gaps],
    )
    now = _naive_now("2024-01-01 00:00:00")

    warehouse.load_gaps(
        conn, "KPHX", "nbs", start, end, [*existing_gaps, new_gap], now=now
    )

    assert conn.execute("select count(*) from raw.ingest_gaps").fetchone() == (101,)
    preserved_null = conn.execute(
        "select first_seen from raw.ingest_gaps where expected = ?",
        [existing_gaps[0].expected],
    ).fetchone()
    assert preserved_null == (None,)
    new_first_seen = conn.execute(
        "select first_seen from raw.ingest_gaps where expected = ?",
        [new_gap.expected],
    ).fetchone()
    assert new_first_seen == (now,)


def test_frame_for_yields_same_schema_for_integral_and_mixed_batches(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """`_frame_for` reads its schema from the table, not from the sampled
    records, so an all-integral batch and a mixed int/float batch for the
    same columns must produce identical dtypes -- the table's own dtypes."""
    columns = ["station", "valid_utc", "tmpf", "max_6h_f", "min_6h_f"]
    all_int_records = [
        {
            "station": "KPHX",
            "valid_utc": _utc("2023-07-14 00:00:00").replace(tzinfo=None),
            "tmpf": 62,
            "max_6h_f": 62,
            "min_6h_f": 62,
        }
        for _ in range(3)
    ]
    mixed_records = [
        {
            "station": "KPHX",
            "valid_utc": _utc("2023-07-14 00:00:00").replace(tzinfo=None),
            "tmpf": 62,
            "max_6h_f": 62.96,
            "min_6h_f": None,
        },
        {
            "station": "KPHX",
            "valid_utc": _utc("2023-07-14 01:00:00").replace(tzinfo=None),
            "tmpf": 62.96,
            "max_6h_f": None,
            "min_6h_f": 62,
        },
    ]

    all_int_frame = warehouse._frame_for(
        conn, "raw.asos_hourly", columns, all_int_records
    )
    mixed_frame = warehouse._frame_for(conn, "raw.asos_hourly", columns, mixed_records)

    expected_schema = {
        "station": pl.String,
        "valid_utc": pl.Datetime("us"),
        "tmpf": pl.Float64,
        "max_6h_f": pl.Float64,
        "min_6h_f": pl.Float64,
    }
    assert dict(all_int_frame.schema) == expected_schema
    assert dict(mixed_frame.schema) == expected_schema


def test_frame_for_raises_for_column_absent_from_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    columns = ["station", "valid_utc", "tmpf", "not_a_real_column"]
    records = [
        {
            "station": "KPHX",
            "valid_utc": _utc("2023-07-14 00:00:00").replace(tzinfo=None),
            "tmpf": 62.0,
            "not_a_real_column": 1,
        }
    ]

    with pytest.raises(ValueError, match="not_a_real_column"):
        warehouse._frame_for(conn, "raw.asos_hourly", columns, records)


def test_frame_for_raises_for_unmapped_duckdb_type(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute("create table raw.scratch_x (d decimal(5, 2))")
    records = [{"d": 1.5}]

    with pytest.raises(ValueError, match="d"):
        warehouse._frame_for(conn, "raw.scratch_x", ["d"], records)


def test_load_hourly_accepts_six_hour_group_after_100_rows_without_one(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """Production shape behind the 2026-09-27 halt: `tmpf` is always a
    Python float (parsers never emit int), but `max_6h_f` is `None` on 100+
    consecutive rows with no METAR 6-hour group, then a real float once one
    arrives. That Null-led column, not an Int64-led one, is what actually
    broke `load_hourly` (#41)."""
    base = _utc("2023-07-14 00:00:00")
    rows = [
        HourlyObservation(
            "KPHX", base + timedelta(hours=i), 60.1, max_6h_f=None, min_6h_f=None
        )
        for i in range(100)
    ]
    rows.append(
        HourlyObservation(
            "KPHX",
            base + timedelta(hours=100),
            60.1,
            max_6h_f=62.96,
            min_6h_f=None,
        )
    )

    warehouse.load_hourly(conn, "KPHX", date(2023, 7, 14), date(2023, 7, 19), rows)

    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (101,)
    last_max_6h_f = conn.execute(
        "select max_6h_f from raw.asos_hourly order by valid_utc desc limit 1"
    ).fetchone()
    assert last_max_6h_f == (62.96,)


def test_frame_for_rejects_value_of_wrong_type(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A value that doesn't fit the table's declared dtype must raise.

    This does not guard `strict=False`: for row-oriented dict input polars
    1.44 raises either way (teeth spec `frame_for-strict-disabled`)."""
    columns = ["station", "valid_utc", "tmpf", "max_6h_f", "min_6h_f"]
    records = [
        {
            "station": "KPHX",
            "valid_utc": _utc("2023-07-14 00:00:00").replace(tzinfo=None),
            "tmpf": "sixty",
            "max_6h_f": None,
            "min_6h_f": None,
        }
    ]

    with pytest.raises(
        pl.exceptions.ComputeError, match=r'could not append value: "sixty"'
    ):
        warehouse._frame_for(conn, "raw.asos_hourly", columns, records)
