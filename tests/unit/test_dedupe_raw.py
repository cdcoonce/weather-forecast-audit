"""One-time repair of source duplicates already loaded into the raw tables.

The loaders no longer let a duplicated key in, so every pre-existing
duplicate here is inserted directly with SQL, the way the old loaders left
them in the production database.
"""

import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from weather_forecast_audit import warehouse
from weather_forecast_audit.warehouse import DedupeReport, dedupe_raw_table

pytestmark = [pytest.mark.unit, pytest.mark.io]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dedupe_raw_once.py"
ASOS = "raw.asos_hourly"
NBS = "raw.nbs_guidance"
WINDOWS = "raw.resolved_windows"
ALL_TABLES = (ASOS, NBS, WINDOWS)


@pytest.fixture
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    warehouse.init_db(connection)
    return connection


def _asos(
    conn: duckdb.DuckDBPyConnection,
    valid: str,
    tmpf: float | None,
    max_6h: float | None = None,
    min_6h: float | None = None,
    station: str = "KPHX",
) -> None:
    conn.execute(
        "insert into raw.asos_hourly values (?, ?, ?, ?, ?)",
        [station, valid, tmpf, max_6h, min_6h],
    )


def _nbs(
    conn: duckdb.DuckDBPyConnection,
    runtime: str = "2023-07-14 13:00:00",
    ftime: str = "2023-07-15 00:00:00",
    txn: float | None = 118.0,
) -> None:
    conn.execute(
        "insert into raw.nbs_guidance values (?, ?, ?, 13, ?, 1.0, 100.0)",
        ["KPHX", runtime, ftime, txn],
    )


def _window(
    conn: duckdb.DuckDBPyConnection,
    runtime: str = "2023-07-14 13:00:00",
    ftime: str = "2023-07-15 00:00:00",
) -> None:
    conn.execute(
        "insert into raw.resolved_windows values "
        "(?, ?, ?, 'high', '2023-07-15', 1, '2023-07-15 07:00:00', "
        "'2023-07-16 07:00:00', 111.0, 24, 24, 24, true, 'metar', 4, 110.0)",
        ["KPHX", runtime, ftime],
    )


def _rows(conn: duckdb.DuckDBPyConnection, table: str) -> list[tuple[object, ...]]:
    return conn.execute(f"select * from {table} order by all").fetchall()  # noqa: S608


def _snapshot(conn: duckdb.DuckDBPyConnection) -> dict[str, list[tuple[object, ...]]]:
    return {table: _rows(conn, table) for table in ALL_TABLES}


def _asos_for(conn: duckdb.DuckDBPyConnection, valid: str) -> list[tuple[object, ...]]:
    return conn.execute(
        "select tmpf, max_6h_f, min_6h_f from raw.asos_hourly "
        "where station = 'KPHX' and valid_utc = ?",
        [valid],
    ).fetchall()


def test_asos_valued_row_then_null_row_keeps_the_value(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0, max_6h=104.0)
    _asos(conn, "2023-07-14 00:00:00", None)

    report = dedupe_raw_table(conn, ASOS, apply=True)

    assert _asos_for(conn, "2023-07-14 00:00:00") == [(100.0, 104.0, None)]
    assert report == DedupeReport(ASOS, 2, 1, 1, True)


def test_asos_null_row_first_then_valued_row_keeps_the_value(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 00:00:00", 100.0, min_6h=95.0)

    dedupe_raw_table(conn, ASOS, apply=True)

    assert _asos_for(conn, "2023-07-14 00:00:00") == [(100.0, None, 95.0)]


def test_asos_columns_are_merged_across_the_two_rows(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0, max_6h=None, min_6h=None)
    _asos(conn, "2023-07-14 00:00:00", None, max_6h=104.0, min_6h=None)

    dedupe_raw_table(conn, ASOS, apply=True)

    assert _asos_for(conn, "2023-07-14 00:00:00") == [(100.0, 104.0, None)]


def test_asos_exact_copies_collapse_to_one_row(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0, max_6h=104.0)
    _asos(conn, "2023-07-14 00:00:00", 100.0, max_6h=104.0)

    dedupe_raw_table(conn, ASOS, apply=True)

    assert _asos_for(conn, "2023-07-14 00:00:00") == [(100.0, 104.0, None)]


def test_three_copies_one_valued_two_null_collapse_to_the_valued_row(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 00:00:00", 98.0)
    _asos(conn, "2023-07-14 00:00:00", None)

    report = dedupe_raw_table(conn, ASOS, apply=True)

    assert _asos_for(conn, "2023-07-14 00:00:00") == [(98.0, None, None)]
    assert (report.rows_before, report.rows_after, report.keys_collapsed) == (3, 1, 1)


def test_nbs_guidance_exact_copies_collapse_to_one_row(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _nbs(conn)
    _nbs(conn)

    report = dedupe_raw_table(conn, NBS, apply=True)

    assert _rows(conn, NBS) == [
        (
            "KPHX",
            _ts("2023-07-14 13:00:00"),
            _ts("2023-07-15 00:00:00"),
            13,
            118.0,
            1.0,
            100.0,
        )
    ]
    assert (report.rows_before, report.rows_after) == (2, 1)


def test_resolved_windows_exact_copies_collapse_to_one_row(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _window(conn)
    _window(conn)

    report = dedupe_raw_table(conn, WINDOWS, apply=True)

    assert len(_rows(conn, WINDOWS)) == 1
    assert (report.rows_before, report.rows_after) == (2, 1)


def _ts(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007


def _seed_conflict(conn: duckdb.DuckDBPyConnection) -> None:
    """A repairable duplicate in every table plus one real asos conflict."""
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 01:00:00", 70.0)
    _asos(conn, "2023-07-14 01:00:00", 71.0)
    _asos(conn, "2023-07-14 02:00:00", 90.0)
    _nbs(conn)
    _nbs(conn)
    _window(conn)
    _window(conn)


def test_real_conflict_raises_naming_the_key_and_writes_nothing(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _seed_conflict(conn)
    before = _snapshot(conn)

    with pytest.raises(ValueError, match="conflicting non-null values") as dry:
        dedupe_raw_table(conn, ASOS, apply=False)
    assert _snapshot(conn) == before

    with pytest.raises(ValueError, match="conflicting non-null values") as applied:
        dedupe_raw_table(conn, ASOS, apply=True)
    assert _snapshot(conn) == before

    for info in (dry, applied):
        text = str(info.value)
        assert ASOS in text
        assert "station=KPHX" in text
        assert "valid_utc=2023-07-14 01:00:00" in text
        assert "tmpf" in text
        assert "70.0" in text
        assert "71.0" in text


def test_dry_run_writes_nothing_and_predicts_what_apply_produces(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 01:00:00", 90.0)
    _asos(conn, "2023-07-14 02:00:00", 80.0)
    _asos(conn, "2023-07-14 02:00:00", 80.0)
    _asos(conn, "2023-07-14 02:00:00", None)
    before = _snapshot(conn)

    predicted = dedupe_raw_table(conn, ASOS, apply=False)

    assert _snapshot(conn) == before
    assert predicted == DedupeReport(ASOS, 6, 3, 2, False)

    applied = dedupe_raw_table(conn, ASOS, apply=True)

    assert applied == DedupeReport(ASOS, 6, 3, 2, True)
    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (3,)


def test_dry_run_works_on_a_read_only_connection(tmp_path: Path) -> None:
    path = tmp_path / "ro.duckdb"
    with duckdb.connect(str(path)) as writer:
        warehouse.init_db(writer)
        _asos(writer, "2023-07-14 00:00:00", 100.0)
        _asos(writer, "2023-07-14 00:00:00", None)

    with duckdb.connect(str(path), read_only=True) as reader:
        report = dedupe_raw_table(reader, ASOS, apply=False)

    assert report == DedupeReport(ASOS, 2, 1, 1, False)


def test_apply_leaves_every_non_duplicated_row_identical(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    for hour in range(1, 10):
        _asos(conn, f"2023-07-14 {hour:02d}:00:00", 60.0 + hour, max_6h=70.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 00:00:00", 99.0)
    _asos(conn, "2023-07-14 00:00:00", 99.0)
    _asos(conn, "2023-07-14 05:00:00", 61.0, station="KTUS")
    keep = (
        "select * from raw.asos_hourly where valid_utc <> '2023-07-14 00:00:00' "
        "order by all"
    )
    before_keep = conn.execute(keep).fetchall()
    rows_before = conn.execute("select count(*) from raw.asos_hourly").fetchone()

    report = dedupe_raw_table(conn, ASOS, apply=True)

    assert conn.execute(keep).fetchall() == before_keep
    assert rows_before is not None
    assert report.rows_before == rows_before[0] == 13
    assert report.rows_after == 11  # 13 rows less the 2 extra copies of one key
    assert conn.execute("select count(*) from raw.asos_hourly").fetchone() == (11,)


def test_apply_touches_only_the_named_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _seed_conflict(conn)
    conn.execute("delete from raw.asos_hourly where valid_utc = '2023-07-14 01:00:00'")
    before_nbs = _rows(conn, NBS)
    before_windows = _rows(conn, WINDOWS)

    dedupe_raw_table(conn, ASOS, apply=True)

    assert _rows(conn, NBS) == before_nbs
    assert _rows(conn, WINDOWS) == before_windows


def test_second_apply_is_a_no_op(conn: duckdb.DuckDBPyConnection) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 01:00:00", 90.0)
    dedupe_raw_table(conn, ASOS, apply=True)
    after_first = _snapshot(conn)

    again = dedupe_raw_table(conn, ASOS, apply=True)

    assert again == DedupeReport(ASOS, 2, 2, 0, True)
    assert _snapshot(conn) == after_first


def test_clean_empty_table_reports_zero(conn: duckdb.DuckDBPyConnection) -> None:
    assert dedupe_raw_table(conn, NBS, apply=True) == DedupeReport(NBS, 0, 0, 0, True)


@pytest.mark.parametrize("table", ["raw.cli_daily", "raw.nope", "asos_hourly", ""])
def test_unknown_table_raises_before_any_query(
    conn: duckdb.DuckDBPyConnection, table: str
) -> None:
    with pytest.raises(ValueError, match="no per-key repair"):
        dedupe_raw_table(conn, table, apply=True)


def test_failed_count_verification_rolls_back(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 01:00:00", 90.0)
    _asos(conn, "2023-07-14 01:00:00", None)
    before = _snapshot(conn)

    def lose_a_key(
        table: str,
        key_columns: tuple[str, ...],
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        real = _real_collapse(table, key_columns, records)
        return real[:-1]  # silently loses one whole key

    monkeypatch.setattr(warehouse, "_collapse_by_key", lose_a_key)

    with pytest.raises(RuntimeError, match="verification"):
        dedupe_raw_table(conn, ASOS, apply=True)

    assert _snapshot(conn) == before


def test_failed_duplicate_verification_rolls_back(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _asos(conn, "2023-07-14 01:00:00", 90.0)
    _asos(conn, "2023-07-14 01:00:00", None)
    before = _snapshot(conn)

    def keep_one_key_unmerged(
        table: str,
        key_columns: tuple[str, ...],
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        # Same row count as a correct merge, but key 00:00 stays duplicated
        # and key 01:00 vanishes.
        return [r for r in records if r["valid_utc"] == _ts("2023-07-14 00:00:00")]

    monkeypatch.setattr(warehouse, "_collapse_by_key", keep_one_key_unmerged)

    with pytest.raises(RuntimeError, match="duplicate"):
        dedupe_raw_table(conn, ASOS, apply=True)

    assert _snapshot(conn) == before


def test_connection_is_usable_after_a_failed_apply(
    conn: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    monkeypatch.setattr(
        warehouse, "_collapse_by_key", lambda table, key_columns, records: records
    )
    with pytest.raises(RuntimeError):
        dedupe_raw_table(conn, ASOS, apply=True)
    monkeypatch.undo()

    # No transaction is left open: a fresh begin works and the repair succeeds.
    report = dedupe_raw_table(conn, ASOS, apply=True)

    assert report.rows_after == 1


def _start_statement_log(conn: duckdb.DuckDBPyConnection) -> None:
    """Have DuckDB itself log every statement the connection runs.

    The merge raising before any write is a claim about *order*; the table
    after a rollback cannot tell "never wrote" from "wrote and rolled
    back", so these tests read the statement log instead.
    """
    conn.execute("call enable_logging('QueryLog')")


def _statements(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        "select message from duckdb_logs where type = 'QueryLog' order by timestamp"
    ).fetchall()
    normalised = [" ".join(str(message).split()).lower() for (message,) in rows]
    return [s for s in normalised if "duckdb_logs" not in s]


def _writes(statements: list[str]) -> list[str]:
    return [s for s in statements if s.startswith(("delete", "insert"))]


def test_a_conflict_is_found_before_any_delete_or_insert_is_issued(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _seed_conflict(conn)
    _start_statement_log(conn)

    with pytest.raises(ValueError, match="conflicting non-null values"):
        dedupe_raw_table(conn, ASOS, apply=True)

    statements = _statements(conn)
    assert statements  # the run was observed
    assert _writes(statements) == []


def test_a_dry_run_issues_only_selects(conn: duckdb.DuckDBPyConnection) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 00:00:00", None)
    _start_statement_log(conn)

    dedupe_raw_table(conn, ASOS, apply=False)

    statements = _statements(conn)
    assert statements
    assert all(s.startswith("select") for s in statements)


def test_a_table_without_duplicates_is_not_written_to(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _asos(conn, "2023-07-14 00:00:00", 100.0)
    _asos(conn, "2023-07-14 01:00:00", None)
    _start_statement_log(conn)

    report = dedupe_raw_table(conn, ASOS, apply=True)

    assert report == DedupeReport(ASOS, 2, 2, 0, True)
    statements = _statements(conn)
    assert statements
    assert _writes(statements) == []


def test_an_unknown_table_is_refused_before_the_connection_is_touched(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _start_statement_log(conn)

    with pytest.raises(ValueError, match="no per-key repair"):
        dedupe_raw_table(conn, "raw.cli_daily", apply=True)

    assert _statements(conn) == []


@pytest.mark.parametrize("table", ALL_TABLES)
def test_a_dry_run_on_a_table_without_duplicates_is_not_reported_applied(
    conn: duckdb.DuckDBPyConnection, table: str
) -> None:
    if table == ASOS:
        _asos(conn, "2023-07-14 00:00:00", 100.0)

    report = dedupe_raw_table(conn, table, apply=False)

    expected_rows = 1 if table == ASOS else 0
    assert report == DedupeReport(table, expected_rows, expected_rows, 0, False)


def test_exact_copies_holding_nan_collapse_instead_of_conflicting(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    for _ in range(2):
        conn.execute(
            "insert into raw.asos_hourly values "
            "('KPHX', '2023-07-14 00:00:00', 'nan'::double, 104.0, null)"
        )

    report = dedupe_raw_table(conn, ASOS, apply=True)

    rows = _asos_for(conn, "2023-07-14 00:00:00")
    assert len(rows) == 1
    tmpf, max_6h, min_6h = rows[0]
    assert isinstance(tmpf, float)
    assert math.isnan(tmpf)
    assert (max_6h, min_6h) == (104.0, None)
    assert (report.rows_before, report.rows_after) == (2, 1)


def _real_collapse(
    table: str,
    key_columns: tuple[str, ...],
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    # The un-patched function, bound at import time of this test module.
    return _ORIGINAL_COLLAPSE(table, key_columns, records)


_ORIGINAL_COLLAPSE = warehouse._collapse_by_key  # noqa: SLF001


# -- the CLI -------------------------------------------------------------


def _make_db(path: Path, *, conflict: bool = False) -> None:
    with duckdb.connect(str(path)) as db:
        warehouse.init_db(db)
        _asos(db, "2023-07-14 00:00:00", 100.0)
        _asos(db, "2023-07-14 00:00:00", None)
        _nbs(db)
        _nbs(db)
        _window(db)
        _window(db)
        if conflict:
            _asos(db, "2023-07-14 01:00:00", 70.0)
            _asos(db, "2023-07-14 01:00:00", 71.0)


def _run(*args: str, env_db: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "WFA_DUCKDB_PATH"}
    if env_db is not None:
        env["WFA_DUCKDB_PATH"] = env_db
    return subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _db_snapshot(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with duckdb.connect(str(path), read_only=True) as db:
        return _snapshot(db)


def test_cli_defaults_to_a_dry_run(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)
    before = _db_snapshot(db)

    result = _run("--db", str(db))

    assert result.returncode == 0, result.stderr
    assert _db_snapshot(db) == before
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 3
    assert all(line.endswith("[DRY RUN]") for line in lines)
    assert "raw.asos_hourly  2 -> 1  (keys_collapsed=1)  [DRY RUN]" in lines


def test_cli_apply_repairs_the_database(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)

    result = _run("--db", str(db), "--apply")

    assert result.returncode == 0, result.stderr
    assert "raw.asos_hourly  2 -> 1  (keys_collapsed=1)  [APPLIED]" in result.stdout
    after = _db_snapshot(db)
    assert all(len(rows) == 1 for rows in after.values())
    assert after[ASOS][0][2] == 100.0


def test_cli_reads_the_path_from_the_environment(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)

    result = _run(env_db=str(db))

    assert result.returncode == 0, result.stderr
    assert "[DRY RUN]" in result.stdout


def test_cli_table_option_limits_the_run(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)

    result = _run("--db", str(db), "--apply", "--table", NBS)

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 1
    after = _db_snapshot(db)
    assert len(after[NBS]) == 1
    assert len(after[ASOS]) == 2
    assert len(after[WINDOWS]) == 2


def test_cli_missing_database_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "nope.duckdb"

    result = _run("--db", str(missing), "--apply")

    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert not missing.exists()


def test_cli_without_a_path_exits_2() -> None:
    result = _run()

    assert result.returncode == 2
    assert "WFA_DUCKDB_PATH" in result.stderr


def test_cli_unknown_table_exits_2(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)

    result = _run("--db", str(db), "--table", "raw.cli_daily")

    assert result.returncode == 2
    assert "raw.cli_daily" in result.stderr


def test_cli_conflict_exits_1_and_writes_nothing(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db, conflict=True)
    before = _db_snapshot(db)

    result = _run("--db", str(db), "--apply")

    assert result.returncode == 1
    assert "valid_utc=2023-07-14 01:00:00" in result.stderr
    assert "tmpf" in result.stderr
    # asos is the first table, so nothing was applied before the conflict.
    assert _db_snapshot(db) == before


def test_cli_conflict_leaves_earlier_tables_applied_and_says_so(
    tmp_path: Path,
) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db, conflict=True)
    before = _db_snapshot(db)

    result = _run("--db", str(db), "--apply", "--table", NBS, "--table", ASOS)

    assert result.returncode == 1
    after = _db_snapshot(db)
    assert len(after[NBS]) == 1
    assert after[ASOS] == before[ASOS]
    assert after[WINDOWS] == before[WINDOWS]
    assert "already applied" in result.stderr
    assert NBS in result.stderr


def test_cli_dry_run_on_a_clean_database_says_dry_run(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    with duckdb.connect(str(db)) as writer:
        warehouse.init_db(writer)
        _asos(writer, "2023-07-14 00:00:00", 100.0)

    result = _run("--db", str(db))

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 3
    assert all(line.endswith("[DRY RUN]") for line in lines)
    assert "raw.asos_hourly  1 -> 1  (keys_collapsed=0)  [DRY RUN]" in lines


def test_cli_dry_run_works_on_a_database_file_it_cannot_write(
    tmp_path: Path,
) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)
    db.chmod(0o444)

    result = _run("--db", str(db))

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[DRY RUN]") == 3


def test_cli_a_repeated_table_is_repaired_and_reported_once(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    _make_db(db)

    result = _run("--db", str(db), "--apply", "--table", NBS, "--table", NBS)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines() == [
        "raw.nbs_guidance  2 -> 1  (keys_collapsed=1)  [APPLIED]"
    ]
