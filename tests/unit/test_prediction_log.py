"""Live prediction log: append-only writes, sealed partitions, immutability."""

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from weather_forecast_audit import cli, warehouse
from weather_forecast_audit.prediction_log import (
    GENESIS,
    HASH_VERSION,
    SOURCES,
    LogIntegrityError,
    PartitionSealedError,
    append_partition,
    assert_immutable,
    partition_hash,
    verify_immutability,
)

pytestmark = pytest.mark.unit

D1 = date(2026, 3, 1)
D2 = date(2026, 3, 2)
D3 = date(2026, 3, 3)

_SCHEMA: dict[str, Any] = {
    "station": pl.String,
    "run_date": pl.Date,
    "runtime_utc": pl.Datetime("us"),
    "lead_day": pl.Int32,
    "variable": pl.String,
    "target_date": pl.Date,
    "source": pl.String,
    "forecast_f": pl.Float64,
    "raw_forecast_f": pl.Float64,
    "window_end_utc": pl.Datetime("us"),
    "retrained_on": pl.Date,
    "trained_through": pl.Datetime("us"),
    "fallback": pl.Boolean,
    "params": pl.String,
}

# (station, lead_day, variable, source, forecast_f, raw_forecast_f)
_KEYS = [
    ("KPHX", 1, "high", "baseline", 70.0, 69.0),
    ("KPHX", 1, "high", "challenger", 71.5, 69.0),
    ("KSEA", 2, "low", "baseline", 60.25, 59.0),
]


def _utc_naive(*parts: int) -> datetime:
    """A naive-UTC datetime, the warehouse's storage convention."""
    return datetime(*parts, tzinfo=UTC).replace(tzinfo=None)


def _now_for(run_date: date) -> datetime:
    return datetime.combine(run_date, time(12, 0), tzinfo=UTC)


def _naive(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(tzinfo=None)


def _records(
    run_date: date, *, window_end: datetime | None = None
) -> list[dict[str, Any]]:
    end = window_end or datetime.combine(run_date + timedelta(days=2), time(0, 0))
    return [
        {
            "station": station,
            "run_date": run_date,
            "runtime_utc": datetime.combine(run_date, time(6, 0)),
            "lead_day": lead,
            "variable": variable,
            "target_date": run_date + timedelta(days=lead),
            "source": source,
            "forecast_f": forecast,
            "raw_forecast_f": raw,
            "window_end_utc": end,
            "retrained_on": run_date - timedelta(days=1),
            "trained_through": datetime.combine(run_date, time(0, 0)),
            "fallback": False,
            "params": '{"window_days": 30}',
        }
        for station, lead, variable, source, forecast, raw in _KEYS
    ]


def _frame(records: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=_SCHEMA)


def _rows(run_date: date, *, window_end: datetime | None = None) -> pl.DataFrame:
    return _frame(_records(run_date, window_end=window_end))


@pytest.fixture
def conn(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(tmp_path / "log.duckdb"))
    warehouse.init_db(connection)
    return connection


def _append(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    rows: pl.DataFrame | None = None,
    now: datetime | None = None,
) -> Any:  # noqa: ANN401
    return append_partition(
        conn,
        run_date,
        _rows(run_date) if rows is None else rows,
        now=now or _now_for(run_date),
    )


def _count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"select count(*) from live.{table}").fetchone()
    assert row is not None
    return int(row[0])


def _three_partitions(conn: duckdb.DuckDBPyConnection) -> None:
    for run_date in (D1, D2, D3):
        _append(conn, run_date)


# 1. append then verify ----------------------------------------------------


def test_append_then_verify_ok(conn: duckdb.DuckDBPyConnection) -> None:
    now = _now_for(D1)
    result = _append(conn, D1, now=now)

    assert result.run_date == D1
    assert result.n_appended == 3
    assert result.n_late_rejected == 0
    assert result.late_keys == ()
    assert result.sealed is True
    assert result.partition_sha256 is not None
    assert result.chain_sha256 is not None

    report = verify_immutability(conn)
    assert report.ok
    assert report.partitions == 1
    assert report.rows == 3
    assert report.problems == ()
    assert report.head_chain_sha256 == result.chain_sha256


def test_seal_row_fields_and_issued_at_from_now(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _now_for(D1)
    backdated = _rows(D1).with_columns(
        pl.lit(_utc_naive(2020, 1, 1)).alias("issued_at")
    )  # a caller-supplied column must be ignored, not trusted
    result = _append(conn, D1, backdated, now)

    seal = conn.execute(
        "select run_date, n_rows, n_late_rejected, partition_sha256, "
        "prev_chain_sha256, chain_sha256, hash_version, issued_at, sealed_at "
        "from live.partition_seals"
    ).fetchall()
    assert len(seal) == 1
    (rd, n_rows, n_late, part, prev, chain, version, issued, sealed) = seal[0]
    assert rd == D1
    assert n_rows == 3
    assert n_late == 0
    assert part == result.partition_sha256 == partition_hash(conn, D1)[1]
    assert prev == GENESIS
    assert chain == result.chain_sha256
    assert version == HASH_VERSION == 1
    assert issued == _naive(now)
    assert sealed == _naive(now)

    issued_values = {
        r[0]
        for r in conn.execute("select issued_at from live.prediction_log").fetchall()
    }
    assert issued_values == {_naive(now)}


def test_chain_links_consecutive_partitions(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    first = _append(conn, D1)
    second = _append(conn, D2)
    prev = conn.execute(
        "select prev_chain_sha256 from live.partition_seals where run_date = ?", [D2]
    ).fetchone()
    assert prev is not None
    assert prev[0] == first.chain_sha256
    assert second.chain_sha256 != first.chain_sha256


def test_empty_log_verifies_ok_with_genesis_head(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    report = verify_immutability(conn)
    assert report.ok
    assert report.partitions == 0
    assert report.rows == 0
    assert report.head_chain_sha256 == GENESIS
    assert SOURCES == ("baseline", "challenger")


# 2. sealed partitions -----------------------------------------------------


def test_second_append_to_sealed_partition_raises(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    before = partition_hash(conn, D1)

    with pytest.raises(PartitionSealedError):
        _append(conn, D1)  # identical rows are refused too
    with pytest.raises(PartitionSealedError):
        _append(conn, D1, now=_now_for(D1) + timedelta(minutes=5))

    assert partition_hash(conn, D1) == before
    assert _count(conn, "prediction_log") == 3
    assert _count(conn, "partition_seals") == 1


def test_rows_without_a_seal_also_block_append(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    conn.execute("delete from live.partition_seals")
    with pytest.raises(PartitionSealedError):
        _append(conn, D1)
    assert _count(conn, "prediction_log") == 3


# 3. tamper -----------------------------------------------------------------

_D2 = "date '2026-03-02'"
_EXTRA_ROW = (
    "insert into live.prediction_log values ('KXXX', date '2026-03-02', "
    "timestamp '2026-03-02 06:00:00', 1, 'high', date '2026-03-03', 'baseline', "
    "50.0, 50.0, timestamp '2026-03-04 00:00:00', "
    "(select issued_at from live.partition_seals where run_date = "
    f"{_D2}), null, null, false, null)"
)

_TAMPERS: list[tuple[str, str, str]] = [
    (
        "update_forecast",
        "update live.prediction_log set forecast_f = forecast_f + 1 "
        f"where run_date = {_D2} and station = 'KPHX' and source = 'baseline'",
        "2026-03-02",
    ),
    (
        "update_issued_at",
        "update live.prediction_log set issued_at = issued_at - interval 1 day "
        f"where run_date = {_D2} and station = 'KPHX'",
        "2026-03-02",
    ),
    (
        "delete_row",
        f"delete from live.prediction_log where run_date = {_D2} and station = 'KSEA'",
        "2026-03-02",
    ),
    ("insert_extra_row", _EXTRA_ROW, "2026-03-02"),
    (
        "delete_seal",
        f"delete from live.partition_seals where run_date = {_D2}",
        "2026-03-02",
    ),
    (
        "edit_partition_sha",
        "update live.partition_seals set partition_sha256 = repeat('a', 64) "
        f"where run_date = {_D2}",
        "2026-03-02",
    ),
    (
        "edit_earlier_chain",
        "update live.partition_seals set chain_sha256 = repeat('b', 64) "
        "where run_date = date '2026-03-01'",
        "2026-03-01",
    ),
    (
        "swap_two_forecasts",
        "update live.prediction_log set forecast_f = "
        "case source when 'baseline' then 71.5 else 70.0 end "
        f"where run_date = {_D2} and station = 'KPHX' and lead_day = 1",
        "2026-03-02",
    ),
]


def test_untouched_three_partition_log_is_ok(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    report = assert_immutable(conn)
    assert report.ok
    assert report.partitions == 3
    assert report.rows == 9
    head = conn.execute(
        "select chain_sha256 from live.partition_seals order by run_date desc limit 1"
    ).fetchone()
    assert head is not None
    assert report.head_chain_sha256 == head[0]


@pytest.mark.parametrize(
    ("sql", "named_date"),
    [pytest.param(sql, named, id=name) for name, sql, named in _TAMPERS],
)
def test_tamper_is_detected_and_names_the_partition(
    conn: duckdb.DuckDBPyConnection, sql: str, named_date: str
) -> None:
    _three_partitions(conn)
    conn.execute(sql)

    report = verify_immutability(conn)
    assert not report.ok
    assert any(named_date in problem for problem in report.problems)

    with pytest.raises(LogIntegrityError) as excinfo:
        assert_immutable(conn)
    assert named_date in str(excinfo.value)


def test_all_problems_are_collected_not_just_the_first(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1 "
        "where run_date = date '2026-03-01' and station = 'KPHX' "
        "and source = 'baseline'"
    )
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1 "
        "where run_date = date '2026-03-03' and station = 'KPHX' "
        "and source = 'baseline'"
    )
    report = verify_immutability(conn)
    assert any("2026-03-01" in p for p in report.problems)
    assert any("2026-03-03" in p for p in report.problems)


# 4. late rows ----------------------------------------------------------------


def test_late_rows_are_rejected_reported_and_not_stored(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _now_for(D1)
    naive_now = _naive(now)
    records = _records(D1)
    records[0]["window_end_utc"] = naive_now  # equal to now: late (strict)
    records[1]["window_end_utc"] = naive_now - timedelta(hours=1)  # before now
    records[2]["window_end_utc"] = naive_now + timedelta(microseconds=1)  # open
    result = _append(conn, D1, _frame(records), now)

    assert result.sealed is True
    assert result.n_appended == 1
    assert result.n_late_rejected == 2
    assert set(result.late_keys) == {
        ("KPHX", 1, "high", "baseline"),
        ("KPHX", 1, "high", "challenger"),
    }
    stored = conn.execute(
        "select station, lead_day, variable, source from live.prediction_log"
    ).fetchall()
    assert stored == [("KSEA", 2, "low", "baseline")]
    seal = conn.execute(
        "select n_rows, n_late_rejected from live.partition_seals"
    ).fetchone()
    assert seal == (1, 2)
    assert verify_immutability(conn).ok


def test_all_rows_late_writes_nothing_and_does_not_seal(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _now_for(D1)
    late = _rows(D1, window_end=_naive(now) - timedelta(hours=1))
    result = _append(conn, D1, late, now)

    assert result.sealed is False
    assert result.n_appended == 0
    assert result.n_late_rejected == 3
    assert result.partition_sha256 is None
    assert result.chain_sha256 is None
    assert _count(conn, "prediction_log") == 0
    assert _count(conn, "partition_seals") == 0

    # Nothing was issued, so a timely run for the same date is still possible.
    retry = _append(conn, D1, _rows(D1), now)
    assert retry.sealed is True


# 5. validation ---------------------------------------------------------------


def _drop_params() -> pl.DataFrame:
    return _rows(D1).drop("params")


def _wrong_run_date() -> pl.DataFrame:
    records = _records(D1)
    records[1]["run_date"] = D2
    return _frame(records)


def _bad_source() -> pl.DataFrame:
    records = _records(D1)
    records[0]["source"] = "oracle"
    return _frame(records)


def _duplicate_key() -> pl.DataFrame:
    records = _records(D1)
    records.append(dict(records[0]))
    return _frame(records)


def _nan_forecast() -> pl.DataFrame:
    records = _records(D1)
    records[0]["forecast_f"] = float("nan")
    return _frame(records)


def _inf_raw() -> pl.DataFrame:
    records = _records(D1)
    records[2]["raw_forecast_f"] = float("inf")
    return _frame(records)


def _runtime_after_now() -> pl.DataFrame:
    records = _records(D1)
    records[1]["runtime_utc"] = _naive(_now_for(D1)) + timedelta(seconds=1)
    return _frame(records)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(_drop_params, id="missing_column"),
        pytest.param(_wrong_run_date, id="wrong_run_date"),
        pytest.param(_bad_source, id="unknown_source"),
        pytest.param(_duplicate_key, id="duplicate_key"),
        pytest.param(_nan_forecast, id="nan_forecast"),
        pytest.param(_inf_raw, id="inf_raw_forecast"),
        pytest.param(_runtime_after_now, id="runtime_after_now"),
    ],
)
def test_invalid_frames_raise_and_write_nothing(
    conn: duckdb.DuckDBPyConnection, build: Callable[[], pl.DataFrame]
) -> None:
    with pytest.raises(ValueError) as excinfo:
        _append(conn, D1, build())
    assert type(excinfo.value) is ValueError
    assert _count(conn, "prediction_log") == 0
    assert _count(conn, "partition_seals") == 0


def test_naive_now_raises(conn: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(ValueError, match="UTC"):
        append_partition(conn, D1, _rows(D1), now=_naive(_now_for(D1)))
    assert _count(conn, "prediction_log") == 0


def test_non_utc_now_raises(conn: duckdb.DuckDBPyConnection) -> None:
    phoenix = timezone(timedelta(hours=-7))
    with pytest.raises(ValueError, match="UTC"):
        append_partition(conn, D1, _rows(D1), now=_now_for(D1).astimezone(phoenix))
    assert _count(conn, "prediction_log") == 0


# 6. out of order -----------------------------------------------------------


def test_out_of_order_append_is_refused(conn: duckdb.DuckDBPyConnection) -> None:
    _append(conn, D3)
    with pytest.raises(ValueError, match="2026-03-03") as excinfo:
        _append(conn, D1)
    assert type(excinfo.value) is ValueError
    assert _count(conn, "prediction_log") == 3
    assert _count(conn, "partition_seals") == 1
    assert verify_immutability(conn).ok


# 7. canonicalization pin ---------------------------------------------------

_PIN_NOW = datetime(2026, 3, 2, 12, 30, 15, 250000, tzinfo=UTC)

# Independent expectation: the CANONICAL HASH rules applied literally to the
# two fixed rows below (JSON arrays written out by hand, six fractional digits
# on every timestamp, doubles in JSON repr, null/true/false literals).
_PIN_ROW_1 = (
    '["KPHX","2026-03-02","2026-03-02T06:00:00.000000",1,"high","2026-03-03",'
    '"baseline",70.0,69.5,"2026-03-04T00:00:00.000000",'
    '"2026-03-02T12:30:15.250000",null,null,false,null]'
)
_PIN_ROW_2 = (
    '["KPHX","2026-03-02","2026-03-02T06:00:00.000000",1,"high","2026-03-03",'
    '"challenger",71.25,69.5,"2026-03-04T00:00:00.000000",'
    '"2026-03-02T12:30:15.250000","2026-03-01","2026-03-02T00:00:00.125000",'
    'true,"{\\"window_days\\":30}"]'
)
_PIN_PARTITION_SHA = "0d96e11eaad12911b913248565187326a2503023c4a7d9d396563c9dceb943aa"
_PIN_CHAIN_SHA = "414085eafdf0580cc41fb8d490aaf0379c078049c9e88e3ceb2970efde6b97bb"


def _pin_frame() -> pl.DataFrame:
    base = {
        "station": "KPHX",
        "run_date": D2,
        "runtime_utc": _utc_naive(2026, 3, 2, 6, 0, 0),
        "lead_day": 1,
        "variable": "high",
        "target_date": D3,
        "raw_forecast_f": 69.5,
        "window_end_utc": _utc_naive(2026, 3, 4, 0, 0, 0),
    }
    first = {
        **base,
        "source": "baseline",
        "forecast_f": 70.0,
        "retrained_on": None,
        "trained_through": None,
        "fallback": False,
        "params": None,
    }
    second = {
        **base,
        "source": "challenger",
        "forecast_f": 71.25,
        "retrained_on": D1,
        "trained_through": _utc_naive(2026, 3, 2, 0, 0, 0, 125000),
        "fallback": True,
        "params": '{"window_days":30}',
    }
    # Deliberately handed over in reverse key order: the hash must not depend
    # on the caller's row order.
    return _frame([second, first])


def _reference_hashes() -> tuple[str, str]:
    body = "wfa-prediction-log-v1\n" + "\n".join([_PIN_ROW_1, _PIN_ROW_2])
    partition = hashlib.sha256(body.encode("utf-8")).hexdigest()
    chain = hashlib.sha256(f"{GENESIS}:2026-03-02:{partition}".encode()).hexdigest()
    return partition, chain


def test_canonicalization_pin(conn: duckdb.DuckDBPyConnection) -> None:
    result = append_partition(conn, D2, _pin_frame(), now=_PIN_NOW)
    reference_partition, reference_chain = _reference_hashes()

    assert result.partition_sha256 == reference_partition
    assert result.chain_sha256 == reference_chain
    assert partition_hash(conn, D2) == (2, reference_partition)
    # Pinned literals: a change here is a change of the public hash contract
    # and requires a HASH_VERSION bump.
    assert reference_partition == _PIN_PARTITION_SHA
    assert reference_chain == _PIN_CHAIN_SHA


# 8. tz handling --------------------------------------------------------------


def test_tz_aware_rows_hash_like_naive_rows(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    other = duckdb.connect(str(tmp_path / "other.duckdb"))
    warehouse.init_db(other)
    now = _now_for(D1)
    naive_result = _append(conn, D1, _rows(D1), now)

    aware = _rows(D1).with_columns(
        pl.col("runtime_utc").dt.replace_time_zone("UTC"),
        pl.col("window_end_utc").dt.replace_time_zone("UTC"),
        pl.col("trained_through").dt.replace_time_zone("UTC"),
    )
    aware_result = _append(other, D1, aware, now)
    assert aware_result.partition_sha256 == naive_result.partition_sha256
    assert aware_result.chain_sha256 == naive_result.chain_sha256

    # A non-UTC zone is converted to UTC, not stripped.
    phoenix = _rows(D1).with_columns(
        pl.col("runtime_utc")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/Phoenix"),
        pl.col("window_end_utc")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/Phoenix"),
        pl.col("trained_through")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/Phoenix"),
    )
    third = duckdb.connect(str(tmp_path / "third.duckdb"))
    warehouse.init_db(third)
    phoenix_result = _append(third, D1, phoenix, now)
    assert phoenix_result.partition_sha256 == naive_result.partition_sha256
    other.close()
    third.close()


# 9. CLI ------------------------------------------------------------------------


def _cli_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "cli.duckdb"
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    return db_path


def test_cli_log_verify_ok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = _cli_db(tmp_path, monkeypatch)
    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        _three_partitions(conn)
        head = verify_immutability(conn).head_chain_sha256

    cli.main(["log", "verify"])  # exit 0: returns without SystemExit

    out = capsys.readouterr().out
    assert out.startswith("OK partitions=3 rows=9 head=")
    assert head[:12] in out


def test_cli_log_verify_fails_after_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = _cli_db(tmp_path, monkeypatch)
    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        _three_partitions(conn)
        conn.execute(
            "update live.prediction_log set forecast_f = 0 "
            "where run_date = date '2026-03-02' and station = 'KSEA'"
        )

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["log", "verify"])

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL")
    assert "2026-03-02" in out


def test_cli_log_verify_on_database_without_live_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = _cli_db(tmp_path, monkeypatch)
    with duckdb.connect(str(db_path)) as conn:
        conn.execute("create schema raw")  # an old database: no `live` schema

    cli.main(["log", "verify"])

    out = capsys.readouterr().out
    assert out.startswith("OK partitions=0 rows=0 head=000000000000")


# 10. init_db ---------------------------------------------------------------------


def test_init_db_creates_log_tables_and_is_idempotent(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    warehouse.init_db(conn)  # second call must neither raise nor drop rows

    tables = {
        row[0]
        for row in conn.execute(
            "select table_name from information_schema.tables "
            "where table_schema = 'live'"
        ).fetchall()
    }
    assert tables == {"prediction_log", "partition_seals"}
    assert _count(conn, "prediction_log") == 3
    assert verify_immutability(conn).ok


# 11. teeth additions: guards the first suite did not exercise ----------------


def _forge_consistent_seals(conn: duckdb.DuckDBPyConnection) -> None:
    """Rewrite every seal's counts, hashes and links from the stored rows.

    This is a careful forger: afterwards the only thing that can still give
    the edit away is a check that is not a pure function of the row bytes.
    Hashing is re-done here independently of the module's private helpers.
    """
    dates = [
        r[0]
        for r in conn.execute(
            "select run_date from live.partition_seals order by run_date"
        ).fetchall()
    ]
    prev = GENESIS
    for run_date in dates:
        n_rows, part = partition_hash(conn, run_date)
        chain = hashlib.sha256(
            f"{prev}:{run_date.isoformat()}:{part}".encode()
        ).hexdigest()
        conn.execute(
            "update live.partition_seals set n_rows = ?, partition_sha256 = ?, "
            "prev_chain_sha256 = ?, chain_sha256 = ? where run_date = ?",
            [n_rows, part, prev, chain, run_date],
        )
        prev = chain


def test_extra_tamper_shapes_are_detected(conn: duckdb.DuckDBPyConnection) -> None:
    # A seal whose row count was edited but whose hash still matches.
    _three_partitions(conn)
    conn.execute(
        "update live.partition_seals set n_rows = n_rows + 1 "
        "where run_date = date '2026-03-02'"
    )
    problems = verify_immutability(conn).problems
    assert any("2026-03-02" in p and "seal records" in p for p in problems)


def test_edited_head_chain_is_detected(conn: duckdb.DuckDBPyConnection) -> None:
    # Nothing follows the head seal, so only recomputing its chain can see this.
    _three_partitions(conn)
    conn.execute(
        "update live.partition_seals set chain_sha256 = repeat('c', 64) "
        "where run_date = date '2026-03-03'"
    )
    problems = verify_immutability(conn).problems
    assert any("2026-03-03" in p and "does not match its inputs" in p for p in problems)


def test_removing_a_middle_partition_whole_breaks_the_chain(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute("delete from live.prediction_log where run_date = date '2026-03-02'")
    conn.execute("delete from live.partition_seals where run_date = date '2026-03-02'")

    report = verify_immutability(conn)
    assert not report.ok
    assert any("2026-03-03" in p and "chain broken" in p for p in report.problems)


def test_removing_the_newest_partition_whole_verifies_but_moves_the_head(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # The documented limit: self-consistency cannot see a truncated tail.
    # What does see it is the published head, which must therefore move.
    _three_partitions(conn)
    published_head = verify_immutability(conn).head_chain_sha256
    conn.execute("delete from live.prediction_log where run_date = date '2026-03-03'")
    conn.execute("delete from live.partition_seals where run_date = date '2026-03-03'")

    report = verify_immutability(conn)
    assert report.ok
    assert report.partitions == 2
    assert report.head_chain_sha256 != published_head


def test_forged_seals_still_expose_issued_at_different_from_seal(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute(
        "update live.prediction_log set issued_at = issued_at - interval 1 day "
        "where run_date = date '2026-03-03'"
    )
    _forge_consistent_seals(conn)

    report = verify_immutability(conn)
    assert not report.ok
    assert any(
        "2026-03-03" in p and "different from the seal's issued_at" in p
        for p in report.problems
    )
    assert not any("hash" in p or "chain" in p for p in report.problems)


@pytest.mark.parametrize("offset_seconds", [0, 1])
def test_forged_issued_at_not_before_window_end_is_detected(
    conn: duckdb.DuckDBPyConnection, offset_seconds: int
) -> None:
    # issued_at == window_end_utc is already too late (strict), as is after.
    _three_partitions(conn)
    conn.execute(
        "update live.prediction_log set issued_at = window_end_utc + "
        f"interval {offset_seconds} second where run_date = date '2026-03-03'"
    )
    conn.execute(
        "update live.partition_seals set issued_at = (select min(issued_at) "
        "from live.prediction_log where run_date = date '2026-03-03') "
        "where run_date = date '2026-03-03'"
    )
    _forge_consistent_seals(conn)

    report = verify_immutability(conn)
    assert not report.ok
    assert [p for p in report.problems if "2026-03-03" in p] == [
        "2026-03-03: 3 row(s) have issued_at at or after window_end_utc"
    ]


def test_unsupported_hash_version_is_reported(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute(
        "update live.partition_seals set hash_version = 2 "
        "where run_date = date '2026-03-02'"
    )
    report = verify_immutability(conn)
    assert not report.ok
    assert any(
        "2026-03-02" in p and "unsupported hash_version 2" in p for p in report.problems
    )


def test_seal_without_rows_still_blocks_append(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    conn.execute("delete from live.prediction_log")
    with pytest.raises(PartitionSealedError):
        _append(conn, D1)
    assert _count(conn, "prediction_log") == 0
    assert _count(conn, "partition_seals") == 1


def test_failed_seal_rolls_back_rows_and_leaves_connection_usable(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # Make the seal insert fail after the rows went in: the whole append must
    # vanish, and the connection must not be stuck in an aborted transaction.
    conn.execute("drop table live.partition_seals")
    conn.execute(
        "create table live.partition_seals (run_date date primary key, "
        "n_rows integer not null, n_late_rejected integer not null, "
        "partition_sha256 varchar not null, prev_chain_sha256 varchar not null, "
        "chain_sha256 varchar not null, hash_version integer not null, "
        "issued_at timestamp not null, sealed_at timestamp not null, "
        "check (n_rows < 0))"
    )
    with pytest.raises(duckdb.ConstraintException):
        _append(conn, D1)

    assert _count(conn, "prediction_log") == 0
    assert _count(conn, "partition_seals") == 0
    conn.execute("drop table live.partition_seals")
    warehouse.init_db(conn)
    retry = _append(conn, D1)
    assert retry.sealed is True
    assert verify_immutability(conn).ok


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_forecast_is_refused_by_the_validator_itself(
    conn: duckdb.DuckDBPyConnection, bad: float
) -> None:
    # The hash also refuses NaN, which would mask a missing validator if the
    # message were not pinned.
    records = _records(D1)
    records[1]["raw_forecast_f"] = bad
    with pytest.raises(ValueError, match="must be finite"):
        _append(conn, D1, _frame(records))
    assert _count(conn, "prediction_log") == 0


def test_nan_in_a_stored_cell_is_refused_by_the_hash(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    conn.execute(
        "update live.prediction_log set forecast_f = 'NaN'::double "
        "where station = 'KSEA'"
    )
    with pytest.raises(ValueError, match="JSON"):
        partition_hash(conn, D1)


def test_null_in_a_required_column_is_a_value_error(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    records = _records(D1)
    records[0]["forecast_f"] = None
    with pytest.raises(ValueError, match="must not contain nulls") as excinfo:
        _append(conn, D1, _frame(records))
    assert type(excinfo.value) is ValueError
    assert _count(conn, "prediction_log") == 0


def test_uncastable_column_is_a_value_error(conn: duckdb.DuckDBPyConnection) -> None:
    bad = _rows(D1).with_columns(pl.lit("not-a-number").alias("lead_day"))
    with pytest.raises(ValueError, match="cannot be cast") as excinfo:
        _append(conn, D1, bad)
    assert type(excinfo.value) is ValueError
    assert _count(conn, "prediction_log") == 0


def test_runtime_exactly_now_is_accepted(conn: duckdb.DuckDBPyConnection) -> None:
    now = _now_for(D1)
    records = _records(D1)
    records[0]["runtime_utc"] = _naive(now)
    assert _append(conn, D1, _frame(records), now).sealed is True


def test_late_keys_are_sorted_whatever_the_input_order(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = _now_for(D1)
    late = _rows(D1, window_end=_naive(now))
    reversed_rows = late.reverse()
    result = _append(conn, D1, reversed_rows, now)
    assert result.late_keys == (
        ("KPHX", 1, "high", "baseline"),
        ("KPHX", 1, "high", "challenger"),
        ("KSEA", 2, "low", "baseline"),
    )


def test_sub_second_now_is_kept_in_issued_at_and_seal(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime(2026, 3, 1, 12, 0, 0, 123456, tzinfo=UTC)
    _append(conn, D1, now=now)
    stored = conn.execute(
        "select distinct issued_at from live.prediction_log"
    ).fetchall()
    assert stored == [(_naive(now),)]
    assert verify_immutability(conn).ok


def _pin_key_order_frame() -> pl.DataFrame:
    def row(station: str, lead: int, variable: str, source: str) -> dict[str, Any]:
        return {
            "station": station,
            "run_date": D1,
            "runtime_utc": _utc_naive(2026, 3, 1, 6, 0, 0),
            "lead_day": lead,
            "variable": variable,
            "target_date": D1 + timedelta(days=lead),
            "source": source,
            "forecast_f": 1.5,
            "raw_forecast_f": 2.5,
            "window_end_utc": _utc_naive(2026, 3, 5, 0, 0, 0),
            "retrained_on": None,
            "trained_through": None,
            "fallback": False,
            "params": None,
        }

    # Handed over scrambled: no key order but the documented one reproduces
    # the expected body below.
    return _frame(
        [
            row("KBBB", 1, "high", "baseline"),
            row("KAAA", 2, "low", "challenger"),
            row("KAAA", 1, "low", "baseline"),
            row("KAAA", 2, "high", "baseline"),
            row("KAAA", 2, "low", "baseline"),
        ]
    )


def test_hash_orders_rows_by_station_lead_variable_source(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    append_partition(conn, D1, _pin_key_order_frame(), now=now)

    def line(station: str, lead: int, variable: str, source: str) -> str:
        target = f"2026-03-0{1 + lead}"
        return (
            f'["{station}","2026-03-01","2026-03-01T06:00:00.000000",{lead},'
            f'"{variable}","{target}","{source}",1.5,2.5,'
            '"2026-03-05T00:00:00.000000","2026-03-01T12:00:00.000000",'
            "null,null,false,null]"
        )

    expected_order = [
        line("KAAA", 1, "low", "baseline"),
        line("KAAA", 2, "high", "baseline"),
        line("KAAA", 2, "low", "baseline"),
        line("KAAA", 2, "low", "challenger"),
        line("KBBB", 1, "high", "baseline"),
    ]
    body = "wfa-prediction-log-v1\n" + "\n".join(expected_order)
    assert partition_hash(conn, D1) == (
        5,
        hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )


def test_hash_escapes_non_ascii_and_awkward_strings(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    now = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
    records = _records(D1)[:1]
    records[0]["station"] = 'K,"X'
    records[0]["params"] = "café ☃\nline\\two"
    result = append_partition(conn, D1, _frame(records), now=now)

    expected_row = (
        '["K,\\"X","2026-03-01","2026-03-01T06:00:00.000000",1,"high",'
        '"2026-03-02","baseline",70.0,69.0,"2026-03-03T00:00:00.000000",'
        '"2026-03-01T12:00:00.000000","2026-02-28","2026-03-01T00:00:00.000000",'
        'false,"caf\\u00e9 \\u2603\\nline\\\\two"]'
    )
    body = "wfa-prediction-log-v1\n" + expected_row
    assert result.partition_sha256 == hashlib.sha256(body.encode()).hexdigest()
    assert verify_immutability(conn).ok


def test_engine_enforces_key_uniqueness_independently_of_the_writer(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _append(conn, D1)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "insert into live.prediction_log select * from live.prediction_log "
            "where station = 'KSEA'"
        )
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "insert into live.partition_seals select * from live.partition_seals"
        )


def test_issued_at_problems_are_reported_alongside_hash_problems(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute(
        "update live.prediction_log set issued_at = issued_at - interval 1 day "
        "where run_date = date '2026-03-03'"
    )
    _forge_consistent_seals(conn)
    # After the forger is done, a plain row edit in an earlier partition.
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1 "
        "where run_date = date '2026-03-01' and station = 'KPHX'"
    )

    problems = verify_immutability(conn).problems
    assert any("2026-03-01" in p and "partition hash mismatch" in p for p in problems)
    assert any("2026-03-03" in p and "issued_at different" in p for p in problems)


# 12. defects found by the teeth probe, now fixed and pinned ------------------


def test_datetime_run_date_argument_is_refused_and_writes_nothing(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    # datetime is-a date, so the type hint alone does not stop it.
    as_datetime = datetime.combine(D1, time(0, 0))
    with pytest.raises(ValueError, match="datetime"):
        append_partition(conn, as_datetime, _rows(D1), now=_now_for(D1))
    assert conn.execute("select count(*) from live.prediction_log").fetchone() == (0,)
    assert verify_immutability(conn).ok


def test_verify_reports_a_non_finite_stored_cell_instead_of_raising(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _three_partitions(conn)
    conn.execute(
        "update live.prediction_log set forecast_f = 'NaN'::double "
        "where run_date = date '2026-03-01' and station = 'KSEA'"
    )
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1 "
        "where run_date = date '2026-03-03' and station = 'KPHX'"
    )
    report = verify_immutability(conn)
    assert not report.ok
    assert any("2026-03-01" in p for p in report.problems)
    assert any("2026-03-03" in p for p in report.problems)
