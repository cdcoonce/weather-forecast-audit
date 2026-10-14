"""Append-only log of live predictions, sealed per partition and hash-chained.

`raw.model_predictions` is a rebuildable cache: `load_predictions` deletes and
re-inserts it, which is exactly right for a backfill and exactly wrong for a
claim of the form "this forecast existed before the weather did". The `live`
schema holds the other kind of record. A prediction is written once, with
the writer's own clock as `issued_at`, strictly before its verification
window closes, and the partition it belongs to (one `run_date`) is then
sealed: its rows are hashed, and the hash is chained to the previous
partition's. `verify_immutability` recomputes everything from the stored
rows, so any later UPDATE, DELETE or INSERT shows up as a named mismatch.

Nothing here fits a model or reads `raw.*`; callers hand in finished rows.

The hash is computed in Python, not in DuckDB SQL: a float-to-varchar cast is
engine-version dependent, and an engine upgrade must never invalidate hashes
that were sealed and possibly published. The byte layout is therefore a
public contract (`HASH_VERSION`), documented in `docs/methodology.md`.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import polars as pl

SOURCES = ("baseline", "challenger")
HASH_VERSION = 1
GENESIS = "0" * 64

_HASH_HEADER = "wfa-prediction-log-v1\n"

DDL = """
create schema if not exists live;

create table if not exists live.prediction_log (
    station varchar not null,
    run_date date not null,
    runtime_utc timestamp not null,
    lead_day integer not null,
    variable varchar not null,
    target_date date not null,
    source varchar not null,
    forecast_f double not null,
    raw_forecast_f double not null,
    window_end_utc timestamp not null,
    issued_at timestamp not null,
    retrained_on date,
    trained_through timestamp,
    fallback boolean not null,
    params varchar,
    primary key (station, run_date, lead_day, variable, source)
);

create table if not exists live.partition_seals (
    run_date date primary key,
    n_rows integer not null,
    n_late_rejected integer not null,
    partition_sha256 varchar not null,
    prev_chain_sha256 varchar not null,
    chain_sha256 varchar not null,
    hash_version integer not null,
    issued_at timestamp not null,
    sealed_at timestamp not null
);
"""

# Columns a caller supplies. `issued_at` is deliberately absent: the writer
# stamps it from its own `now`, so a caller cannot backdate a prediction.
_INPUT_COLUMNS = (
    "station",
    "run_date",
    "runtime_utc",
    "lead_day",
    "variable",
    "target_date",
    "source",
    "forecast_f",
    "raw_forecast_f",
    "window_end_utc",
    "retrained_on",
    "trained_through",
    "fallback",
    "params",
)

# The fixed column order of the canonical row encoding (HASH_VERSION 1).
_HASH_COLUMNS = (
    "station",
    "run_date",
    "runtime_utc",
    "lead_day",
    "variable",
    "target_date",
    "source",
    "forecast_f",
    "raw_forecast_f",
    "window_end_utc",
    "issued_at",
    "retrained_on",
    "trained_through",
    "fallback",
    "params",
)

_TARGET_DTYPES: dict[str, Any] = {
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

_NOT_NULL = (
    "station",
    "run_date",
    "runtime_utc",
    "lead_day",
    "variable",
    "target_date",
    "source",
    "forecast_f",
    "raw_forecast_f",
    "window_end_utc",
    "fallback",
)

_KEY = ("station", "lead_day", "variable", "source")


class PartitionSealedError(ValueError):
    """The `run_date` already has a seal or rows; it is never appended to again."""


class LogIntegrityError(RuntimeError):
    """The stored log no longer matches what was sealed."""


@dataclass(frozen=True)
class AppendResult:
    """What one `append_partition` call did, including what it refused."""

    run_date: date
    n_appended: int
    n_late_rejected: int
    late_keys: tuple[tuple[str, int, str, str], ...]
    partition_sha256: str | None
    chain_sha256: str | None
    sealed: bool


@dataclass(frozen=True)
class VerifyReport:
    """Result of `verify_immutability`; `problems` lists every mismatch found."""

    ok: bool
    partitions: int
    rows: int
    problems: tuple[str, ...]
    head_chain_sha256: str


def init_log(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the `live` schema and its two tables (idempotent).

    Called from `warehouse.init_db` so every connection path has the tables,
    and from `wfa log verify` so a database that predates the log verifies
    as an empty log rather than failing on a missing schema.
    """
    conn.execute(DDL)


def _canonical_value(value: object) -> object:
    """One cell as a JSON-ready value of the version-1 row encoding.

    `bool` is tested before `int` and `datetime` before `date` because each
    is a subclass of the other, and a wrong branch would silently change
    every sealed hash.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S.%f")
    if isinstance(value, date):
        return value.isoformat()
    msg = f"unsupported value type in prediction log row: {type(value).__name__}"
    raise TypeError(msg)


def _row_string(values: tuple[object, ...]) -> str:
    return json.dumps(
        [_canonical_value(v) for v in values],
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stored_rows(
    conn: duckdb.DuckDBPyConnection, run_date: date
) -> list[tuple[object, ...]]:
    columns = ", ".join(_HASH_COLUMNS)
    rows = conn.execute(
        f"select {columns} from live.prediction_log where run_date = ?", [run_date]
    ).fetchall()
    # Ordered in Python, on the key, so the contract does not lean on the
    # engine's collation.
    index = {name: i for i, name in enumerate(_HASH_COLUMNS)}
    key_positions = [index[name] for name in _KEY]
    return sorted(rows, key=lambda row: tuple(row[p] for p in key_positions))


def partition_hash(conn: duckdb.DuckDBPyConnection, run_date: date) -> tuple[int, str]:
    """Return (row count, sha256 hex) of one partition as currently stored.

    Both the seal and the verifier go through this one function so that
    "what was sealed" and "what is checked" cannot be computed two ways.
    """
    rows = _stored_rows(conn, run_date)
    body = _HASH_HEADER + "\n".join(_row_string(tuple(row)) for row in rows)
    return len(rows), hashlib.sha256(body.encode("utf-8")).hexdigest()


def _chain_hash(prev_chain_sha256: str, run_date: date, partition_sha256: str) -> str:
    text = f"{prev_chain_sha256}:{run_date.isoformat()}:{partition_sha256}"
    return hashlib.sha256(text.encode()).hexdigest()


def _naive_utc_now(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        msg = f"`now` must be a timezone-aware UTC datetime, got {now!r}"
        raise ValueError(msg)
    return now.astimezone(UTC).replace(tzinfo=None)


def _normalize(rows: pl.DataFrame) -> pl.DataFrame:
    """Select the contract columns, cast them, and make datetimes naive UTC."""
    missing = [name for name in _INPUT_COLUMNS if name not in rows.columns]
    if missing:
        msg = f"rows is missing required columns: {', '.join(missing)}"
        raise ValueError(msg)
    frame = rows.select(_INPUT_COLUMNS)
    expressions = []
    for name in _INPUT_COLUMNS:
        column = pl.col(name)
        actual = frame.schema[name]
        if isinstance(actual, pl.Datetime) and actual.time_zone is not None:
            column = column.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        expressions.append(column.cast(_TARGET_DTYPES[name]).alias(name))
    try:
        return frame.select(expressions)
    except pl.exceptions.PolarsError as error:
        msg = f"rows has a column that cannot be cast to its contract type: {error}"
        raise ValueError(msg) from error


def _validate(frame: pl.DataFrame, run_date: date, naive_now: datetime) -> None:
    for name in _NOT_NULL:
        if frame[name].null_count() > 0:
            msg = f"column {name!r} must not contain nulls"
            raise ValueError(msg)
    if frame.height == 0:
        return
    wrong = frame.filter(pl.col("run_date") != run_date)
    if wrong.height > 0:
        msg = f"{wrong.height} row(s) have a run_date other than {run_date}"
        raise ValueError(msg)
    unknown = sorted(set(frame["source"].to_list()) - set(SOURCES))
    if unknown:
        msg = f"unknown source(s) {unknown}; expected one of {list(SOURCES)}"
        raise ValueError(msg)
    if frame.select(_KEY).is_duplicated().any():
        msg = "duplicate (station, lead_day, variable, source) keys in rows"
        raise ValueError(msg)
    for name in ("forecast_f", "raw_forecast_f"):
        if not frame[name].is_finite().all():
            msg = f"column {name!r} must be finite (no NaN or infinity)"
            raise ValueError(msg)
    future = frame.filter(pl.col("runtime_utc") > naive_now)
    if future.height > 0:
        msg = f"{future.height} row(s) have runtime_utc after now ({naive_now})"
        raise ValueError(msg)


def append_partition(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    rows: pl.DataFrame,
    *,
    now: datetime,
) -> AppendResult:
    """Write one run date's predictions once, then seal the partition.

    Why once: the log's whole value is that a row cannot be revised after
    its outcome is known, so a sealed (or merely populated) partition
    refuses every later append, identical rows included.

    Why `now` and a strict late rule: `issued_at` is stamped here from the
    caller's clock, never read from `rows`, and a row is issuable only while
    `now < window_end_utc`. A row whose window has already closed could have
    been scored against its own answer; it is counted and reported, never
    stored.

    Why refuse out-of-order dates: a seal's `prev_chain_sha256` is the
    chain of the previous run date. Inserting an earlier date after a later
    seal would put the new seal in the middle and silently invalidate every
    later link, so append-only means date order.

    The rows and the seal are written in one transaction so a crash cannot
    leave a populated-but-unsealed partition, which `verify_immutability`
    would (rightly) report as tampering.
    """
    naive_now = _naive_utc_now(now)
    frame = _normalize(rows)
    _validate(frame, run_date, naive_now)

    sealed_row = conn.execute(
        "select 1 from live.partition_seals where run_date = ?", [run_date]
    ).fetchone()
    logged_row = conn.execute(
        "select 1 from live.prediction_log where run_date = ? limit 1", [run_date]
    ).fetchone()
    if sealed_row is not None or logged_row is not None:
        msg = f"partition {run_date} is already sealed or populated; never re-appended"
        raise PartitionSealedError(msg)
    later = conn.execute(
        "select min(run_date) from live.partition_seals where run_date > ?",
        [run_date],
    ).fetchone()
    if later is not None and later[0] is not None:
        msg = (
            f"cannot append {run_date}: {later[0]} is already sealed and the "
            "log is append-only in run_date order"
        )
        raise ValueError(msg)

    issuable = frame.filter(pl.col("window_end_utc") > naive_now)
    late = frame.filter(pl.col("window_end_utc") <= naive_now)
    late_keys = tuple(
        sorted(
            (str(s), int(lead), str(v), str(src))
            for s, lead, v, src in late.select(_KEY).iter_rows()
        )
    )
    if issuable.height == 0:
        return AppendResult(
            run_date=run_date,
            n_appended=0,
            n_late_rejected=late.height,
            late_keys=late_keys,
            partition_sha256=None,
            chain_sha256=None,
            sealed=False,
        )

    to_write = issuable.with_columns(
        pl.lit(naive_now, dtype=pl.Datetime("us")).alias("issued_at")
    ).select(_HASH_COLUMNS)
    column_list = ", ".join(_HASH_COLUMNS)

    conn.execute("begin transaction")
    try:
        conn.register("_wfa_issuable_rows", to_write)
        try:
            conn.execute(
                f"insert into live.prediction_log ({column_list}) "
                f"select {column_list} from _wfa_issuable_rows"
            )
        finally:
            conn.unregister("_wfa_issuable_rows")
        n_rows, partition_sha256 = partition_hash(conn, run_date)
        previous = conn.execute(
            "select chain_sha256 from live.partition_seals where run_date < ? "
            "order by run_date desc limit 1",
            [run_date],
        ).fetchone()
        prev_chain = previous[0] if previous is not None else GENESIS
        chain = _chain_hash(prev_chain, run_date, partition_sha256)
        conn.execute(
            "insert into live.partition_seals (run_date, n_rows, n_late_rejected, "
            "partition_sha256, prev_chain_sha256, chain_sha256, hash_version, "
            "issued_at, sealed_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_date,
                n_rows,
                late.height,
                partition_sha256,
                prev_chain,
                chain,
                HASH_VERSION,
                naive_now,
                naive_now,
            ],
        )
        conn.execute("commit")
    except Exception:
        conn.execute("rollback")
        raise

    return AppendResult(
        run_date=run_date,
        n_appended=n_rows,
        n_late_rejected=late.height,
        late_keys=late_keys,
        partition_sha256=partition_sha256,
        chain_sha256=chain,
        sealed=True,
    )


def verify_immutability(conn: duckdb.DuckDBPyConnection) -> VerifyReport:
    """Recompute every seal from the stored rows and report all mismatches.

    All problems are collected rather than stopping at the first, because
    the pattern of failures (one partition vs. a broken chain) is what tells
    an operator whether a row was edited or a seal was.

    Limit: this proves the log is self-consistent, i.e. nothing changed
    after sealing. Removing the newest seal together with its rows leaves
    a shorter log that still verifies; only comparing `head_chain_sha256`
    against a value published elsewhere catches that.
    """
    problems: list[str] = []
    seals = conn.execute(
        "select run_date, n_rows, partition_sha256, prev_chain_sha256, "
        "chain_sha256, hash_version, issued_at "
        "from live.partition_seals order by run_date"
    ).fetchall()

    expected_prev = GENESIS
    for run_date, n_rows, part_sha, prev_sha, chain_sha, version, _ in seals:
        if version != HASH_VERSION:
            problems.append(
                f"{run_date}: unsupported hash_version {version} "
                f"(this verifier knows {HASH_VERSION}); partition hash not checked"
            )
        else:
            stored_n, recomputed = partition_hash(conn, run_date)
            if stored_n != n_rows:
                problems.append(
                    f"{run_date}: seal records {n_rows} rows but {stored_n} are stored"
                )
            if recomputed != part_sha:
                problems.append(
                    f"{run_date}: partition hash mismatch "
                    f"(seal {part_sha[:12]}, stored rows {recomputed[:12]})"
                )
        if prev_sha != expected_prev:
            problems.append(
                f"{run_date}: chain broken, prev_chain_sha256 "
                f"{prev_sha[:12]} != previous seal's chain {expected_prev[:12]}"
            )
        recomputed_chain = _chain_hash(prev_sha, run_date, part_sha)
        if recomputed_chain != chain_sha:
            problems.append(
                f"{run_date}: chain_sha256 {chain_sha[:12]} does not match "
                f"its inputs (recomputed {recomputed_chain[:12]})"
            )
        expected_prev = chain_sha

    unsealed = conn.execute(
        "select run_date, count(*) from live.prediction_log "
        "where run_date not in (select run_date from live.partition_seals) "
        "group by run_date order by run_date"
    ).fetchall()
    for run_date, n in unsealed:
        problems.append(f"{run_date}: {n} unsealed row(s) in prediction_log")

    issue_rows = conn.execute(
        "select l.run_date, "
        "count(*) filter (where s.issued_at is not null "
        "and l.issued_at <> s.issued_at), "
        "count(*) filter (where l.issued_at >= l.window_end_utc) "
        "from live.prediction_log l "
        "left join live.partition_seals s on l.run_date = s.run_date "
        "group by l.run_date order by l.run_date"
    ).fetchall()
    for run_date, n_mismatch, n_not_before in issue_rows:
        if n_mismatch:
            problems.append(
                f"{run_date}: {n_mismatch} row(s) have issued_at different "
                "from the seal's issued_at"
            )
        if n_not_before:
            problems.append(
                f"{run_date}: {n_not_before} row(s) have issued_at at or after "
                "window_end_utc"
            )

    total = conn.execute("select count(*) from live.prediction_log").fetchone()
    n_total = int(total[0]) if total is not None else 0
    head = seals[-1][4] if seals else GENESIS
    return VerifyReport(
        ok=not problems,
        partitions=len(seals),
        rows=n_total,
        problems=tuple(problems),
        head_chain_sha256=head,
    )


def assert_immutable(conn: duckdb.DuckDBPyConnection) -> VerifyReport:
    """Return the report when the log verifies; raise listing every problem."""
    report = verify_immutability(conn)
    if not report.ok:
        listing = "\n".join(f"  - {p}" for p in report.problems)
        msg = f"prediction log failed verification:\n{listing}"
        raise LogIntegrityError(msg)
    return report
