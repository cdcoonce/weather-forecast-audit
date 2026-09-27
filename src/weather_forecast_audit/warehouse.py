"""Raw warehouse schema and idempotent loaders.

This module owns every `create table` for the `raw` schema; dbt only reads
these tables (`models/staging/_sources.yml`), never creates them. Every
timestamp column is a portable `timestamp` (naive) holding UTC by
convention, documented here and on each dbt source, not by the column type
(Snowflake and DuckDB both support `timestamp`; a naive `timestamptz` is not
portable).

Loaders are idempotent: each one deletes the station's existing rows in the
loaded date range inside a transaction, then inserts the new rows, so
re-running the same ingest does not duplicate rows.
"""

from dataclasses import dataclass
from datetime import date, datetime

import duckdb
import polars as pl

from weather_forecast_audit.gaps import GapRecord
from weather_forecast_audit.iem.guidance import GuidanceRow
from weather_forecast_audit.iem.observations import CliDaily, HourlyObservation
from weather_forecast_audit.resolver import ExtremeSource, Variable

DDL = """
create schema if not exists raw;

create table if not exists raw.nbs_guidance (
    station varchar not null,
    runtime_utc timestamp not null,
    ftime_utc timestamp not null,
    cycle_hour integer not null,
    txn double,
    xnd double,
    tmp double
);

create table if not exists raw.asos_hourly (
    station varchar not null,
    valid_utc timestamp not null,
    tmpf double,
    max_6h_f double,
    min_6h_f double
);

create table if not exists raw.cli_daily (
    station varchar not null,
    local_date date not null,
    high_f integer,
    low_f integer
);

create table if not exists raw.ingest_gaps (
    station varchar not null,
    source varchar not null,
    expected varchar not null,
    reason varchar not null,
    first_seen timestamp
);

create table if not exists raw.model_predictions (
    station varchar not null,
    run_date date not null,
    runtime_utc timestamp not null,
    lead_day integer not null,
    variable varchar not null,
    target_date date not null,
    source varchar not null,
    forecast_f double not null,
    raw_forecast_f double not null,
    retrained_on date,
    trained_through timestamp,
    fallback boolean not null,
    params varchar,
    generated_at timestamp not null
);

create table if not exists raw.resolved_windows (
    station varchar not null,
    runtime_utc timestamp not null,
    ftime_utc timestamp not null,
    variable varchar not null,
    target_date date not null,
    lead_day integer not null,
    window_start_utc timestamp not null,
    window_end_utc timestamp not null,
    observed_f double,
    n_obs integer not null,
    hours_covered integer not null,
    hours_expected integer not null,
    scorable boolean not null,
    extreme_source varchar,
    periods_found integer,
    hourly_observed_f double
);
"""


@dataclass(frozen=True)
class ResolvedWindowRow:
    station: str
    runtime_utc: datetime
    ftime_utc: datetime
    variable: Variable
    target_date: date
    lead_day: int
    window_start_utc: datetime
    window_end_utc: datetime
    observed_f: float | None
    n_obs: int
    hours_covered: int
    hours_expected: int
    scorable: bool
    extreme_source: ExtremeSource
    periods_found: int
    hourly_observed_f: float | None


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create every `raw` table, then upgrade any that predate a schema change.

    `create table if not exists` alone leaves an already-existing table on
    its old schema; the DuckDB file is a long-lived cache, so the explicit
    `alter table ... add column if not exists` calls below let a database
    created before the METAR 6-hour columns existed pick them up in place,
    without dropping its rows.
    """
    conn.execute(DDL)
    conn.execute("alter table raw.asos_hourly add column if not exists max_6h_f double")
    conn.execute("alter table raw.asos_hourly add column if not exists min_6h_f double")
    conn.execute(
        "alter table raw.resolved_windows add column if not exists "
        "extreme_source varchar"
    )
    conn.execute(
        "alter table raw.resolved_windows add column if not exists "
        "periods_found integer"
    )
    conn.execute(
        "alter table raw.resolved_windows add column if not exists "
        "hourly_observed_f double"
    )
    conn.execute(
        "alter table raw.ingest_gaps add column if not exists first_seen timestamp"
    )


def _naive(value: datetime) -> datetime:
    """Strip tzinfo from an already-UTC datetime for portable storage."""
    return value.replace(tzinfo=None)


def _replace(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    station: str,
    date_column_sql: str,
    start: date,
    end: date,
    records: list[dict[str, object]],
) -> None:
    conn.execute("begin transaction")
    try:
        conn.execute(
            f"delete from {table} where station = ? "  # noqa: S608 (table is a fixed constant, not user input)
            f"and cast({date_column_sql} as date) between ? and ?",
            [station, start, end],
        )
        if records:
            frame = pl.DataFrame(records)  # noqa: F841 (read by name via duckdb's scan)
            column_list = ", ".join(columns)
            conn.execute(
                f"insert into {table} ({column_list}) select {column_list} from frame"
            )
        conn.execute("commit")
    except Exception:
        conn.execute("rollback")
        raise


def load_guidance(
    conn: duckdb.DuckDBPyConnection,
    station: str,
    start: date,
    end: date,
    rows: list[GuidanceRow],
) -> None:
    columns = ["station", "runtime_utc", "ftime_utc", "cycle_hour", "txn", "xnd", "tmp"]
    records = [
        {
            "station": row.station,
            "runtime_utc": _naive(row.runtime),
            "ftime_utc": _naive(row.ftime),
            "cycle_hour": row.cycle_hour,
            "txn": row.txn,
            "xnd": row.xnd,
            "tmp": row.tmp,
        }
        for row in rows
    ]
    _replace(
        conn, "raw.nbs_guidance", columns, station, "runtime_utc", start, end, records
    )


def load_hourly(
    conn: duckdb.DuckDBPyConnection,
    station: str,
    start: date,
    end: date,
    rows: list[HourlyObservation],
) -> None:
    columns = ["station", "valid_utc", "tmpf", "max_6h_f", "min_6h_f"]
    records = [
        {
            "station": row.station,
            "valid_utc": _naive(row.valid_utc),
            "tmpf": row.tmpf,
            "max_6h_f": row.max_6h_f,
            "min_6h_f": row.min_6h_f,
        }
        for row in rows
    ]
    _replace(
        conn, "raw.asos_hourly", columns, station, "valid_utc", start, end, records
    )


def load_cli(
    conn: duckdb.DuckDBPyConnection,
    station: str,
    start: date,
    end: date,
    rows: list[CliDaily],
) -> None:
    columns = ["station", "local_date", "high_f", "low_f"]
    records = [
        {
            "station": row.station,
            "local_date": row.local_date,
            "high_f": row.high_f,
            "low_f": row.low_f,
        }
        for row in rows
    ]
    _replace(conn, "raw.cli_daily", columns, station, "local_date", start, end, records)


def load_gaps(
    conn: duckdb.DuckDBPyConnection,
    station: str,
    source: str,
    start: date,
    end: date,
    gaps: list[GapRecord],
    *,
    now: datetime,
) -> None:
    """Replace `station`/`source`'s gaps in `[start, end]`, preserving `first_seen`.

    `now` is a naive UTC datetime (the caller's clock). A gap already on file
    for the same `(station, source, expected, reason)` keeps its original
    `first_seen`; a gap not seen before gets `first_seen = now`; a gap that
    was on file but is absent from `gaps` is deleted (build spec D3).

    Every gap must belong to the slice being replaced. One outside it would
    be inserted but never deleted by a re-run of this slice, so partitions
    would silently accumulate duplicate gaps.
    """
    for gap in gaps:
        gap_date = date.fromisoformat(gap.expected[:10])
        if gap.station != station or gap.source != source or not (
            start <= gap_date <= end
        ):
            msg = (
                f"gap {gap} is outside the replaced slice "
                f"({station}, {source}, {start}..{end})"
            )
            raise ValueError(msg)
    conn.execute("begin transaction")
    try:
        existing_first_seen = {
            (row[0], row[1], row[2], row[3]): row[4]
            for row in conn.execute(
                "select station, source, expected, reason, first_seen "
                "from raw.ingest_gaps where station = ? and source = ? "
                "and cast(substr(expected, 1, 10) as date) between ? and ?",
                [station, source, start, end],
            ).fetchall()
        }
        conn.execute(
            "delete from raw.ingest_gaps where station = ? and source = ? "
            "and cast(substr(expected, 1, 10) as date) between ? and ?",
            [station, source, start, end],
        )
        if gaps:
            records = [
                {
                    "station": gap.station,
                    "source": gap.source,
                    "expected": gap.expected,
                    "reason": gap.reason,
                    "first_seen": existing_first_seen.get(
                        (gap.station, gap.source, gap.expected, gap.reason), now
                    ),
                }
                for gap in gaps
            ]
            frame = pl.DataFrame(records)  # noqa: F841
            conn.execute(
                "insert into raw.ingest_gaps "
                "(station, source, expected, reason, first_seen) "
                "select station, source, expected, reason, first_seen from frame"
            )
        conn.execute("commit")
    except Exception:
        conn.execute("rollback")
        raise


def load_resolved_windows(
    conn: duckdb.DuckDBPyConnection,
    station: str,
    start: date,
    end: date,
    rows: list[ResolvedWindowRow],
) -> None:
    columns = [
        "station",
        "runtime_utc",
        "ftime_utc",
        "variable",
        "target_date",
        "lead_day",
        "window_start_utc",
        "window_end_utc",
        "observed_f",
        "n_obs",
        "hours_covered",
        "hours_expected",
        "scorable",
        "extreme_source",
        "periods_found",
        "hourly_observed_f",
    ]
    records = [
        {
            "station": row.station,
            "runtime_utc": _naive(row.runtime_utc),
            "ftime_utc": _naive(row.ftime_utc),
            "variable": row.variable,
            "target_date": row.target_date,
            "lead_day": row.lead_day,
            "window_start_utc": _naive(row.window_start_utc),
            "window_end_utc": _naive(row.window_end_utc),
            "observed_f": row.observed_f,
            "n_obs": row.n_obs,
            "hours_covered": row.hours_covered,
            "hours_expected": row.hours_expected,
            "scorable": row.scorable,
            "extreme_source": row.extreme_source,
            "periods_found": row.periods_found,
            "hourly_observed_f": row.hourly_observed_f,
        }
        for row in rows
    ]
    _replace(
        conn,
        "raw.resolved_windows",
        columns,
        station,
        "runtime_utc",
        start,
        end,
        records,
    )


_MODEL_PREDICTION_COLUMNS = [
    "station",
    "run_date",
    "runtime_utc",
    "lead_day",
    "variable",
    "target_date",
    "source",
    "forecast_f",
    "raw_forecast_f",
    "retrained_on",
    "trained_through",
    "fallback",
    "params",
    "generated_at",
]


def load_predictions(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    start: date,
    end: date,
    rows_frame: pl.DataFrame,
    *,
    now: datetime,
) -> None:
    """Idempotently replace `source`'s prediction rows for `run_date` in `[start, end]`.

    One transaction: delete this `source`'s existing `raw.model_predictions`
    rows in the `run_date` range, then insert `rows_frame` (every other
    source's rows, and rows for other date ranges, are left untouched).
    `rows_frame` must carry every column but `generated_at`, which is
    stamped here from `now` (a naive-UTC datetime, the caller's clock) so a
    re-load is reproducible rather than depending on wall-clock time read
    inside this function.
    """
    frame = rows_frame.with_columns(pl.lit(now).alias("generated_at"))  # noqa: F841
    conn.execute("begin transaction")
    try:
        conn.execute(
            "delete from raw.model_predictions where source = ? "
            "and run_date between ? and ?",
            [source, start, end],
        )
        column_list = ", ".join(_MODEL_PREDICTION_COLUMNS)
        conn.execute(
            f"insert into raw.model_predictions ({column_list}) "
            f"select {column_list} from frame"
        )
        conn.execute("commit")
    except Exception:
        conn.execute("rollback")
        raise
