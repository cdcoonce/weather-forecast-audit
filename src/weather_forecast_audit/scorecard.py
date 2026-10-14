"""Live scorecard: the append-only prediction log graded against observations.

Where `export` publishes the audit of raw NBM over the whole archive, this
module answers a different question: how are the predictions that were
*issued before their outcome was known* doing right now. Its rows come from
`live.prediction_log` only; backfill and walk-forward results are never mixed
in, because they were not issued prospectively and the log's late rule would
refuse them.

Three decisions worth keeping in one place:

* The window is `WINDOW_DAYS` consecutive `target_date`s ending at the latest
  `target_date` that has a scorable observation among logged keys. A date
  with no observation yet never shortens or shifts the window.
* `raw_nbm` is taken from the log's own `raw_forecast_f`, one row per logged
  key (deduplicated across `baseline`/`challenger`, which must agree), never
  from the warehouse's current `forecast_f`: the log records what was issued,
  and a later re-ingest of guidance must not move a published number.
* Scoring is `scoring.score` at its library defaults. Nothing is tuned here.
  A 30-day window holds only a few 14-day blocks, so `min_sample_flag` is
  true everywhere; that is the project's own floor speaking and it is
  exported, not hidden.

Values here are unrounded; `export` owns rounding and the file layout.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import polars as pl

from weather_forecast_audit.prediction_log import (
    GENESIS,
    VerifyReport,
    assert_immutable,
)
from weather_forecast_audit.scoring import (
    BLOCK_DAYS,
    CI_LEVEL,
    MIN_SAMPLE_BLOCKS,
    N_BOOT,
    score,
)

WINDOW_DAYS = 30
WINDOW_BASIS = "target_date"
RAW_SOURCE = "raw_nbm"
SOURCE_ORDER = (RAW_SOURCE, "baseline", "challenger")

_VARIABLE_ORDER = {"max": 0, "min": 1}
_KEY = ("station", "run_date", "lead_day", "variable")

_LOGGED_QUERY = """
    select l.station, l.run_date, l.lead_day, l.variable, l.source,
           l.target_date, l.forecast_f, l.raw_forecast_f, p.observed_f,
           coalesce(p.scorable, false)
               and coalesce(isfinite(p.observed_f), false) as scorable
    from live.prediction_log as l
    left join int_raw_verification_pairs as p
        on p.station = l.station
        and p.run_date = l.run_date
        and p.lead_day = l.lead_day
        and p.variable = l.variable
"""

_LATEST_GRADED_QUERY = """
    select max(l.target_date)
    from live.prediction_log as l
    join int_raw_verification_pairs as p
        on p.station = l.station
        and p.run_date = l.run_date
        and p.lead_day = l.lead_day
        and p.variable = l.variable
    where p.scorable and coalesce(isfinite(p.observed_f), false)
"""

_FRAME_SCHEMA = {
    "station": pl.String,
    "run_date": pl.Date,
    "lead_day": pl.Int32,
    "variable": pl.String,
    "source": pl.String,
    "target_date": pl.Date,
    "scorable": pl.Boolean,
    "error_f": pl.Float64,
}


@dataclass(frozen=True)
class Scorecard:
    """The live scorecard, unrounded.

    `window` is `(start, end)` inclusive, or `None` when no logged key has a
    scorable observation yet (then `national` and `stations` are empty).
    """

    window: tuple[date, date] | None
    log: dict[str, object]
    national: list[dict[str, object]]
    stations: dict[str, list[dict[str, object]]]


def intervals_note() -> str:
    """The standing caveat that travels with every exported scorecard."""
    return (
        f"The {WINDOW_DAYS}-day window spans only a few {BLOCK_DAYS}-day blocks, "
        f"fewer than min_sample_blocks ({MIN_SAMPLE_BLOCKS}), so every interval "
        "is approximate and the window is too short for a significance call."
    )


def intervals_block() -> dict[str, object]:
    return {
        "method": "date-block bootstrap",
        "block_days": BLOCK_DAYS,
        "n_boot": N_BOOT,
        "level": CI_LEVEL,
        "min_sample_blocks": MIN_SAMPLE_BLOCKS,
        "note": intervals_note(),
    }


def _log_tables_exist(conn: duckdb.DuckDBPyConnection) -> bool:
    row = conn.execute(
        "select count(*) from information_schema.tables "
        "where table_schema = 'live' "
        "and table_name in ('prediction_log', 'partition_seals')"
    ).fetchone()
    return row is not None and int(row[0]) == 2


def verified_log_report(conn: duckdb.DuckDBPyConnection) -> VerifyReport:
    """Verify the log and return its report, or raise `LogIntegrityError`.

    A database that predates the `live` schema has an empty log, which
    verifies; it is not a failure to find a table.
    """
    if not _log_tables_exist(conn):
        return VerifyReport(
            ok=True, partitions=0, rows=0, problems=(), head_chain_sha256=GENESIS
        )
    return assert_immutable(conn)


def _log_block(
    conn: duckdb.DuckDBPyConnection, report: VerifyReport
) -> dict[str, object]:
    first_run: date | None = None
    last_run: date | None = None
    late_total = 0
    if report.partitions > 0:
        row = conn.execute(
            "select min(run_date), max(run_date), "
            "coalesce(sum(n_late_rejected), 0) from live.partition_seals"
        ).fetchone()
        if row is not None:
            first_run, last_run, late_total = row[0], row[1], int(row[2])
    return {
        "partitions": report.partitions,
        "rows": report.rows,
        "first_run_date": first_run.isoformat() if first_run else None,
        "last_run_date": last_run.isoformat() if last_run else None,
        "late_rejected_total": late_total,
        "head_chain_sha256": report.head_chain_sha256,
        "verified": report.ok,
    }


def _graded_frame(logged: pl.DataFrame) -> pl.DataFrame:
    """One row per (logged key, source) with `error_f` where scorable."""
    issued = logged.select(
        *_KEY, "source", "target_date", "forecast_f", "observed_f", "scorable"
    )
    # Both sources are issued from the same run row, so they must agree on
    # the raw forecast; a disagreement is a corrupt log, not a tie to break.
    disagreeing = (
        logged.group_by(list(_KEY))
        .agg(pl.col("raw_forecast_f").n_unique().alias("_n"))
        .filter(pl.col("_n") > 1)
    )
    if disagreeing.height > 0:
        msg = (
            f"{disagreeing.height} logged key(s) carry different raw_forecast_f "
            "across sources; refusing to pick one"
        )
        raise ValueError(msg)
    raw = (
        logged.sort("source")
        .unique(subset=list(_KEY), keep="first", maintain_order=True)
        .select(
            *_KEY,
            pl.lit(RAW_SOURCE).alias("source"),
            "target_date",
            pl.col("raw_forecast_f").alias("forecast_f"),
            "observed_f",
            "scorable",
        )
    )
    return (
        pl.concat([issued, raw])
        .with_columns(
            error_f=pl.when(pl.col("scorable"))
            .then(pl.col("forecast_f") - pl.col("observed_f"))
            .otherwise(None)
        )
        .select(list(_FRAME_SCHEMA))
        .cast(_FRAME_SCHEMA)  # type: ignore[arg-type]
    )


def _stat_records(scored: pl.DataFrame) -> list[tuple[str | None, dict[str, object]]]:
    records: list[tuple[str | None, dict[str, object]]] = []
    for row in scored.iter_rows(named=True):
        stat: dict[str, object] = {
            "variable": row["variable"],
            "lead_day": int(row["lead_day"]),
            "source": row["source"],
            "n": int(row["n"]),
            "n_dates": int(row["n_dates"]),
            "n_blocks": int(row["n_blocks"]),
            "mae_f": row["mae"],
            "mae_lo_f": row["mae_lo"],
            "mae_hi_f": row["mae_hi"],
            "bias_f": row["bias"],
            "bias_lo_f": row["bias_lo"],
            "bias_hi_f": row["bias_hi"],
            "skill": row["skill"],
            "skill_lo": row["skill_lo"],
            "skill_hi": row["skill_hi"],
            "min_sample_flag": bool(row["min_sample_flag"]),
            "no_detectable_bias": bool(row["no_detectable_bias"]),
        }
        records.append((row.get("station"), stat))
    return records


def _sort_key(stat: dict[str, object]) -> tuple[int, int, int]:
    return (
        _VARIABLE_ORDER[str(stat["variable"])],
        int(stat["lead_day"]),  # type: ignore[call-overload]
        SOURCE_ORDER.index(str(stat["source"])),
    )


def build_scorecard(
    conn: duckdb.DuckDBPyConnection, report: VerifyReport | None = None
) -> Scorecard:
    """Compute the live scorecard from the log, verifying it first.

    `report` is the caller's already-verified `VerifyReport`; when omitted
    the log is verified here, so a scorecard is never computed over a log
    that fails its own check (`LogIntegrityError` propagates).
    """
    if report is None:
        report = verified_log_report(conn)
    log = _log_block(conn, report)
    empty = Scorecard(window=None, log=log, national=[], stations={})
    if report.rows == 0:
        return empty

    # The window is found in SQL and only its rows are loaded: the log grows
    # by about two million rows a year and this runs on a shared host.
    latest = conn.execute(_LATEST_GRADED_QUERY).fetchone()
    if latest is None or latest[0] is None:
        return empty
    end: date = latest[0]
    start = end - timedelta(days=WINDOW_DAYS - 1)

    in_window = conn.execute(
        _LOGGED_QUERY + " where l.target_date between ? and ?", [start, end]
    ).pl()
    frame = _graded_frame(in_window)

    national = [
        stat
        for _, stat in _stat_records(
            score(frame, by=("variable", "lead_day"), reference=RAW_SOURCE)
        )
    ]
    national.sort(key=_sort_key)

    stations: dict[str, list[dict[str, object]]] = {}
    for station, stat in _stat_records(
        score(frame, by=("station", "variable", "lead_day"), reference=RAW_SOURCE)
    ):
        stations.setdefault(str(station), []).append(stat)
    for stats in stations.values():
        stats.sort(key=_sort_key)

    return Scorecard(
        window=(start, end), log=log, national=national, stations=stations
    )
