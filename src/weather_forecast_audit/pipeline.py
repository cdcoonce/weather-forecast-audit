"""Ingest one station's raw data and resolve verification windows.

`ingest_station` is the whole vertical slice: fetch NBS guidance, ASOS
hourly obs, and CLI daily reports for a station; load every row and gap
into DuckDB; then `resolve_station` reads the loaded raw guidance and obs
back out, runs the pure resolver, and replaces `raw.resolved_windows`.

The ASOS/CLI window is padded relative to the guidance run-date range so
every lead-0..3 verification window (up to 30 hours past the run) has its
observations on hand: a run on `end` can carry a lead-3 target date of
`end + 3`, whose max window runs through `end + 4` 06Z.
"""

from dataclasses import dataclass
from datetime import UTC, date, timedelta

import duckdb

from weather_forecast_audit import warehouse
from weather_forecast_audit.iem.guidance import fetch_guidance
from weather_forecast_audit.iem.http import Fetcher
from weather_forecast_audit.iem.observations import fetch_cli, fetch_hourly
from weather_forecast_audit.regimes import CycleRegime
from weather_forecast_audit.registry import Station
from weather_forecast_audit.resolver import (
    classify_txn,
    lead_day,
    resolve_observed,
    resolve_window,
)
from weather_forecast_audit.warehouse import ResolvedWindowRow

# Covers every lead-0..3 window: a lead-3 target's max window ends at
# target+1 06Z, and target can be as late as run_date+3.
OBS_LOOKBACK_DAYS = 1
OBS_LOOKAHEAD_DAYS = 4


@dataclass(frozen=True)
class IngestSummary:
    station: str
    guidance_rows: int
    asos_rows: int
    cli_rows: int
    gaps_by_reason: dict[str, int]
    resolved_rows: int


def _count_by_reason(*gap_lists: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gaps in gap_lists:
        for gap in gaps:
            counts[gap.reason] = counts.get(gap.reason, 0) + 1
    return counts


def ingest_station(
    conn: duckdb.DuckDBPyConnection,
    station: Station,
    start: date,
    end: date,
    fetcher: Fetcher,
    regimes: list[CycleRegime],
) -> IngestSummary:
    """Fetch, load, and resolve one station's data for run dates [start, end]."""
    obs_start = start - timedelta(days=OBS_LOOKBACK_DAYS)
    obs_end = end + timedelta(days=OBS_LOOKAHEAD_DAYS)

    guidance_result = fetch_guidance(
        station.nbm_station_id, start, end, fetcher, regimes
    )
    hourly_result = fetch_hourly(station, obs_start, obs_end, fetcher)
    cli_result = fetch_cli(station, obs_start, obs_end, fetcher)

    warehouse.load_guidance(conn, station.icao, start, end, guidance_result.rows)
    warehouse.load_hourly(conn, station.icao, obs_start, obs_end, hourly_result.rows)
    warehouse.load_cli(conn, station.icao, obs_start, obs_end, cli_result.rows)

    warehouse.load_gaps(conn, station.icao, "nbs", start, end, guidance_result.gaps)
    warehouse.load_gaps(
        conn, station.icao, "asos", obs_start, obs_end, hourly_result.gaps
    )
    warehouse.load_gaps(conn, station.icao, "cli", obs_start, obs_end, cli_result.gaps)

    resolved_rows = resolve_station(conn, station.icao, start, end)

    return IngestSummary(
        station=station.icao,
        guidance_rows=len(guidance_result.rows),
        asos_rows=len(hourly_result.rows),
        cli_rows=len(cli_result.rows),
        gaps_by_reason=_count_by_reason(
            guidance_result.gaps, hourly_result.gaps, cli_result.gaps
        ),
        resolved_rows=resolved_rows,
    )


def resolve_station(
    conn: duckdb.DuckDBPyConnection, icao: str, start: date, end: date
) -> int:
    """Resolve verification windows for guidance runs in [start, end].

    Reads raw guidance (txn not null, any lead) and every raw hourly ob on
    file for the station, runs the pure resolver, and replaces
    `raw.resolved_windows` for that station/run-date range. Lead filtering
    to {1, 2, 3} happens downstream, in the dbt fact model.
    """
    guidance_rows = conn.execute(
        "select runtime_utc, ftime_utc from raw.nbs_guidance "
        "where station = ? and cast(runtime_utc as date) between ? and ? "
        "and txn is not null",
        [icao, start, end],
    ).fetchall()

    reports = [
        (valid.replace(tzinfo=UTC), tmpf, max_6h_f, min_6h_f)
        for valid, tmpf, max_6h_f, min_6h_f in conn.execute(
            "select valid_utc, tmpf, max_6h_f, min_6h_f from raw.asos_hourly "
            "where station = ?",
            [icao],
        ).fetchall()
    ]

    resolved: list[ResolvedWindowRow] = []
    for runtime_naive, ftime_naive in guidance_rows:
        runtime = runtime_naive.replace(tzinfo=UTC)
        ftime = ftime_naive.replace(tzinfo=UTC)
        variable, target_date = classify_txn(ftime)
        window = resolve_window(target_date, variable)
        observed = resolve_observed(window, variable, reports)
        resolved.append(
            ResolvedWindowRow(
                station=icao,
                runtime_utc=runtime,
                ftime_utc=ftime,
                variable=variable,
                target_date=target_date,
                lead_day=lead_day(runtime, target_date),
                window_start_utc=window.start_utc,
                window_end_utc=window.end_utc,
                observed_f=observed.value_f,
                n_obs=observed.n_obs,
                hours_covered=observed.hours_covered,
                hours_expected=observed.hours_expected,
                scorable=observed.scorable,
                extreme_source=observed.extreme_source,
                periods_found=observed.periods_found,
                hourly_observed_f=observed.hourly_value_f,
            )
        )

    warehouse.load_resolved_windows(conn, icao, start, end, resolved)
    return len(resolved)
