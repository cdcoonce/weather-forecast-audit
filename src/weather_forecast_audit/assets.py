"""Daily-partitioned raw ingest assets and their checks (build spec #10 D1, D2, D7).

Each asset owns exactly its own partition's rows in its own raw table: a
run's partition `[start, end]` is passed straight through to that source's
`fetch_*` function and to `warehouse.load_*`/`load_gaps`, which
delete-then-insert exactly that range, so "re-materializing a date replaces
that date's rows" is literally true and partitions never overwrite each
other (D1).

Table D1 lines up each asset's partition date with its `fetch_*`'s own range
semantics, so no boundary adaptation is needed:

- `raw/nbs_guidance`: partition date = run date, UTC. `fetch_guidance`'s
  `[start, end]` is already a run-date range, UTC.
- `raw/asos_hourly`: partition date = observation date, UTC.
  `fetch_hourly`'s `[start, end]` is already a UTC observation-date range.
- `raw/cli_daily`: partition date = the station's local climate date.
  `fetch_cli`'s `[start, end]` is already a local-date range.
- `raw/resolved_windows`: partition date = run date, UTC; no fetch of its
  own -- it reads the two raw tables above back out of DuckDB.

`tests/dagster/test_definitions.py`'s partition-ownership test proves this
directly: materializing 07-15 after 07-14 leaves 07-14's rows
byte-identical, and re-materializing 07-14 with fewer fixture rows only
changes 07-14's rows.

Stations are processed sequentially in ICAO order (`StationsResource.stations()`
already sorts) through one `Fetcher` from `IemResource`, matching D2's
"batching sized to the single run slot".
"""

from datetime import UTC, date, datetime, timedelta

import dagster as dg

from weather_forecast_audit import warehouse
from weather_forecast_audit.checks import (
    GAP_RATE_WARN,
    GUIDANCE_MAX_AGE_HOURS,
    OBS_MAX_AGE_HOURS,
    evaluate_freshness,
    evaluate_gap_rate,
)
from weather_forecast_audit.gaps import GapRecord
from weather_forecast_audit.iem._chunking import month_chunks
from weather_forecast_audit.iem.guidance import fetch_guidance
from weather_forecast_audit.iem.http import FetchStats
from weather_forecast_audit.iem.observations import fetch_cli, fetch_hourly
from weather_forecast_audit.pipeline import (
    OBS_LOOKAHEAD_DAYS,
    OBS_LOOKBACK_DAYS,
    resolve_station,
)
from weather_forecast_audit.regimes import load_archive_start, load_cycle_regimes
from weather_forecast_audit.resources import (
    ClockResource,
    IemResource,
    StationsResource,
    WarehouseResource,
)

ARCHIVE_START = load_archive_start()

daily_partitions = dg.DailyPartitionsDefinition(
    start_date=ARCHIVE_START.isoformat(), timezone="UTC"
)

# D2: a run receives up to a month (context.partition_time_window) and
# fetches each station once over the whole range; the clients already chunk
# requests by month (guidance/asos) or year (cli), so a month of backfill is
# about 3 x stations requests, not 31 x 3 x stations.
INGEST_BACKFILL_POLICY = dg.BackfillPolicy.multi_run(max_partitions_per_run=31)

RAW_NBS_GUIDANCE_KEY = dg.AssetKey(["raw", "nbs_guidance"])
RAW_ASOS_HOURLY_KEY = dg.AssetKey(["raw", "asos_hourly"])
RAW_CLI_DAILY_KEY = dg.AssetKey(["raw", "cli_daily"])
RAW_RESOLVED_WINDOWS_KEY = dg.AssetKey(["raw", "resolved_windows"])
RAW_INGEST_GAPS_KEY = dg.AssetKey(["raw", "ingest_gaps"])


def _http_stats_metadata(fetcher: object) -> dict[str, int | float | str]:
    """Measured request statistics, when the fetcher records them.

    `Fetcher` stays a Protocol with no stats member: only a fetcher whose
    `stats` is a `FetchStats` (the real `UrllibFetcher`) contributes keys, so
    fixture fetchers are untouched.
    """
    stats = getattr(fetcher, "stats", None)
    return stats.as_metadata() if isinstance(stats, FetchStats) else {}


def _slowest_attempt_log(fetcher: object) -> str:
    """A log-line suffix naming the slowest HTTP attempt ('' without stats)."""
    stats = getattr(fetcher, "stats", None)
    if not isinstance(stats, FetchStats):
        return ""
    return (
        f"; slowest attempt {stats.slowest_attempt_seconds:.1f}s "
        f"({stats.slowest_attempt_url or 'n/a'})"
    )


def _partition_date_range(context: dg.AssetExecutionContext) -> tuple[date, date]:
    """This run's partition window as an inclusive `[start, end]` date range."""
    window = context.partition_time_window
    return window.start.date(), (window.end - timedelta(days=1)).date()


def _gap_date(expected: str) -> date:
    """The gap's expected date, the same parse `warehouse.load_gaps` uses."""
    return date.fromisoformat(expected[:10])


def _stations_with_gap_in_window(
    gaps_by_station: dict[str, list[GapRecord]], start: date, end: date
) -> int:
    return sum(
        1
        for gaps in gaps_by_station.values()
        if any(start <= _gap_date(gap.expected) <= end for gap in gaps)
    )


def _gap_rate_check_result(
    check_name: str, gaps_by_station: dict[str, list[GapRecord]], start: date, end: date
) -> dg.AssetCheckResult:
    stations_total = len(gaps_by_station)
    stations_with_gap = _stations_with_gap_in_window(gaps_by_station, start, end)
    passed, rate = evaluate_gap_rate(stations_with_gap, stations_total, GAP_RATE_WARN)
    return dg.AssetCheckResult(
        check_name=check_name,
        passed=passed,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={
            "rate": rate,
            "stations_with_gap": stations_with_gap,
            "stations_total": stations_total,
            "threshold": GAP_RATE_WARN,
        },
    )


# -- raw/ingest_gaps: three writers, no materialization function -------------
#
# nbs_guidance, asos_hourly, and cli_daily all write into raw.ingest_gaps
# (warehouse.load_gaps), so no single asset "owns" it. This AssetSpec keeps
# the lineage real -- stg_ingest_gaps depends on something -- without
# claiming any one of the three writers materializes the whole table (D4).
ingest_gaps_spec = dg.AssetSpec(
    key=RAW_INGEST_GAPS_KEY,
    deps=[RAW_NBS_GUIDANCE_KEY, RAW_ASOS_HOURLY_KEY, RAW_CLI_DAILY_KEY],
    description=(
        "raw.ingest_gaps has three writers (nbs_guidance, asos_hourly, "
        "cli_daily via warehouse.load_gaps); this asset is not itself "
        "materializable, but gives stg_ingest_gaps a real upstream."
    ),
)


@dg.asset(
    key=RAW_NBS_GUIDANCE_KEY,
    partitions_def=daily_partitions,
    backfill_policy=INGEST_BACKFILL_POLICY,
    check_specs=[
        dg.AssetCheckSpec(name="guidance_gap_rate", asset=RAW_NBS_GUIDANCE_KEY)
    ],
    description=(
        "Archived NBS guidance, one partition per run date (UTC, "
        "cast(runtime_utc as date)). fetch_guidance's [start, end] is "
        "already a run-date range in UTC, so the partition window is passed "
        "straight through with no boundary adaptation."
    ),
)
def raw_nbs_guidance(
    context: dg.AssetExecutionContext,
    warehouse_resource: WarehouseResource,
    iem: IemResource,
    stations: StationsResource,
) -> dg.MaterializeResult:
    start, end = _partition_date_range(context)
    station_list = stations.stations()
    fetcher = iem.fetcher()
    regimes = load_cycle_regimes()
    now = datetime.now(UTC).replace(tzinfo=None)
    request_count = 0
    gaps_by_station: dict[str, list[GapRecord]] = {}

    with warehouse_resource.connect() as conn:
        for station in station_list:
            result = fetch_guidance(
                station.nbm_station_id, start, end, fetcher, regimes
            )
            request_count += len(month_chunks(start, end))
            warehouse.load_guidance(conn, station.icao, start, end, result.rows)
            warehouse.load_gaps(
                conn, station.icao, "nbs", start, end, result.gaps, now=now
            )
            gaps_by_station[station.icao] = result.gaps

    context.log.info(
        "raw/nbs_guidance %s..%s: %d stations, %d requests%s",
        start,
        end,
        len(station_list),
        request_count,
        _slowest_attempt_log(fetcher),
    )
    return dg.MaterializeResult(
        metadata={
            "station_count": len(station_list),
            "request_count": request_count,
            **_http_stats_metadata(fetcher),
        },
        check_results=[
            _gap_rate_check_result("guidance_gap_rate", gaps_by_station, start, end)
        ],
    )


@dg.asset(
    key=RAW_ASOS_HOURLY_KEY,
    partitions_def=daily_partitions,
    backfill_policy=INGEST_BACKFILL_POLICY,
    check_specs=[dg.AssetCheckSpec(name="asos_gap_rate", asset=RAW_ASOS_HOURLY_KEY)],
    description=(
        "Hourly ASOS observations, one partition per observation date (UTC, "
        "cast(valid_utc as date)). fetch_hourly's [start, end] is already a "
        "UTC observation-date range (its request URL uses UTC day "
        "boundaries and its gaps key off valid.date()), so the partition "
        "window is passed straight through with no boundary adaptation."
    ),
)
def raw_asos_hourly(
    context: dg.AssetExecutionContext,
    warehouse_resource: WarehouseResource,
    iem: IemResource,
    stations: StationsResource,
) -> dg.MaterializeResult:
    start, end = _partition_date_range(context)
    station_list = stations.stations()
    fetcher = iem.fetcher()
    now = datetime.now(UTC).replace(tzinfo=None)
    request_count = 0
    gaps_by_station: dict[str, list[GapRecord]] = {}

    with warehouse_resource.connect() as conn:
        for station in station_list:
            result = fetch_hourly(station, start, end, fetcher)
            request_count += len(month_chunks(start, end))
            warehouse.load_hourly(conn, station.icao, start, end, result.rows)
            warehouse.load_gaps(
                conn, station.icao, "asos", start, end, result.gaps, now=now
            )
            gaps_by_station[station.icao] = result.gaps

    context.log.info(
        "raw/asos_hourly %s..%s: %d stations, %d requests%s",
        start,
        end,
        len(station_list),
        request_count,
        _slowest_attempt_log(fetcher),
    )
    return dg.MaterializeResult(
        metadata={
            "station_count": len(station_list),
            "request_count": request_count,
            **_http_stats_metadata(fetcher),
        },
        check_results=[
            _gap_rate_check_result("asos_gap_rate", gaps_by_station, start, end)
        ],
    )


@dg.asset(
    key=RAW_CLI_DAILY_KEY,
    partitions_def=daily_partitions,
    backfill_policy=INGEST_BACKFILL_POLICY,
    description=(
        "NWS CLI daily high/low reports, one partition per station-local "
        "climate date. fetch_cli's [start, end] is already a local-date "
        "range (it fetches the calendar year(s) touched and filters to "
        "local_date locally), so the partition window is passed straight "
        "through with no boundary adaptation."
    ),
)
def raw_cli_daily(
    context: dg.AssetExecutionContext,
    warehouse_resource: WarehouseResource,
    iem: IemResource,
    stations: StationsResource,
) -> dg.MaterializeResult:
    start, end = _partition_date_range(context)
    station_list = stations.stations()
    fetcher = iem.fetcher()
    now = datetime.now(UTC).replace(tzinfo=None)
    request_count = 0

    with warehouse_resource.connect() as conn:
        for station in station_list:
            result = fetch_cli(station, start, end, fetcher)
            request_count += end.year - start.year + 1
            warehouse.load_cli(conn, station.icao, start, end, result.rows)
            warehouse.load_gaps(
                conn, station.icao, "cli", start, end, result.gaps, now=now
            )

    context.log.info(
        "raw/cli_daily %s..%s: %d stations, %d requests%s",
        start,
        end,
        len(station_list),
        request_count,
        _slowest_attempt_log(fetcher),
    )
    return dg.MaterializeResult(
        metadata={
            "station_count": len(station_list),
            "request_count": request_count,
            **_http_stats_metadata(fetcher),
        }
    )


@dg.asset(
    key=RAW_RESOLVED_WINDOWS_KEY,
    partitions_def=daily_partitions,
    backfill_policy=INGEST_BACKFILL_POLICY,
    deps=[
        RAW_NBS_GUIDANCE_KEY,
        dg.AssetDep(
            asset=RAW_ASOS_HOURLY_KEY,
            partition_mapping=dg.TimeWindowPartitionMapping(
                start_offset=-OBS_LOOKBACK_DAYS,
                end_offset=OBS_LOOKAHEAD_DAYS,
                allow_nonexistent_upstream_partitions=True,
            ),
        ),
    ],
    description=(
        "Verification windows resolved from raw guidance and hourly obs, "
        "one partition per guidance run date (UTC); replaces "
        "pipeline.resolve_station's output per station over that date. "
        "Depends on raw/nbs_guidance through the identity mapping and on "
        "raw/asos_hourly through D-OBS_LOOKBACK_DAYS..D+OBS_LOOKAHEAD_DAYS "
        "(matching pipeline.OBS_LOOKBACK_DAYS=1 / OBS_LOOKAHEAD_DAYS=4): a "
        "run on D carries a lead-3 max whose window closes D+4 06Z. "
        "Consequence: a resolved partition for a recent D is incomplete "
        "(its windows are unscorable) until asos D+4 exists, and must be "
        "re-materialized then -- scheduling that is #16's job, not this "
        "build's."
    ),
)
def raw_resolved_windows(
    context: dg.AssetExecutionContext,
    warehouse_resource: WarehouseResource,
    stations: StationsResource,
) -> dg.MaterializeResult:
    start, end = _partition_date_range(context)
    station_list = stations.stations()
    total_resolved = 0

    with warehouse_resource.connect() as conn:
        for station in station_list:
            total_resolved += resolve_station(conn, station.icao, start, end)

    context.log.info(
        "raw/resolved_windows %s..%s: %d stations, %d resolved rows",
        start,
        end,
        len(station_list),
        total_resolved,
    )
    return dg.MaterializeResult(
        metadata={"station_count": len(station_list), "resolved_rows": total_resolved}
    )


RAW_INGEST_ASSETS = [
    raw_nbs_guidance,
    raw_asos_hourly,
    raw_cli_daily,
    raw_resolved_windows,
]


# -- D7: checks wired to the pure evaluators in weather_forecast_audit.checks

@dg.asset_check(asset=RAW_NBS_GUIDANCE_KEY, name="guidance_freshness")
def guidance_freshness(
    warehouse_resource: WarehouseResource, clock: ClockResource
) -> dg.AssetCheckResult:
    """The latest guidance run is no more than GUIDANCE_MAX_AGE_HOURS old."""
    with warehouse_resource.connect() as conn:
        latest = conn.execute(
            "select max(runtime_utc) from raw.nbs_guidance"
        ).fetchone()[0]
    passed, metadata = evaluate_freshness(
        latest, clock.now(), timedelta(hours=GUIDANCE_MAX_AGE_HOURS)
    )
    return dg.AssetCheckResult(
        passed=passed, severity=dg.AssetCheckSeverity.ERROR, metadata=metadata
    )


@dg.asset_check(asset=RAW_ASOS_HOURLY_KEY, name="obs_freshness")
def obs_freshness(
    warehouse_resource: WarehouseResource, clock: ClockResource
) -> dg.AssetCheckResult:
    """The latest hourly ob is no more than OBS_MAX_AGE_HOURS old."""
    with warehouse_resource.connect() as conn:
        latest = conn.execute(
            "select max(valid_utc) from raw.asos_hourly"
        ).fetchone()[0]
    passed, metadata = evaluate_freshness(
        latest, clock.now(), timedelta(hours=OBS_MAX_AGE_HOURS)
    )
    return dg.AssetCheckResult(
        passed=passed, severity=dg.AssetCheckSeverity.ERROR, metadata=metadata
    )


FRESHNESS_CHECKS = [guidance_freshness, obs_freshness]
