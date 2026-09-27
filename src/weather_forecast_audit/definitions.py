"""Dagster definitions: the `weather-forecast-audit` code location.

Served on rammingspeed by `dagster api grpc -m weather_forecast_audit.definitions`
on port 4002, behind the shared webserver/daemon and its single run slot.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

from dagster import (
    AssetCheckKey,
    AssetExecutionContext,
    AssetSelection,
    Definitions,
    MaterializeResult,
    asset,
    define_asset_job,
)

from weather_forecast_audit.assets import (
    FRESHNESS_CHECKS,
    RAW_ASOS_HOURLY_KEY,
    RAW_INGEST_ASSETS,
    RAW_NBS_GUIDANCE_KEY,
    ingest_gaps_spec,
)
from weather_forecast_audit.dbt_assets import dbt_resource, dbt_transform_assets
from weather_forecast_audit.platform_smoke import PlatformSmokeError, run_platform_smoke
from weather_forecast_audit.resources import (
    ClockResource,
    IemResource,
    StationsResource,
    WarehouseResource,
)

# rammingspeed has one run slot shared by every tenant, so every job carries a
# max runtime: a hung run would otherwise block oura and waga indefinitely.
MAX_RUNTIME_SECONDS = 600


@asset(description="Proves DuckDB, LightGBM and Polars run on the host CPU.")
def platform_smoke(context: AssetExecutionContext) -> MaterializeResult:
    with tempfile.TemporaryDirectory() as workdir:
        report = run_platform_smoke(Path(workdir))
    context.log.info("platform smoke report: %s", asdict(report))
    if report.polars_runtime != "compat":
        msg = f"Polars loaded the {report.polars_runtime!r} runtime, not 'compat'"
        raise PlatformSmokeError(msg)
    return MaterializeResult(
        metadata={
            "avx2": str(report.avx2),
            "duckdb_rows": report.duckdb.rows,
            "lightgbm_deterministic": report.lightgbm.deterministic,
            "lightgbm_r2": report.lightgbm.r2,
            "polars_runtime": report.polars_runtime,
            "polars_duckdb_insert_rows": report.polars_duckdb_insert.rows,
            **{f"version_{name}": v for name, v in report.versions.items()},
        }
    )


platform_smoke_job = define_asset_job(
    "platform_smoke_job",
    selection=[platform_smoke],
    tags={"dagster/max_runtime": str(MAX_RUNTIME_SECONDS)},
)

# D6: the four partitioned raw assets, batched into one job so a run
# materializes guidance/asos/cli/resolved together for the same partitions.
# AssetSelection.assets(...) pulls in every check on those assets by
# default, including the two freshness checks (co-located on nbs_guidance/
# asos_hourly only because they share those assets' resources) -- excluded
# explicitly, because a backfill of old dates must never fail a freshness
# check (D6). The gap-rate checks stay in: they run with the
# materialization by design.
FRESHNESS_CHECK_KEYS = [
    AssetCheckKey(asset_key=RAW_NBS_GUIDANCE_KEY, name="guidance_freshness"),
    AssetCheckKey(asset_key=RAW_ASOS_HOURLY_KEY, name="obs_freshness"),
]

# Kept as a module-level name (not inlined into define_asset_job) so tests
# can call .resolve_checks(asset_graph) on the exact selection each job
# runs -- resolve_job_def's returned JobDefinition does not expose it back.
INGEST_JOB_SELECTION = AssetSelection.assets(
    *RAW_INGEST_ASSETS
) - AssetSelection.checks(*FRESHNESS_CHECK_KEYS)

ingest_job = define_asset_job(
    "ingest_job",
    selection=INGEST_JOB_SELECTION,
    tags={"dagster/max_runtime": str(MAX_RUNTIME_SECONDS)},
)

# dbt assets are unpartitioned: a full `dbt build` over all raw data (D4).
transform_job = define_asset_job(
    "transform_job",
    selection=AssetSelection.assets(dbt_transform_assets),
    tags={"dagster/max_runtime": str(MAX_RUNTIME_SECONDS)},
)

# D6: only the two freshness checks, materializing nothing -- a backfill of
# old dates must never fail a freshness check, so this must not run inside
# ingest_job. #16 schedules this job; no schedule is added here.
FRESHNESS_CHECK_JOB_SELECTION = AssetSelection.checks(*FRESHNESS_CHECK_KEYS)

freshness_check_job = define_asset_job(
    "freshness_check_job",
    selection=FRESHNESS_CHECK_JOB_SELECTION,
    tags={"dagster/max_runtime": str(MAX_RUNTIME_SECONDS)},
)

defs = Definitions(
    assets=[
        platform_smoke,
        *RAW_INGEST_ASSETS,
        ingest_gaps_spec,
        dbt_transform_assets,
    ],
    asset_checks=FRESHNESS_CHECKS,
    jobs=[platform_smoke_job, ingest_job, transform_job, freshness_check_job],
    resources={
        "warehouse_resource": WarehouseResource(),
        "iem": IemResource(),
        "stations": StationsResource(),
        "clock": ClockResource(),
        "dbt": dbt_resource,
    },
)
