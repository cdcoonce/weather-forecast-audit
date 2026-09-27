"""Dagster definitions: the `weather-forecast-audit` code location.

Served on rammingspeed by `dagster api grpc -m weather_forecast_audit.definitions`
on port 4002, behind the shared webserver/daemon and its single run slot.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    Definitions,
    MaterializeResult,
    asset,
    define_asset_job,
    in_process_executor,
)

from weather_forecast_audit.assets import (
    FRESHNESS_CHECKS,
    RAW_INGEST_ASSETS,
    ingest_gaps_spec,
)
from weather_forecast_audit.dbt_assets import dbt_resource, dbt_transform_assets
from weather_forecast_audit.jobs import (
    FRESHNESS_CHECK_JOB_SELECTION,
    INGEST_JOB_SELECTION,
    MAX_RUNTIME_SECONDS,
    freshness_check_job,
    ingest_job,
    transform_job,
)
from weather_forecast_audit.platform_smoke import PlatformSmokeError, run_platform_smoke
from weather_forecast_audit.resources import (
    ClockResource,
    IemResource,
    StationsResource,
    WarehouseResource,
)
from weather_forecast_audit.sensors import national_backfill_sensor

__all__ = [
    "FRESHNESS_CHECK_JOB_SELECTION",
    "INGEST_JOB_SELECTION",
    "defs",
    "freshness_check_job",
    "ingest_job",
    "transform_job",
]


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

# ingest_job, transform_job, freshness_check_job (and the constants behind
# them) live in jobs.py: sensors.py targets ingest_job/transform_job too,
# and putting them in a leaf module both can import avoids a cycle between
# this module (which registers the sensor) and sensors.py (which targets
# these jobs).

# One process per run: DuckDB allows a single writer process per database
# file, and the default multiprocess executor would run a run's independent
# raw assets in parallel subprocesses that race for the write lock
# (rammingspeed run 8e53a237). The run slot serializes runs; this serializes
# steps. Set on Definitions so every job, including future ones, inherits it.
defs = Definitions(
    executor=in_process_executor,
    assets=[
        platform_smoke,
        *RAW_INGEST_ASSETS,
        ingest_gaps_spec,
        dbt_transform_assets,
    ],
    asset_checks=FRESHNESS_CHECKS,
    jobs=[platform_smoke_job, ingest_job, transform_job, freshness_check_job],
    sensors=[national_backfill_sensor],
    resources={
        "warehouse_resource": WarehouseResource(),
        "iem": IemResource(),
        "stations": StationsResource(),
        "clock": ClockResource(),
        "dbt": dbt_resource,
    },
)
