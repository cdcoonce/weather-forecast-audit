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
)

from weather_forecast_audit.platform_smoke import PlatformSmokeError, run_platform_smoke

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
            **{f"version_{name}": v for name, v in report.versions.items()},
        }
    )


platform_smoke_job = define_asset_job(
    "platform_smoke_job",
    selection=[platform_smoke],
    tags={"dagster/max_runtime": str(MAX_RUNTIME_SECONDS)},
)

defs = Definitions(assets=[platform_smoke], jobs=[platform_smoke_job])
