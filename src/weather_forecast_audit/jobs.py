"""Job definitions shared by `definitions.py` (the code location) and
`sensors.py` (the national backfill sensor, issue #11).

Split out of `definitions.py` so `sensors.py` can import `ingest_job` and
`transform_job` without an import cycle: `definitions.py` registers
`national_backfill_sensor`, so it must import `sensors.py`, and `sensors.py`
must in turn target `ingest_job`/`transform_job` -- putting the job objects
in their own leaf module lets both import from here instead of from each
other. `platform_smoke_job` stays in `definitions.py`: it only selects the
`platform_smoke` asset defined there, and the sensor never touches it.
"""

from dagster import AssetCheckKey, AssetSelection, define_asset_job

from weather_forecast_audit.assets import (
    RAW_ASOS_HOURLY_KEY,
    RAW_INGEST_ASSETS,
    RAW_NBS_GUIDANCE_KEY,
)
from weather_forecast_audit.dbt_assets import dbt_transform_assets

# rammingspeed has one run slot shared by every tenant, so every job carries a
# max runtime: a hung run would otherwise block oura and waga indefinitely.
MAX_RUNTIME_SECONDS = 600

# Run-time model for one national-backfill unit (one month x a chunk of
# stations, `sensors.BACKFILL_CHUNK_SIZE`). Measured on the live backfill:
# ~17s per station across the four ingest steps (raw__asos_hourly ~8s,
# raw__nbs_guidance ~7.6s, cli ~1.2s, resolved ~0.2s). Slow units carry an
# extra stall of ~60-82s (one request hits the HTTP client's 60s timeout,
# then backoff and a retry). The cap is enforced by a polling monitor, so a
# unit that only just fits survives by timing luck; a unit must therefore
# fit its baseline PLUS two stalls inside 95% of the cap. Never raise
# MAX_RUNTIME_SECONDS to make a bigger chunk fit: it protects a run slot
# shared with other tenants. The guard is a test (tests/dagster/
# test_sensors.py), deliberately not an import-time assertion.
OBSERVED_SECONDS_PER_STATION = 17.0
OBSERVED_STALL_SECONDS = 82.0
STALLS_TOLERATED = 2
RUNTIME_BUDGET_FRACTION = 0.95


def unit_runtime_budget_s(chunk_size: int) -> float:
    """Modelled worst-case seconds for a unit of `chunk_size` stations:
    the per-station baseline plus `STALLS_TOLERATED` request stalls."""
    return (
        chunk_size * OBSERVED_SECONDS_PER_STATION
        + STALLS_TOLERATED * OBSERVED_STALL_SECONDS
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
