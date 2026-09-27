import pytest
from dagster import AssetCheckKey, AssetKey, DagsterInstance

from weather_forecast_audit.definitions import (
    FRESHNESS_CHECK_JOB_SELECTION,
    INGEST_JOB_SELECTION,
    defs,
)
from weather_forecast_audit.regimes import load_archive_start

pytestmark = pytest.mark.dagster

RAW_KEYS = [
    AssetKey(["raw", "nbs_guidance"]),
    AssetKey(["raw", "asos_hourly"]),
    AssetKey(["raw", "cli_daily"]),
    AssetKey(["raw", "resolved_windows"]),
]


def test_definitions_expose_platform_smoke_job_with_max_runtime() -> None:
    job = defs.resolve_job_def("platform_smoke_job")

    assert AssetKey("platform_smoke") in defs.resolve_asset_graph().get_all_asset_keys()
    # One run slot on rammingspeed: a hung run would block every tenant.
    assert int(job.tags["dagster/max_runtime"]) > 0


def test_definitions_expose_expected_asset_keys() -> None:
    all_keys = defs.resolve_asset_graph().get_all_asset_keys()

    for key in [*RAW_KEYS, AssetKey(["raw", "ingest_gaps"])]:
        assert key in all_keys
    for name in [
        "stg_nbs_guidance",
        "stg_asos_hourly",
        "stg_cli_daily",
        "stg_ingest_gaps",
        "stg_resolved_windows",
        "gap_ledger",
        "fct_forecast_verification",
    ]:
        assert AssetKey(name) in all_keys


def test_definitions_expose_ingest_transform_and_freshness_jobs() -> None:
    for name in ["ingest_job", "transform_job", "freshness_check_job"]:
        job = defs.resolve_job_def(name)
        assert int(job.tags["dagster/max_runtime"]) > 0


def test_raw_partitions_start_at_the_pinned_archive_start() -> None:
    asset_graph = defs.resolve_asset_graph()
    archive_start = load_archive_start()

    for key in RAW_KEYS:
        partitions_def = asset_graph.get(key).partitions_def
        assert partitions_def is not None
        assert archive_start.isoformat() in partitions_def.get_partition_keys()[:1]


def test_raw_nbs_guidance_and_asos_hourly_are_upstream_of_staging() -> None:
    asset_graph = defs.resolve_asset_graph()

    guidance_downstream = {
        node.key for node in asset_graph.asset_nodes
        if AssetKey(["raw", "nbs_guidance"]) in node.parent_keys
    }
    assert AssetKey("stg_nbs_guidance") in guidance_downstream

    asos_downstream = {
        node.key for node in asset_graph.asset_nodes
        if AssetKey(["raw", "asos_hourly"]) in node.parent_keys
    }
    assert AssetKey("stg_asos_hourly") in asos_downstream


def test_raw_ingest_gaps_is_upstream_of_stg_ingest_gaps() -> None:
    asset_graph = defs.resolve_asset_graph()

    downstream = {
        node.key for node in asset_graph.asset_nodes
        if AssetKey(["raw", "ingest_gaps"]) in node.parent_keys
    }
    assert AssetKey("stg_ingest_gaps") in downstream


def test_ingest_job_excludes_freshness_checks_but_keeps_gap_rate_checks() -> None:
    """D6: a backfill of old dates must never fail a freshness check."""
    asset_graph = defs.resolve_asset_graph()

    check_keys = INGEST_JOB_SELECTION.resolve_checks(asset_graph)
    guidance_key = AssetKey(["raw", "nbs_guidance"])
    asos_key = AssetKey(["raw", "asos_hourly"])

    assert AssetCheckKey(guidance_key, "guidance_freshness") not in check_keys
    assert AssetCheckKey(asos_key, "obs_freshness") not in check_keys
    assert AssetCheckKey(guidance_key, "guidance_gap_rate") in check_keys
    assert AssetCheckKey(asos_key, "asos_gap_rate") in check_keys


def test_freshness_check_job_selects_only_the_two_freshness_checks() -> None:
    asset_graph = defs.resolve_asset_graph()

    check_keys = FRESHNESS_CHECK_JOB_SELECTION.resolve_checks(asset_graph)

    assert check_keys == {
        AssetCheckKey(AssetKey(["raw", "nbs_guidance"]), "guidance_freshness"),
        AssetCheckKey(AssetKey(["raw", "asos_hourly"]), "obs_freshness"),
    }
    # Materializes nothing: only check-bearing nodes are selected.
    assert FRESHNESS_CHECK_JOB_SELECTION.resolve(asset_graph) == set()


@pytest.mark.io
def test_platform_smoke_materializes_with_report_metadata() -> None:
    job = defs.resolve_job_def("platform_smoke_job")

    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(instance=instance)

    assert result.success
    [materialization] = result.asset_materializations_for_node("platform_smoke")
    metadata = {key: value.value for key, value in materialization.metadata.items()}
    assert metadata["polars_runtime"] == "compat"
    assert metadata["lightgbm_deterministic"] is True
    assert metadata["duckdb_rows"] == 6
    assert metadata["polars_duckdb_insert_rows"] == 2
    assert metadata["version_pyarrow"]


@pytest.mark.parametrize(
    "job_name",
    ["platform_smoke_job", "ingest_job", "transform_job", "freshness_check_job"],
)
def test_every_job_runs_steps_in_one_process(job_name: str) -> None:
    """DuckDB allows one writer process per database file. Under Dagster's
    default multiprocess executor, one run's independent raw assets execute
    in parallel subprocesses that each open the file for writing, so all but
    the first fail on the lock (rammingspeed run 8e53a237, 2026-09-27). The
    single run slot serializes runs, not steps. `execute_in_process` in the
    other tests always runs steps sequentially, so only this assertion can
    see the production executor.
    """
    job = defs.resolve_job_def(job_name)
    assert job.executor_def.name == "in_process"
