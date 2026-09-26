import pytest
from dagster import AssetKey, DagsterInstance

from weather_forecast_audit.definitions import defs

pytestmark = pytest.mark.dagster


def test_definitions_expose_platform_smoke_job_with_max_runtime() -> None:
    job = defs.resolve_job_def("platform_smoke_job")

    assert AssetKey("platform_smoke") in defs.resolve_asset_graph().get_all_asset_keys()
    # One run slot on rammingspeed: a hung run would block every tenant.
    assert int(job.tags["dagster/max_runtime"]) > 0


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
