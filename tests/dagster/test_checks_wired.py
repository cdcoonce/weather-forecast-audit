"""Checks wired: freshness_check_job and a real ingest_job's gap-rate checks
(build spec #10 test 8).
"""

from datetime import timedelta
from pathlib import Path

import duckdb
import pytest
from conftest import (
    FixtureIemResource,
    OneStationMissingGuidanceIemResource,
    build_ingest_test_defs,
)
from dagster import AssetKey, DagsterInstance, Definitions

from weather_forecast_audit.assets import FRESHNESS_CHECKS, RAW_INGEST_ASSETS
from weather_forecast_audit.definitions import freshness_check_job
from weather_forecast_audit.resources import (
    ClockResource,
    StationsResource,
    WarehouseResource,
)

pytestmark = [pytest.mark.dagster, pytest.mark.io]

RAW_NBS_GUIDANCE_KEY = AssetKey(["raw", "nbs_guidance"])
RAW_ASOS_HOURLY_KEY = AssetKey(["raw", "asos_hourly"])


def _seed_kphx_guidance_and_asos(db_path: Path) -> None:
    ingest_defs = build_ingest_test_defs(str(db_path), ["KPHX"], FixtureIemResource())
    job = ingest_defs.resolve_job_def("ingest_job")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance,
            partition_key="2023-07-14",
            asset_selection=[RAW_NBS_GUIDANCE_KEY, RAW_ASOS_HOURLY_KEY],
        )
        assert result.success


def _freshness_defs(db_path: Path, now_override: str) -> Definitions:
    # RAW_INGEST_ASSETS must be present so the two checks (attached to
    # raw/nbs_guidance and raw/asos_hourly) resolve, even though this job
    # selects only the checks and materializes nothing; `iem`/`stations` are
    # required to satisfy those (unexecuted) assets' resource requirements.
    return Definitions(
        assets=RAW_INGEST_ASSETS,
        asset_checks=FRESHNESS_CHECKS,
        jobs=[freshness_check_job],
        resources={
            "warehouse_resource": WarehouseResource(duckdb_path=str(db_path)),
            "iem": FixtureIemResource(),
            "stations": StationsResource(only=["KPHX"]),
            "clock": ClockResource(now_override=now_override),
        },
    )


def test_freshness_check_job_passes_just_after_and_fails_far_after(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wfa.duckdb"
    _seed_kphx_guidance_and_asos(db_path)

    with duckdb.connect(str(db_path)) as conn:
        latest_guidance = conn.execute(
            "select max(runtime_utc) from raw.nbs_guidance"
        ).fetchone()[0]
        latest_obs = conn.execute(
            "select max(valid_utc) from raw.asos_hourly"
        ).fetchone()[0]
    latest = max(latest_guidance, latest_obs)

    def _evaluate(now_override: str) -> dict[str, bool]:
        defs = _freshness_defs(db_path, now_override)
        job = defs.resolve_job_def("freshness_check_job")
        with DagsterInstance.ephemeral() as instance:
            result = job.execute_in_process(instance=instance)
        return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}

    just_after = (latest + timedelta(hours=1)).isoformat() + "Z"
    far_after = (latest + timedelta(hours=1000)).isoformat() + "Z"

    assert _evaluate(just_after) == {
        "guidance_freshness": True,
        "obs_freshness": True,
    }
    assert _evaluate(far_after) == {
        "guidance_freshness": False,
        "obs_freshness": False,
    }


def _gap_rate_results(db_path: Path, only: list[str], iem_resource: object) -> dict:
    test_defs = build_ingest_test_defs(str(db_path), only, iem_resource)
    job = test_defs.resolve_job_def("ingest_job")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance,
            partition_key="2023-07-14",
            asset_selection=[RAW_NBS_GUIDANCE_KEY],
        )
        assert result.success
    return {e.check_name: e.passed for e in result.get_asset_check_evaluations()}


def test_gap_rate_check_fails_warn_when_gaps_exceed_threshold(tmp_path: Path) -> None:
    """One of two stations (KORD) has a missing guidance run: 50% > 5%."""
    db_path = tmp_path / "wfa.duckdb"

    evaluations = _gap_rate_results(
        db_path, ["KPHX", "KORD"], OneStationMissingGuidanceIemResource()
    )

    assert evaluations["guidance_gap_rate"] is False


def test_gap_rate_check_passes_when_no_stations_have_a_gap(tmp_path: Path) -> None:
    db_path = tmp_path / "wfa.duckdb"

    evaluations = _gap_rate_results(db_path, ["KPHX", "KORD"], FixtureIemResource())

    assert evaluations["guidance_gap_rate"] is True
