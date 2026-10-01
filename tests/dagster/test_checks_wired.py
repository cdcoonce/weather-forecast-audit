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


def _gap_rate_results(
    db_path: Path, only: list[str], iem_resource: object
) -> dict[str, tuple[bool, str]]:
    test_defs = build_ingest_test_defs(str(db_path), only, iem_resource)
    job = test_defs.resolve_job_def("ingest_job")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance,
            partition_key="2023-07-14",
            asset_selection=[RAW_NBS_GUIDANCE_KEY],
        )
        assert result.success
    return {
        e.check_name: (e.passed, e.severity.value)
        for e in result.get_asset_check_evaluations()
    }


def test_gap_rate_check_fails_warn_when_gaps_exceed_threshold(tmp_path: Path) -> None:
    """One of two stations (KORD) has a missing guidance run: 50% > 5%."""
    db_path = tmp_path / "wfa.duckdb"

    evaluations = _gap_rate_results(
        db_path, ["KPHX", "KORD"], OneStationMissingGuidanceIemResource()
    )

    passed, severity = evaluations["guidance_gap_rate"]
    assert passed is False
    # D7: gap-rate failures are a WARN, not an ERROR -- a backfill's failing
    # gap rate must not be treated as gravely as a stale/empty table.
    assert severity == "WARN"


def test_gap_rate_check_passes_when_no_stations_have_a_gap(tmp_path: Path) -> None:
    db_path = tmp_path / "wfa.duckdb"

    evaluations = _gap_rate_results(db_path, ["KPHX", "KORD"], FixtureIemResource())

    passed, severity = evaluations["guidance_gap_rate"]
    assert passed is True
    assert severity == "WARN"


def test_gap_rate_counts_ambiguous_gap_once_per_station_under_warn() -> None:
    # Characterization: the check counts stations with any gap in the window,
    # so an ambiguous 6-hour group marks its station once (however many
    # gaps it has) and a lone one among 40 stations does not trip WARN.
    from datetime import date

    from weather_forecast_audit.assets import _gap_rate_check_result
    from weather_forecast_audit.gaps import GapRecord

    day = date(2020, 11, 23)
    gaps_by_station = {f"K{i:03d}": [] for i in range(39)}
    gaps_by_station["KCAK"] = [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@05:51"),
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@17:51"),
        GapRecord("KCAK", "asos", "2020-11-23", "missing_observations"),
    ]

    result = _gap_rate_check_result("asos_gap_rate", gaps_by_station, day, day)

    assert result.passed is True
    assert result.metadata["stations_with_gap"].value == 1
    assert result.metadata["stations_total"].value == 40
