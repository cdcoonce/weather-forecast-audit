"""transform_job after materializing two stations' ingest (build spec #10 test 9).

Reuses tests/dagster/test_ingest_materialize.py's seeding (KPHX + KORD,
2023-07-14, with KORD's CLI report engineered missing) so both the "both
stations show up in the fact table" and "gap_ledger has a real non-null
first_seen row" assertions hold against the same data.
"""

import os
from pathlib import Path

import duckdb
import pytest
from conftest import OneStationMissingCliReportIemResource, build_ingest_test_defs
from dagster import AssetKey, DagsterInstance

from weather_forecast_audit.definitions import defs

pytestmark = [pytest.mark.dagster, pytest.mark.dbt, pytest.mark.io]

RAW_NBS_GUIDANCE_KEY = AssetKey(["raw", "nbs_guidance"])
RAW_ASOS_HOURLY_KEY = AssetKey(["raw", "asos_hourly"])
RAW_CLI_DAILY_KEY = AssetKey(["raw", "cli_daily"])
RAW_RESOLVED_WINDOWS_KEY = AssetKey(["raw", "resolved_windows"])

ASOS_DATES = [
    "2023-07-13",
    "2023-07-14",
    "2023-07-15",
    "2023-07-16",
    "2023-07-17",
    "2023-07-18",
]


def _materialize_kphx_kord(db_path: Path, instance: DagsterInstance) -> None:
    test_defs = build_ingest_test_defs(
        str(db_path), ["KPHX", "KORD"], OneStationMissingCliReportIemResource()
    )
    job = test_defs.resolve_job_def("ingest_job")

    for partition in ASOS_DATES:
        result = job.execute_in_process(
            instance=instance,
            partition_key=partition,
            asset_selection=[RAW_ASOS_HOURLY_KEY],
        )
        assert result.success, partition

    result = job.execute_in_process(
        instance=instance,
        partition_key="2023-07-14",
        asset_selection=[RAW_NBS_GUIDANCE_KEY, RAW_CLI_DAILY_KEY],
    )
    assert result.success

    result = job.execute_in_process(
        instance=instance,
        partition_key="2023-07-14",
        asset_selection=[RAW_RESOLVED_WINDOWS_KEY],
    )
    assert result.success


def test_transform_job_builds_fct_and_gap_ledger_for_two_stations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "wfa.duckdb"

    with DagsterInstance.ephemeral() as instance:
        _materialize_kphx_kord(db_path, instance)

    # dbt_transform_assets' DbtCliResource reads the checked-in dbt profile,
    # which resolves the DuckDB path via env_var("WFA_DUCKDB_PATH") at each
    # invocation -- so pointing it at this test's temp DB is just an env var.
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    transform_job = defs.resolve_job_def("transform_job")
    with DagsterInstance.ephemeral() as instance:
        result = transform_job.execute_in_process(instance=instance)
    assert result.success

    with duckdb.connect(str(db_path)) as conn:
        fct_stations = {
            row[0]
            for row in conn.execute(
                "select distinct station from fct_forecast_verification"
            ).fetchall()
        }
        gap_ledger_first_seen = conn.execute(
            "select first_seen from gap_ledger where station = 'KORD' "
            "and kind = 'cli' and reason = 'missing_report'"
        ).fetchall()

    assert fct_stations == {"KPHX", "KORD"}
    assert gap_ledger_first_seen
    assert all(row[0] is not None for row in gap_ledger_first_seen)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # dbt's profile also reads WFA_SNOWFLAKE_* if present; the tracer test's
    # own fixture does the same cleanup so a leftover env var never redirects
    # this DuckDB-only build.
    for name in [n for n in os.environ if n.startswith("WFA_SNOWFLAKE_")]:
        monkeypatch.delenv(name, raising=False)
