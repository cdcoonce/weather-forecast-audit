"""Materializing ingest_job for two stations: rows, gaps, idempotency, and
partition ownership (build spec #10 tests 3, 4, 5).
"""

from pathlib import Path

import duckdb
import pytest
from conftest import (
    DagsterFixtureFetcher,
    FixtureIemResource,
    OneStationMissingCliReportIemResource,
    build_ingest_test_defs,
)
from dagster import AssetKey, DagsterInstance

from weather_forecast_audit.iem.http import HttpResponse

pytestmark = [pytest.mark.dagster, pytest.mark.io]

RAW_NBS_GUIDANCE_KEY = AssetKey(["raw", "nbs_guidance"])
RAW_ASOS_HOURLY_KEY = AssetKey(["raw", "asos_hourly"])
RAW_CLI_DAILY_KEY = AssetKey(["raw", "cli_daily"])
RAW_RESOLVED_WINDOWS_KEY = AssetKey(["raw", "resolved_windows"])

# The asos range resolved_windows 2023-07-14 depends on (test_partition_mapping.py).
ASOS_DATES = [
    "2023-07-13",
    "2023-07-14",
    "2023-07-15",
    "2023-07-16",
    "2023-07-17",
    "2023-07-18",
]

RAW_TABLES = [
    "nbs_guidance",
    "asos_hourly",
    "cli_daily",
    "resolved_windows",
    "ingest_gaps",
]


def _materialize_kphx_kord(
    db_path: Path, instance: DagsterInstance, iem_resource: object
) -> None:
    """The asos range first, then guidance+cli, then resolved, for 07-14."""
    test_defs = build_ingest_test_defs(str(db_path), ["KPHX", "KORD"], iem_resource)
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


def _table_counts(db_path: Path) -> dict[str, tuple]:
    with duckdb.connect(str(db_path)) as conn:
        return {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()  # noqa: S608
            for table in RAW_TABLES
        }


def test_materialize_one_partition_for_two_stations_writes_rows_and_gaps(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wfa.duckdb"

    with DagsterInstance.ephemeral() as instance:
        _materialize_kphx_kord(
            db_path, instance, OneStationMissingCliReportIemResource()
        )

    with duckdb.connect(str(db_path)) as conn:
        guidance_stations = {
            row[0]
            for row in conn.execute(
                "select distinct station from raw.nbs_guidance"
            ).fetchall()
        }
        asos_stations = {
            row[0]
            for row in conn.execute(
                "select distinct station from raw.asos_hourly"
            ).fetchall()
        }
        resolved_stations = {
            row[0]
            for row in conn.execute(
                "select distinct station from raw.resolved_windows"
            ).fetchall()
        }
        gap_rows = conn.execute(
            "select station, source, reason from raw.ingest_gaps"
        ).fetchall()

    # Both stations get rows in every raw table that has any data at all --
    # KORD's engineered gap is in cli_daily (a missing_report), which
    # pipeline.resolve_station never reads, so it doesn't cascade into
    # guidance/asos/resolved.
    assert guidance_stations == {"KPHX", "KORD"}
    assert asos_stations == {"KPHX", "KORD"}
    assert resolved_stations == {"KPHX", "KORD"}
    assert ("KORD", "cli", "missing_report") in {(r[0], r[1], r[2]) for r in gap_rows}


def test_idempotent_rematerialize_same_row_counts_and_first_seen(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wfa.duckdb"

    with DagsterInstance.ephemeral() as instance:
        _materialize_kphx_kord(
            db_path, instance, OneStationMissingCliReportIemResource()
        )
    counts_first = _table_counts(db_path)
    with duckdb.connect(str(db_path)) as conn:
        first_seen_before = sorted(
            conn.execute(
                "select station, source, expected, reason, first_seen "
                "from raw.ingest_gaps"
            ).fetchall()
        )

    # A second, later materialization -- using a fresh instance and letting
    # the real wall clock advance between the two calls, so first_seen
    # preservation is proven against an actually later `now`, not an
    # injected one (raw ingest assets don't take ClockResource; only the
    # checks do, per D5).
    with DagsterInstance.ephemeral() as instance:
        _materialize_kphx_kord(
            db_path, instance, OneStationMissingCliReportIemResource()
        )
    counts_second = _table_counts(db_path)
    with duckdb.connect(str(db_path)) as conn:
        first_seen_after = sorted(
            conn.execute(
                "select station, source, expected, reason, first_seen "
                "from raw.ingest_gaps"
            ).fetchall()
        )

    assert counts_first == counts_second
    assert counts_first["asos_hourly"][0] > 0
    assert counts_first["ingest_gaps"][0] > 0
    assert first_seen_before == first_seen_after


def test_asos_partition_ownership(tmp_path: Path) -> None:
    db_path = tmp_path / "wfa.duckdb"
    test_defs = build_ingest_test_defs(str(db_path), ["KPHX"], FixtureIemResource())
    job = test_defs.resolve_job_def("ingest_job")

    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance,
            partition_key="2023-07-14",
            asset_selection=[RAW_ASOS_HOURLY_KEY],
        )
        assert result.success

    with duckdb.connect(str(db_path)) as conn:
        rows_0714_before_0715 = conn.execute(
            "select * from raw.asos_hourly where station = 'KPHX' order by valid_utc"
        ).fetchall()

    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance,
            partition_key="2023-07-15",
            asset_selection=[RAW_ASOS_HOURLY_KEY],
        )
        assert result.success

    with duckdb.connect(str(db_path)) as conn:
        rows_0714_after_0715 = conn.execute(
            "select * from raw.asos_hourly where station = 'KPHX' "
            "and cast(valid_utc as date) = '2023-07-14' order by valid_utc"
        ).fetchall()

    # Materializing 07-15 must leave 07-14's rows byte-identical.
    assert rows_0714_before_0715 == rows_0714_after_0715
    assert rows_0714_before_0715  # sanity: 07-14 actually had rows

    class FewerRowsFetcher(DagsterFixtureFetcher):
        def get(self, url: str) -> HttpResponse:
            response = super().get(url)
            if "asos.py" not in url:
                return response
            header, *rows = response.body.decode().splitlines()
            trimmed = rows[: len(rows) // 2]
            return HttpResponse(200, ("\n".join([header, *trimmed]) + "\n").encode())

    class FewerRowsIemResource(FixtureIemResource):
        def fetcher(self) -> FewerRowsFetcher:
            return FewerRowsFetcher()

    fewer_defs = build_ingest_test_defs(str(db_path), ["KPHX"], FewerRowsIemResource())
    fewer_job = fewer_defs.resolve_job_def("ingest_job")

    with DagsterInstance.ephemeral() as instance:
        result = fewer_job.execute_in_process(
            instance=instance,
            partition_key="2023-07-14",
            asset_selection=[RAW_ASOS_HOURLY_KEY],
        )
        assert result.success

    with duckdb.connect(str(db_path)) as conn:
        rows_0714_after_fewer = conn.execute(
            "select * from raw.asos_hourly where station = 'KPHX' "
            "and cast(valid_utc as date) = '2023-07-14' order by valid_utc"
        ).fetchall()
        rows_0715_after_fewer = conn.execute(
            "select * from raw.asos_hourly where station = 'KPHX' "
            "and cast(valid_utc as date) = '2023-07-15' order by valid_utc"
        ).fetchall()

    # Re-materializing 07-14 with a fetcher returning fewer rows only
    # changes 07-14's rows; 07-15 is untouched.
    assert len(rows_0714_after_fewer) < len(rows_0714_before_0715)
    assert rows_0715_after_fewer  # 07-15 still has its rows
