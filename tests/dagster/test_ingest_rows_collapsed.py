"""The ingest assets report how many duplicate source rows their loaders dropped.

`rows_collapsed` is the number of input records a loader folded into another
record of the same key (the source served a run twice, or sent a second,
valueless report at the same valid time). It is metadata, not a failure: a
real non-null disagreement raises in the loader instead.
"""

import duckdb
import pytest
from conftest import (
    DagsterFixtureFetcher,
    FixtureIemResource,
    build_ingest_test_defs,
)
from dagster import AssetKey, DagsterInstance

from weather_forecast_audit.iem.http import HttpResponse

pytestmark = [pytest.mark.dagster, pytest.mark.io]

PARTITION = "2023-07-14"
GUIDANCE_KEY = AssetKey(["raw", "nbs_guidance"])
HOURLY_KEY = AssetKey(["raw", "asos_hourly"])
RESOLVED_KEY = AssetKey(["raw", "resolved_windows"])


def _doubled_data_rows(body: bytes) -> bytes:
    """`body` (a CSV) with every data row served twice, header once."""
    header, *rows = body.decode().splitlines()
    return "\n".join([header, *(row for row in rows for _ in range(2))]).encode()


class _DoublingFetcher(DagsterFixtureFetcher):
    """The fixture fetcher, but IEM serves every NBS and ASOS row twice."""

    def get(self, url: str) -> HttpResponse:
        response = super().get(url)
        if "mos.py" in url or "asos.py" in url:
            return HttpResponse(response.status, _doubled_data_rows(response.body))
        return response


class _DoublingIemResource(FixtureIemResource):
    def fetcher(self) -> _DoublingFetcher:
        return _DoublingFetcher()


def _materialize(
    db_path: str, resource: FixtureIemResource, key: AssetKey
) -> dict[str, object]:
    defs = build_ingest_test_defs(db_path, ["KPHX", "KORD"], resource)
    job = defs.resolve_job_def("ingest_job")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance, partition_key=PARTITION, asset_selection=[key]
        )
    assert result.success
    node = "__".join(key.path)
    [materialization] = result.asset_materializations_for_node(node)
    return {k: v.value for k, v in materialization.metadata.items()}


def _count(db_path: str, table: str) -> int:
    with duckdb.connect(db_path) as conn:
        row = conn.execute(f"select count(*) from {table}").fetchone()  # noqa: S608
    assert row is not None
    return int(row[0])


@pytest.mark.parametrize(
    ("key", "table"),
    [
        pytest.param(GUIDANCE_KEY, "raw.nbs_guidance", id="guidance"),
        pytest.param(HOURLY_KEY, "raw.asos_hourly", id="hourly"),
    ],
)
def test_ingest_asset_reports_the_duplicate_rows_it_collapsed(
    tmp_path: object, key: AssetKey, table: str
) -> None:
    db_path = str(tmp_path / "wfa.duckdb")  # type: ignore[operator]

    metadata = _materialize(db_path, _DoublingIemResource(), key)

    stored = _count(db_path, table)
    assert stored > 0
    # Every source row was served twice, so exactly one copy of each is dropped.
    assert metadata["rows_collapsed"] == stored


def test_resolved_windows_asset_reports_the_duplicates_it_collapsed(
    tmp_path: object,
) -> None:
    db_path = str(tmp_path / "wfa.duckdb")  # type: ignore[operator]
    _materialize(db_path, FixtureIemResource(), GUIDANCE_KEY)
    # A database loaded before the guidance loader collapsed duplicates.
    with duckdb.connect(db_path) as conn:
        conn.execute(
            "insert into raw.nbs_guidance select * from raw.nbs_guidance "
            "where txn is not null"
        )
        duplicated = conn.execute(
            "select count(*) from raw.nbs_guidance where txn is not null"
        ).fetchone()
    assert duplicated is not None
    assert duplicated[0] > 0

    metadata = _materialize(db_path, FixtureIemResource(), RESOLVED_KEY)

    assert metadata["rows_collapsed"] == duplicated[0] // 2
    assert _count(db_path, "raw.resolved_windows") == duplicated[0] // 2
