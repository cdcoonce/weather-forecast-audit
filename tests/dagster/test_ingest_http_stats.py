"""The raw HTTP assets surface measured fetcher statistics in their metadata.

`request_count` stays the planned-request formula; the `http_*` keys are
additive and only appear when the fetcher exposes a `FetchStats` as `.stats`
(fixture fetchers without one are untouched).
"""

import pytest
from conftest import DagsterFixtureFetcher, FixtureIemResource, build_ingest_test_defs
from dagster import AssetKey, DagsterInstance

from weather_forecast_audit.iem.http import FetchStats

pytestmark = [pytest.mark.dagster, pytest.mark.io]

PARTITION = "2023-07-14"
HTTP_ASSETS = [
    pytest.param(AssetKey(["raw", "asos_hourly"]), "raw__asos_hourly", id="asos"),
    pytest.param(AssetKey(["raw", "nbs_guidance"]), "raw__nbs_guidance", id="nbs"),
    pytest.param(AssetKey(["raw", "cli_daily"]), "raw__cli_daily", id="cli"),
]


def _known_stats() -> FetchStats:
    return FetchStats(
        attempts=9,
        retries=2,
        timeouts=1,
        read_errors=0,
        url_errors=1,
        throttled_429=3,
        server_errors_5xx=4,
        seconds_in_attempts=123.46,
        seconds_backing_off=6.0,
        seconds_throttling=8.04,
        slowest_attempt_seconds=60.04,
        slowest_attempt_url="https://example.test/slow",
    )


class _FetcherWithStats(DagsterFixtureFetcher):
    def __init__(self) -> None:
        self.stats = _known_stats()


class _IemResourceWithStats(FixtureIemResource):
    def fetcher(self) -> _FetcherWithStats:
        return _FetcherWithStats()


def _metadata(
    tmp_path: object, iem_resource: object, key: AssetKey, node: str
) -> dict[str, object]:
    defs = build_ingest_test_defs(
        str(tmp_path / "wfa.duckdb"),  # type: ignore[operator]
        ["KPHX", "KORD"],
        iem_resource,  # type: ignore[arg-type]
    )
    job = defs.resolve_job_def("ingest_job")
    with DagsterInstance.ephemeral() as instance:
        result = job.execute_in_process(
            instance=instance, partition_key=PARTITION, asset_selection=[key]
        )
    assert result.success
    [materialization] = result.asset_materializations_for_node(node)
    return {k: v.value for k, v in materialization.metadata.items()}


@pytest.mark.parametrize(("key", "node"), HTTP_ASSETS)
def test_asset_metadata_includes_fetcher_http_stats(
    tmp_path: object, key: AssetKey, node: str
) -> None:
    metadata = _metadata(tmp_path, _IemResourceWithStats(), key, node)

    assert metadata["http_attempts"] == 9
    assert metadata["http_retries"] == 2
    assert metadata["http_timeouts"] == 1
    assert metadata["http_read_errors"] == 0
    assert metadata["http_url_errors"] == 1
    assert metadata["http_throttled_429"] == 3
    assert metadata["http_server_errors_5xx"] == 4
    assert metadata["http_seconds_in_attempts"] == 123.5
    assert metadata["http_seconds_backing_off"] == 6.0
    assert metadata["http_seconds_throttling"] == 8.0
    assert metadata["http_slowest_attempt_seconds"] == 60.0
    assert metadata["http_slowest_attempt_url"] == "https://example.test/slow"
    # The planned-request formula and station count are unchanged.
    assert metadata["station_count"] == 2
    assert metadata["request_count"] == 2


@pytest.mark.parametrize(("key", "node"), HTTP_ASSETS)
def test_asset_metadata_is_unchanged_for_a_fetcher_without_stats(
    tmp_path: object, key: AssetKey, node: str
) -> None:
    metadata = _metadata(tmp_path, FixtureIemResource(), key, node)

    assert metadata == {"station_count": 2, "request_count": 2}
