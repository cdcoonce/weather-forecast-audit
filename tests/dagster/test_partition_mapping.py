"""D1: raw/resolved_windows' upstream asos partitions (build spec #10 test 2).

A resolved partition for run date D depends on asos partitions
D-OBS_LOOKBACK_DAYS..D+OBS_LOOKAHEAD_DAYS (pipeline.OBS_LOOKBACK_DAYS=1,
OBS_LOOKAHEAD_DAYS=4): a run on D carries a lead-3 max whose window closes
D+4 06Z.
"""

import pytest
from dagster import AssetKey

from weather_forecast_audit.definitions import defs

pytestmark = pytest.mark.dagster

RESOLVED_WINDOWS_KEY = AssetKey(["raw", "resolved_windows"])
ASOS_HOURLY_KEY = AssetKey(["raw", "asos_hourly"])
NBS_GUIDANCE_KEY = AssetKey(["raw", "nbs_guidance"])


def _upstream_partition_keys(
    downstream_key: AssetKey, upstream_key: AssetKey, partition: str
) -> set[str]:
    asset_graph = defs.resolve_asset_graph()
    mapping = asset_graph.get_partition_mapping(downstream_key, upstream_key)
    downstream_def = asset_graph.get(downstream_key).partitions_def
    upstream_def = asset_graph.get(upstream_key).partitions_def
    downstream_subset = downstream_def.subset_with_partition_keys([partition])
    result = mapping.get_upstream_mapped_partitions_result_for_partitions(
        downstream_subset,
        downstream_partitions_def=downstream_def,
        upstream_partitions_def=upstream_def,
    )
    return set(result.partitions_subset.get_partition_keys())


def test_resolved_windows_2023_07_14_depends_on_asos_07_13_through_07_18() -> None:
    keys = _upstream_partition_keys(RESOLVED_WINDOWS_KEY, ASOS_HOURLY_KEY, "2023-07-14")

    assert keys == {
        "2023-07-13",
        "2023-07-14",
        "2023-07-15",
        "2023-07-16",
        "2023-07-17",
        "2023-07-18",
    }


def test_resolved_windows_depends_on_nbs_guidance_through_identity_mapping() -> None:
    keys = _upstream_partition_keys(
        RESOLVED_WINDOWS_KEY, NBS_GUIDANCE_KEY, "2023-07-14"
    )

    assert keys == {"2023-07-14"}
