"""Pure registry-building helpers: CONUS filter, region lookup, coastline
distance, synoptic-group presence, and the archive-start search (issue #7).
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pytest

from weather_forecast_audit.registry_sources import (
    allocate_sample_sizes,
    bisect_crossing_date,
    climate_region_for_state,
    count_reports_with_group,
    evenly_spaced_indices,
    feature_to_row,
    filter_conus_features,
    filter_segments_to_bbox,
    first_of_month_range,
    is_conus_state,
    linestrings_to_segments,
    load_climate_regions,
    min_distance_to_coastline_km,
    select_stratified_sample,
    sustained_crossing_date,
)

pytestmark = pytest.mark.unit


# -- CONUS filter --------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("AZ", True),
        ("NY", True),
        ("DC", True),
        ("AK", False),
        ("HI", False),
        ("PR", False),
        ("GU", False),
        ("VI", False),
        ("AS", False),
        ("FM", False),
        ("MH", False),
    ],
)
def test_is_conus_state(state: str, expected: bool) -> None:
    assert is_conus_state(state) is expected


def _feature(state: str, sid: str = "KABC") -> dict:
    return {
        "type": "Feature",
        "properties": {
            "sid": sid,
            "sname": "SOMEWHERE",
            "state": state,
            "tzname": "America/Chicago",
            "elevation": 100.4,
        },
        "geometry": {"type": "Point", "coordinates": [-90.0, 40.0]},
    }


def test_filter_conus_features_drops_non_conus_keeps_dc() -> None:
    features = [_feature("AK"), _feature("HI"), _feature("DC"), _feature("TX")]
    kept = filter_conus_features(features)
    assert [f["properties"]["state"] for f in kept] == ["DC", "TX"]


def test_feature_to_row_maps_settled_columns() -> None:
    feature = _feature("TX", sid="KABC")
    row = feature_to_row(feature)
    assert row == {
        "cli_station": "KABC",
        "icao": "KABC",
        "iana_tz": "America/Chicago",
        "lat": 40.0,
        "lon": -90.0,
        "elevation_m": 100,
        "label": "SOMEWHERE, TX",
        "state": "TX",
    }


# -- climate regions -------------------------------------------------------


def test_load_climate_regions_covers_all_conus_states_including_dc() -> None:
    regions = load_climate_regions()
    from weather_forecast_audit.registry_sources import CONUS_STATE_CODES

    assert set(regions) == CONUS_STATE_CODES


def test_climate_region_for_state_dc_is_northeast() -> None:
    regions = load_climate_regions()
    assert climate_region_for_state("DC", regions) == "Northeast"


def test_climate_region_for_state_unknown_raises() -> None:
    with pytest.raises(KeyError):
        climate_region_for_state("ZZ", {"TX": "South"})


# -- synoptic group presence ------------------------------------------------


def test_count_reports_with_group_counts_only_matching_kind() -> None:
    metars = [
        "METAR KPHX 141751Z RMK 10123 21456",  # has both max and min
        "METAR KPHX 142351Z RMK AO2",  # neither
        "METAR KPHX 140551Z RMK 10089",  # max only
    ]
    assert count_reports_with_group(metars, kind="max") == 2
    assert count_reports_with_group(metars, kind="min") == 1


def test_count_reports_with_group_invalid_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        count_reports_with_group([], kind="bogus")


# -- stratified sample -------------------------------------------------------


def test_allocate_sample_sizes_sums_to_total_and_is_proportional() -> None:
    counts = {"South": 100, "Northeast": 50, "West": 10}
    seats = allocate_sample_sizes(counts, total=60)
    assert sum(seats.values()) == 60
    # South has 2x Northeast's stations, so should get roughly 2x the seats.
    assert seats["South"] >= seats["Northeast"] >= seats["West"]


def test_allocate_sample_sizes_never_exceeds_region_population() -> None:
    counts = {"Tiny": 2, "Big": 200}
    seats = allocate_sample_sizes(counts, total=60)
    assert seats["Tiny"] <= 2
    assert sum(seats.values()) == 60


def test_evenly_spaced_indices_includes_endpoints() -> None:
    indices = evenly_spaced_indices(10, 5)
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == 5


def test_evenly_spaced_indices_k_gte_n_returns_all() -> None:
    assert evenly_spaced_indices(3, 5) == [0, 1, 2]


@dataclass(frozen=True)
class _FakeStation:
    icao: str
    region: str


def test_select_stratified_sample_is_deterministic_and_sorted_by_icao() -> None:
    stations = [
        _FakeStation("KZZZ", "South"),
        _FakeStation("KAAA", "South"),
        _FakeStation("KBBB", "Northeast"),
        _FakeStation("KCCC", "Northeast"),
    ]
    sample1 = select_stratified_sample(
        stations, lambda s: s.region, lambda s: s.icao, total=2
    )
    sample2 = select_stratified_sample(
        stations, lambda s: s.region, lambda s: s.icao, total=2
    )
    assert sample1 == sample2
    assert len(sample1) == 2


# -- archive-start search ----------------------------------------------------


def test_sustained_crossing_date_monotone() -> None:
    dates = [date(2020, 1, i) for i in range(1, 6)]
    values = dict(zip(dates, [0.5, 0.8, 0.95, 0.97, 0.99], strict=True))
    result = sustained_crossing_date(dates, lambda d: values[d], threshold=0.95)
    assert result == date(2020, 1, 3)


def test_sustained_crossing_date_dip_pushes_date_later() -> None:
    dates = [date(2020, 1, i) for i in range(1, 6)]
    # Crosses 0.95 on day 2, dips back on day 3, recovers for good on day 4.
    values = dict(zip(dates, [0.5, 0.96, 0.80, 0.97, 0.98], strict=True))
    result = sustained_crossing_date(dates, lambda d: values[d], threshold=0.95)
    assert result == date(2020, 1, 4)


def test_sustained_crossing_date_never_reaches_raises() -> None:
    dates = [date(2020, 1, i) for i in range(1, 4)]
    values = dict.fromkeys(dates, 0.5)
    with pytest.raises(ValueError, match="never"):
        sustained_crossing_date(dates, lambda d: values[d], threshold=0.95)


def test_bisect_crossing_date_finds_earliest_day() -> None:
    low, high = date(2020, 1, 1), date(2020, 1, 31)

    def coverage(d: date) -> float:
        return 0.99 if d >= date(2020, 1, 17) else 0.10

    result = bisect_crossing_date(low, high, coverage, threshold=0.95)
    assert result == date(2020, 1, 17)


def test_bisect_crossing_date_high_never_reaching_raises() -> None:
    with pytest.raises(ValueError, match="never reaches"):
        bisect_crossing_date(
            date(2020, 1, 1), date(2020, 1, 31), lambda d: 0.1, threshold=0.95
        )


def test_first_of_month_range() -> None:
    months = first_of_month_range(date(2020, 11, 1), date(2021, 2, 1))
    assert months == [
        date(2020, 11, 1),
        date(2020, 12, 1),
        date(2021, 1, 1),
        date(2021, 2, 1),
    ]


def test_first_of_month_range_requires_first_of_month() -> None:
    with pytest.raises(ValueError, match="first-of-month"):
        first_of_month_range(date(2020, 11, 5), date(2021, 2, 1))


# -- coastline distance ------------------------------------------------------


def _segments(
    pairs: list[tuple[float, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_lat = np.array([p[0] for p in pairs])
    a_lon = np.array([p[1] for p in pairs])
    b_lat = np.array([p[2] for p in pairs])
    b_lon = np.array([p[3] for p in pairs])
    return a_lat, a_lon, b_lat, b_lon


def test_point_on_segment_is_zero_distance() -> None:
    # A horizontal segment along 40N, from -100 to -99 lon; the point sits
    # exactly at its midpoint.
    a_lat, a_lon, b_lat, b_lon = _segments([(40.0, -100.0, 40.0, -99.0)])
    dist = min_distance_to_coastline_km(40.0, -99.5, a_lat, a_lon, b_lat, b_lon)
    assert dist == pytest.approx(0.0, abs=1.0)


def test_perpendicular_distance_to_segment_interior() -> None:
    # Same segment; point offset ~0.1 deg north of the midpoint interior.
    # 0.1 deg latitude is about 11.1 km.
    a_lat, a_lon, b_lat, b_lon = _segments([(40.0, -100.0, 40.0, -99.0)])
    dist = min_distance_to_coastline_km(40.1, -99.5, a_lat, a_lon, b_lat, b_lon)
    assert dist == pytest.approx(11.1, rel=0.05)


def test_endpoint_distance_when_closest_point_is_off_segment() -> None:
    # Point due north of segment's start endpoint, well past its extent.
    a_lat, a_lon, b_lat, b_lon = _segments([(40.0, -100.0, 40.0, -99.0)])
    dist_to_endpoint = min_distance_to_coastline_km(
        40.1, -101.0, a_lat, a_lon, b_lat, b_lon
    )
    # Should be close to direct distance to (40.0, -100.0), not the
    # perpendicular cross-track distance (which would assume an on-segment
    # closest point).
    a_lat2, a_lon2, b_lat2, b_lon2 = _segments([(40.0, -100.0, 40.0, -100.0)])
    direct = min_distance_to_coastline_km(40.1, -101.0, a_lat2, a_lon2, b_lat2, b_lon2)
    assert dist_to_endpoint == pytest.approx(direct, rel=0.01)


def test_min_distance_picks_nearest_of_several_segments() -> None:
    a_lat, a_lon, b_lat, b_lon = _segments(
        [
            (40.0, -100.0, 40.0, -99.0),  # far
            (10.0, -99.5, 10.0, -99.4),  # near-ish but still far
            (40.0, -99.51, 40.0, -99.49),  # essentially at the query point
        ]
    )
    dist = min_distance_to_coastline_km(40.0, -99.5, a_lat, a_lon, b_lat, b_lon)
    assert dist == pytest.approx(0.0, abs=1.0)


def test_linestrings_to_segments_flattens_consecutive_vertices() -> None:
    linestrings = [[[-100.0, 40.0], [-99.0, 40.0], [-98.0, 41.0]], [[0.0, 0.0]]]
    a_lat, a_lon, b_lat, b_lon = linestrings_to_segments(linestrings)
    assert len(a_lat) == 2  # 2 segments from the 3-point line; single point yields none
    assert a_lat.tolist() == [40.0, 40.0]
    assert b_lat.tolist() == [40.0, 41.0]


def test_filter_segments_to_bbox_keeps_only_overlapping() -> None:
    a_lat, a_lon, b_lat, b_lon = _segments(
        [
            (40.0, -100.0, 40.0, -99.0),  # inside
            (60.0, -150.0, 61.0, -151.0),  # outside (Alaska-ish)
        ]
    )
    fa_lat, fa_lon, fb_lat, fb_lon = filter_segments_to_bbox(
        a_lat, a_lon, b_lat, b_lon, min_lat=20, max_lat=55, min_lon=-130, max_lon=-60
    )
    assert len(fa_lat) == 1
    assert fa_lat[0] == 40.0
