"""Pure helpers for building the CONUS station registry (issue #7).

No network I/O lives here; the network scripts under `scripts/registry/`
call these functions with data they fetched and cached themselves, which is
what makes this module unit-testable offline.

## CONUS filter

CONUS = the 48 contiguous states + DC. `CONUS_STATE_CODES` is the allow-list;
anything else (AK, HI, PR, GU, VI, AS, FM, MH, or any other code the source
carries) is excluded by construction, not by an explicit deny-list.

## Climate regions

`dbt/seeds/climate_regions.csv` was NOT used; region assignment reads a
package data file, `registry_data/us_climate_regions.csv`, mapping each
2-letter state code to one of NOAA NCEI's nine U.S. climate regions (Karl,
T.R. and Koss, W.J., 1984: "Regional and National Monthly, Seasonal, and
Annual Temperature Weighted by Area, 1895-1983," Historical Climatology
Series 4-3, National Climatic Data Center), as published at
https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-climate-regions.

**Judgment call:** NCEI's own region->state arrays omit Washington DC
entirely (it has no state id in any of the nine regions on that page). DC is
mapped to Northeast in the data file because it is geographically enclosed
by Maryland, which NCEI does assign to Northeast. This is this project's own
convention, not something the source states, and it is documented again in
the evidence README under docs/spikes/2026-09-26-station-registry/.

## Coastline distance

`coastal_flag` uses the great-circle distance from a station to the nearest
*segment* (not just vertex) of the Natural Earth 1:10m coastline, via the
standard cross-track-distance construction (Ed Williams' Aviation
Formulary / the "movable-type" spherical-geometry formulas): for a segment
A->B and point P, the perpendicular ("cross-track") distance is used when
the along-track projection of P falls between A and B; otherwise the
distance to the nearer endpoint is used.
"""

import csv
from collections.abc import Callable, Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np

from weather_forecast_audit.iem.metar import parse_six_hour_groups

EARTH_RADIUS_KM = 6371.0088

CONUS_STATE_CODES = frozenset(
    {
        "AL", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT",
        "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
        "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC",
    }
)  # fmt: skip

CLIMATE_REGIONS = (
    "Northwest",
    "West",
    "Southwest",
    "Northern Rockies and Plains",
    "Upper Midwest",
    "Ohio Valley",
    "South",
    "Southeast",
    "Northeast",
)

DEFAULT_CLIMATE_REGIONS_PATH = (
    Path(__file__).resolve().parent / "registry_data" / "us_climate_regions.csv"
)


def is_conus_state(state: str) -> bool:
    """True iff `state` (a 2-letter USPS code) is one of the 48 states + DC."""
    return state in CONUS_STATE_CODES


def filter_conus_features(features: Iterable[dict]) -> list[dict]:
    """Keep only GeoJSON Features whose `properties.state` is in CONUS."""
    return [
        feature
        for feature in features
        if is_conus_state(feature["properties"]["state"])
    ]


def load_climate_regions(path: Path = DEFAULT_CLIMATE_REGIONS_PATH) -> dict[str, str]:
    """Load the state -> climate-region mapping (see module docstring)."""
    with path.open(newline="") as handle:
        return {row["state"]: row["climate_region"] for row in csv.DictReader(handle)}


def climate_region_for_state(state: str, regions: dict[str, str]) -> str:
    """Look up `state`'s climate region; raises KeyError naming the state."""
    try:
        return regions[state]
    except KeyError:
        msg = f"no climate region mapped for state {state!r}"
        raise KeyError(msg) from None


def feature_to_row(feature: dict) -> dict:
    """Map one NWSCLI GeoJSON Feature to the registry columns settled by #7.

    Produces every column except `climate_region` (needs the region lookup)
    and `coastal_flag` (needs the coastline distance, and both are supplied
    by the caller with data this module has no way to fetch itself.
    """
    props = feature["properties"]
    lon, lat = feature["geometry"]["coordinates"]
    sid = props["sid"]
    return {
        "cli_station": sid,
        "icao": sid,
        "iana_tz": props["tzname"],
        "lat": float(lat),
        "lon": float(lon),
        "elevation_m": round(float(props["elevation"])),
        "label": f"{props['sname']}, {props['state']}",
        "state": props["state"],
    }


def count_reports_with_group(metars: Iterable[str], kind: str = "max") -> int:
    """Count METAR reports carrying a 6-hour max (`kind='max'`) or min group.

    Uses the production `weather_forecast_audit.iem.metar.parse_six_hour_groups`
    parser, per build spec decision 4c, rather than re-matching the regex.
    """
    if kind not in ("max", "min"):
        msg = f"kind must be 'max' or 'min', got {kind!r}"
        raise ValueError(msg)
    count = 0
    for metar in metars:
        groups = parse_six_hour_groups(metar)
        value = groups.max_c if kind == "max" else groups.min_c
        if value is not None:
            count += 1
    return count


# -- stratified sample for the archive-start coarse sweep --------------------


def allocate_sample_sizes(
    counts_by_region: dict[str, int], total: int
) -> dict[str, int]:
    """Apportion `total` samples across regions, proportional to population.

    Largest-remainder (Hamilton) apportionment: each region gets
    floor(total * region_count / grand_total), then the leftover seats go to
    the regions with the largest fractional remainder, tie-broken by region
    name (alphabetical) for determinism. A region with zero stations gets
    zero seats.
    """
    grand_total = sum(counts_by_region.values())
    if grand_total == 0 or total <= 0:
        return dict.fromkeys(counts_by_region, 0)

    exact = {
        region: total * count / grand_total
        for region, count in counts_by_region.items()
    }
    base = {region: int(value) for region, value in exact.items()}
    # Never allocate more seats to a region than it has stations.
    base = {
        region: min(seats, counts_by_region[region]) for region, seats in base.items()
    }
    remaining = total - sum(base.values())

    candidates = [
        region
        for region in counts_by_region
        if base[region] < counts_by_region[region]
    ]
    candidates.sort(key=lambda region: (-(exact[region] - base[region]), region))
    for region in candidates:
        if remaining <= 0:
            break
        base[region] += 1
        remaining -= 1

    return base


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """`k` deterministic, evenly spaced indices into a sequence of length `n`.

    Endpoints included when k > 1. Duplicates from rounding are dropped
    (can make the result shorter than k when k is close to n).
    """
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    positions = np.linspace(0, n - 1, k)
    indices = sorted({int(round(p)) for p in positions})
    return indices


def select_stratified_sample[T](
    stations: Sequence[T],
    region_of: Callable[[T], str],
    icao_of: Callable[[T], str],
    total: int,
) -> list[T]:
    """Deterministic stratified sample of `total` stations across regions.

    Rule (build spec decision 6a): group by climate region, allocate seats
    proportional to each region's station count (largest-remainder
    apportionment, see `allocate_sample_sizes`), then within each region
    sort by ICAO and take evenly spaced indices (`evenly_spaced_indices`).
    Regions are visited in alphabetical order so the result is fully
    deterministic regardless of input order.
    """
    by_region: dict[str, list[T]] = {}
    for station in stations:
        by_region.setdefault(region_of(station), []).append(station)
    for region in by_region:
        by_region[region].sort(key=icao_of)

    counts = {region: len(members) for region, members in by_region.items()}
    seats = allocate_sample_sizes(counts, total)

    sample: list[T] = []
    for region in sorted(by_region):
        members = by_region[region]
        for index in evenly_spaced_indices(len(members), seats.get(region, 0)):
            sample.append(members[index])
    return sample


# -- archive-start search (coverage(date) -> float is injected) --------------


def sustained_crossing_date(
    dates: Sequence[date],
    coverage: Callable[[date], float],
    threshold: float,
) -> date:
    """Earliest `d` in `dates` such that coverage stays >= threshold for `d`
    and every later date in `dates`.

    A crossing that later dips back below threshold does not count: the
    search keeps advancing to the next candidate whose entire remaining
    suffix clears the bar. Raises ValueError if no such date exists (the
    curve never reaches, or never sustains, the threshold through the end
    of `dates`).
    """
    values = [coverage(d) for d in dates]
    n = len(dates)
    for start in range(n):
        if all(value >= threshold for value in values[start:]):
            return dates[start]
    msg = (
        f"coverage never reaches and sustains {threshold:.0%} "
        f"through {dates[-1].isoformat() if dates else '(empty)'}"
    )
    raise ValueError(msg)


def bisect_crossing_date(
    low: date,
    high: date,
    coverage: Callable[[date], float],
    threshold: float,
) -> date:
    """Binary search for the earliest date in [low, high] with coverage >= threshold.

    Assumes coverage is non-decreasing (or at least that `high` clears the
    threshold) over [low, high]; that assumption is exactly what the coarse
    sweep's `sustained_crossing_date` result is meant to justify for the
    bracketing month. Raises ValueError if `coverage(high) < threshold`.
    """
    if low > high:
        msg = f"low ({low.isoformat()}) must not be after high ({high.isoformat()})"
        raise ValueError(msg)
    hi_value = coverage(high)
    if hi_value < threshold:
        msg = (
            f"coverage({high.isoformat()}) = {hi_value:.4f} "
            f"never reaches {threshold:.0%}"
        )
        raise ValueError(msg)

    lo_ord, hi_ord = low.toordinal(), high.toordinal()
    while lo_ord < hi_ord:
        mid_ord = (lo_ord + hi_ord) // 2
        if coverage(date.fromordinal(mid_ord)) >= threshold:
            hi_ord = mid_ord
        else:
            lo_ord = mid_ord + 1
    return date.fromordinal(lo_ord)


def first_of_month_range(start: date, end: date) -> list[date]:
    """Every first-of-month date from `start` through `end`, inclusive.

    Both `start` and `end` must themselves be first-of-month dates.
    """
    if start.day != 1 or end.day != 1:
        msg = "first_of_month_range requires first-of-month start and end"
        raise ValueError(msg)
    months: list[date] = []
    cursor = start
    while cursor <= end:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


# -- great-circle point-to-segment coastline distance -------------------------


def _to_radians(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(np.radians(a) for a in arrays)


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1r, lon1r, lat2r, lon2r = _to_radians(lat1, lon1, lat2, lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def _initial_bearing_rad(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1r, lon1r, lat2r, lon2r = _to_radians(lat1, lon1, lat2, lon2)
    dlon = lon2r - lon1r
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.arctan2(y, x)


def point_to_segments_km(
    point_lat: float,
    point_lon: float,
    seg_a_lat: np.ndarray,
    seg_a_lon: np.ndarray,
    seg_b_lat: np.ndarray,
    seg_b_lon: np.ndarray,
) -> np.ndarray:
    """Great-circle distance (km) from one point to each of N segments.

    Segment `i` runs from (seg_a_lat[i], seg_a_lon[i]) to
    (seg_b_lat[i], seg_b_lon[i]). Vectorized over the N segments; the caller
    loops over stations. Degenerate (zero-length) segments fall back to the
    point-to-endpoint distance, which is correct since both endpoints
    coincide.
    """
    seg_a_lat = np.asarray(seg_a_lat, dtype=float)
    seg_a_lon = np.asarray(seg_a_lon, dtype=float)
    seg_b_lat = np.asarray(seg_b_lat, dtype=float)
    seg_b_lon = np.asarray(seg_b_lon, dtype=float)
    p_lat = np.full_like(seg_a_lat, point_lat)
    p_lon = np.full_like(seg_a_lon, point_lon)

    dist_to_a = _haversine_km(p_lat, p_lon, seg_a_lat, seg_a_lon)
    dist_to_b = _haversine_km(p_lat, p_lon, seg_b_lat, seg_b_lon)
    dist_endpoint = np.minimum(dist_to_a, dist_to_b)

    d13 = dist_to_a / EARTH_RADIUS_KM
    d12 = _haversine_km(seg_a_lat, seg_a_lon, seg_b_lat, seg_b_lon) / EARTH_RADIUS_KM
    brng13 = _initial_bearing_rad(seg_a_lat, seg_a_lon, p_lat, p_lon)
    brng12 = _initial_bearing_rad(seg_a_lat, seg_a_lon, seg_b_lat, seg_b_lon)

    with np.errstate(invalid="ignore"):
        dxt = np.arcsin(np.clip(np.sin(d13) * np.sin(brng13 - brng12), -1, 1))
        cos_dxt = np.cos(dxt)
        # Avoid divide-by-zero warnings for zero-length segments (cos_dxt==0
        # cannot happen there since d13 handles it, but guard anyway).
        safe_cos_dxt = np.where(cos_dxt == 0, np.nan, cos_dxt)
        dat_magnitude = np.arccos(np.clip(np.cos(d13) / safe_cos_dxt, -1, 1))
        # arccos loses the sign of the along-track distance: a point "behind"
        # A (opposite direction from B) gives the same dat_magnitude as one
        # the same distance "ahead" of A. Recover the sign from the bearing
        # difference: within 90 degrees of A->B means ahead (positive),
        # beyond 90 degrees means behind (negative).
        bearing_diff = np.arctan2(np.sin(brng13 - brng12), np.cos(brng13 - brng12))
        dat = np.where(np.abs(bearing_diff) <= np.pi / 2, dat_magnitude, -dat_magnitude)

    on_segment = (d12 > 0) & (dat >= 0) & (dat <= d12) & np.isfinite(dat)
    dist_xt = np.abs(dxt) * EARTH_RADIUS_KM
    return np.where(on_segment, dist_xt, dist_endpoint)


def min_distance_to_coastline_km(
    lat: float,
    lon: float,
    seg_a_lat: np.ndarray,
    seg_a_lon: np.ndarray,
    seg_b_lat: np.ndarray,
    seg_b_lon: np.ndarray,
) -> float:
    """Minimum great-circle distance (km) from (lat, lon) to any segment."""
    distances = point_to_segments_km(
        lat, lon, seg_a_lat, seg_a_lon, seg_b_lat, seg_b_lon
    )
    return float(np.min(distances))


def linestrings_to_segments(
    linestrings: Iterable[list[list[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten GeoJSON LineString coordinate lists into segment endpoint arrays.

    Each LineString's coordinates are `[[lon, lat], [lon, lat], ...]`
    (GeoJSON order). Returns (a_lat, a_lon, b_lat, b_lon) arrays, one entry
    per consecutive-vertex segment, across all input LineStrings.
    """
    a_lat: list[float] = []
    a_lon: list[float] = []
    b_lat: list[float] = []
    b_lon: list[float] = []
    for coords in linestrings:
        for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:], strict=False):
            a_lat.append(lat1)
            a_lon.append(lon1)
            b_lat.append(lat2)
            b_lon.append(lon2)
    return (
        np.array(a_lat, dtype=float),
        np.array(a_lon, dtype=float),
        np.array(b_lat, dtype=float),
        np.array(b_lon, dtype=float),
    )


def filter_segments_to_bbox(
    a_lat: np.ndarray,
    a_lon: np.ndarray,
    b_lat: np.ndarray,
    b_lon: np.ndarray,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only segments with at least one endpoint inside the padded bbox."""

    def _inside(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return (
            (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        )

    keep = _inside(a_lat, a_lon) | _inside(b_lat, b_lon)
    return a_lat[keep], a_lon[keep], b_lat[keep], b_lon[keep]
