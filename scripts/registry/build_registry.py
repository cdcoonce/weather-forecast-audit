#!/usr/bin/env python3
"""Build the CONUS station registry + exclusions seed (issue #7).

Sources (see the evidence README for full citations):
- Station list: IEM's NWSCLI GeoJSON network file.
- Coastal flag: the Natural Earth 1:10m coastline, pinned to the v5.1.2 tag.
- Inclusion probes: IEM's bulk `/api/1/mos.json` (NBS presence) and
  `asos.py` (ASOS presence + METAR 6-hour group presence) endpoints.

Every HTTP response is cached under `.cache/registry/` by URL hash
(`scripts/registry/_cache.py`); `--offline` replays cached responses only
and fails loudly on a cache miss, so a rerun with `--offline` reproduces the
seeds and evidence byte-identically with zero new requests.

Writes:
- dbt/seeds/station_registry.csv
- dbt/seeds/station_exclusions.csv
- docs/spikes/2026-09-26-station-registry/probe_results.csv
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _cache import DEFAULT_CACHE_DIR, CachingFetcher  # noqa: E402

from weather_forecast_audit import registry_sources as rs  # noqa: E402
from weather_forecast_audit.iem.http import UrllibFetcher  # noqa: E402
from weather_forecast_audit.iem.metar import parse_six_hour_groups  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
SEED_REGISTRY = REPO_ROOT / "dbt" / "seeds" / "station_registry.csv"
SEED_EXCLUSIONS = REPO_ROOT / "dbt" / "seeds" / "station_exclusions.csv"
EVIDENCE_DIR = REPO_ROOT / "docs" / "spikes" / "2026-09-26-station-registry"
PROBE_RESULTS_CSV = EVIDENCE_DIR / "probe_results.csv"

NWSCLI_URL = "https://mesonet.agron.iastate.edu/geojson/network/NWSCLI.geojson"
COASTLINE_URL = (
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector"
    "@v5.1.2/geojson/ne_10m_coastline.geojson"
)
MOS_API_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Settled decisions (build spec #7, decision 4): fixed probe runtimes/days.
NBS_PROBE_RUNTIMES = ("2025-06-01T13:00Z", "2025-12-01T13:00Z")
ASOS_PROBE_DAYS = (date(2025, 3, 1), date(2025, 6, 1), date(2025, 9, 1))
# Padded well beyond the CONUS bbox (lat 24-49.5, lon -125 to -66.5) so no
# real coastal station's nearest segment is cut off by the prefilter.
BBOX = {"min_lat": 20.0, "max_lat": 53.0, "min_lon": -128.0, "max_lon": -64.0}

MOS_BATCH = 6
ASOS_BATCH = 50

GREAT_LAKES_CHECK_STATIONS = ("KBUF", "KCLE", "KMKE")
COASTAL_SPOT_CHECKS = {
    "KSFO": True,
    "KBOS": True,
    "KMIA": True,
    "KSEA": True,
    "KPHX": False,
    "KDEN": False,
    "KORD": False,
}


def _asos_code(icao: str) -> str:
    return icao[1:] if len(icao) == 4 and icao.startswith("K") else icao


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_optional_float(value: str) -> float | None:
    return None if value in ("", "M", None) else float(value)


@dataclass
class ProbeState:
    icao: str
    nbs_txn_present: dict[str, bool] = field(default_factory=dict)  # runtime -> bool
    nbm_station_id: str | None = None
    asos_days_with_data: set[date] = field(default_factory=set)
    asos_days_with_2plus_groups: set[date] = field(default_factory=set)
    max_group_counts: dict[date, int] = field(default_factory=dict)


def fetch_nwscli_features(fetcher: CachingFetcher) -> list[dict]:
    response = fetcher.get(NWSCLI_URL)
    payload = json.loads(response.body)
    return payload["features"]


def fetch_coastline_segments(
    fetcher: CachingFetcher,
) -> tuple[object, object, object, object]:
    response = fetcher.get(COASTLINE_URL)
    payload = json.loads(response.body)
    linestrings: list[list[list[float]]] = []
    for feature in payload["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "LineString":
            linestrings.append(geometry["coordinates"])
        elif geometry["type"] == "MultiLineString":
            linestrings.extend(geometry["coordinates"])
    a_lat, a_lon, b_lat, b_lon = rs.linestrings_to_segments(linestrings)
    return rs.filter_segments_to_bbox(a_lat, a_lon, b_lat, b_lon, **BBOX)


def run_nbs_probe(
    fetcher: CachingFetcher, icaos: list[str], states: dict[str, ProbeState]
) -> None:
    for batch in _chunks(icaos, MOS_BATCH):
        for runtime in NBS_PROBE_RUNTIMES:
            query = "&".join(f"station={icao}" for icao in batch)
            url = f"{MOS_API_URL}?{query}&model=NBS&runtime={runtime}"
            response = fetcher.get(url)
            data = json.loads(response.body)["data"]
            seen_this_runtime: set[str] = set()
            for row in data:
                station = row["station"]
                if station not in states:
                    continue
                seen_this_runtime.add(station)
                has_txn = row.get("txn") is not None
                states[station].nbs_txn_present[runtime] = (
                    states[station].nbs_txn_present.get(runtime, False) or has_txn
                )
                if states[station].nbm_station_id is None:
                    states[station].nbm_station_id = station
            for icao in batch:
                states[icao].nbs_txn_present.setdefault(runtime, False)


def run_asos_probe(
    fetcher: CachingFetcher, icaos: list[str], states: dict[str, ProbeState]
) -> None:
    code_to_icao = {_asos_code(icao): icao for icao in icaos}
    for batch in _chunks(icaos, ASOS_BATCH):
        codes = [_asos_code(icao) for icao in batch]
        for day in ASOS_PROBE_DAYS:
            sts = f"{day.isoformat()}T00:00Z"
            ets = f"{(day + timedelta(days=1)).isoformat()}T00:00Z"
            query = "&".join(f"station={code}" for code in codes)
            url = (
                f"{ASOS_URL}?{query}&data=tmpf&data=metar&sts={sts}&ets={ets}"
                "&tz=Etc/UTC&format=onlycomma&missing=M"
                "&report_type=3&report_type=4&latlon=no"
            )
            response = fetcher.get(url)
            reader = csv.DictReader(response.body.decode("utf-8").splitlines())
            for row in reader:
                icao = code_to_icao.get(row["station"])
                if icao is None:
                    continue
                state = states[icao]
                tmpf = _parse_optional_float(row["tmpf"])
                if tmpf is not None:
                    state.asos_days_with_data.add(day)
                groups = parse_six_hour_groups(row["metar"])
                if groups.max_c is not None:
                    state.max_group_counts[day] = state.max_group_counts.get(day, 0) + 1

    for icao in icaos:
        state = states[icao]
        for day in ASOS_PROBE_DAYS:
            if state.max_group_counts.get(day, 0) >= 2:
                state.asos_days_with_2plus_groups.add(day)


def classify(state: ProbeState) -> tuple[str | None, str]:
    """Return (reason, evidence) for exclusion, or (None, evidence) if included.

    Reasons are checked in the order the build spec lists them (decision 4);
    only the first failing reason is recorded.
    """
    nbs_hit_runtimes = [rt for rt, ok in state.nbs_txn_present.items() if ok]
    if not nbs_hit_runtimes:
        evidence = f"non_null_txn_runtimes=0/{len(NBS_PROBE_RUNTIMES)}"
        return "no_nbs_guidance", evidence

    days_with_data = len(state.asos_days_with_data)
    if days_with_data < 2:
        evidence = f"days_with_tmpf={days_with_data}/{len(ASOS_PROBE_DAYS)}"
        return "no_asos_observations", evidence

    days_with_groups = len(state.asos_days_with_2plus_groups)
    if days_with_groups < 2:
        evidence = f"days_with_2plus_groups={days_with_groups}/{len(ASOS_PROBE_DAYS)}"
        return "no_six_hour_groups", evidence

    evidence = (
        f"non_null_txn_runtimes={len(nbs_hit_runtimes)}/{len(NBS_PROBE_RUNTIMES)}; "
        f"days_with_tmpf={days_with_data}/{len(ASOS_PROBE_DAYS)}; "
        f"days_with_2plus_groups={days_with_groups}/{len(ASOS_PROBE_DAYS)}"
    )
    return None, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    inner = UrllibFetcher()
    fetcher = CachingFetcher(inner, cache_dir=DEFAULT_CACHE_DIR, offline=args.offline)

    features = fetch_nwscli_features(fetcher)
    total_listed = len(features)
    conus_features = rs.filter_conus_features(features)
    conus_count = len(conus_features)

    regions = rs.load_climate_regions()
    rows: dict[str, dict] = {}
    for feature in conus_features:
        row = rs.feature_to_row(feature)
        row["climate_region"] = rs.climate_region_for_state(row["state"], regions)
        rows[row["icao"]] = row

    icaos = sorted(rows)

    a_lat, a_lon, b_lat, b_lon = fetch_coastline_segments(fetcher)
    for row in rows.values():
        distance_km = rs.min_distance_to_coastline_km(
            row["lat"], row["lon"], a_lat, a_lon, b_lat, b_lon
        )
        row["coastal_distance_km"] = distance_km
        row["coastal_flag"] = distance_km <= rs.COASTAL_THRESHOLD_KM

    great_lakes_violations = [
        icao
        for icao in GREAT_LAKES_CHECK_STATIONS
        if icao in rows and rows[icao]["coastal_flag"]
    ]
    if great_lakes_violations:
        details = ", ".join(
            f"{icao}={rows[icao]['coastal_distance_km']:.2f}km"
            for icao in great_lakes_violations
        )
        print(
            "STOP: the Natural Earth coastline file appears to include Great "
            f"Lakes shoreline; these stations came out coastal=true: {details}. "
            "Build spec decision 3 requires them false. Halting before writing "
            "seeds; investigate the coastline source before rerunning.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    spot_check_mismatches = [
        f"{icao} expected={expected} got={rows[icao]['coastal_flag']}"
        for icao, expected in COASTAL_SPOT_CHECKS.items()
        if icao in rows and rows[icao]["coastal_flag"] != expected
    ]
    if spot_check_mismatches:
        print(
            "WARNING: coastal_flag spot-check mismatch(es): "
            + "; ".join(spot_check_mismatches),
            file=sys.stderr,
        )

    states: dict[str, ProbeState] = {icao: ProbeState(icao=icao) for icao in icaos}
    run_nbs_probe(fetcher, icaos, states)
    run_asos_probe(fetcher, icaos, states)

    included_rows: list[dict] = []
    excluded_rows: list[dict] = []
    probe_result_rows: list[dict] = []

    for icao in icaos:
        row = rows[icao]
        state = states[icao]
        reason, evidence = classify(state)
        probe_result_rows.append(
            {
                "icao": icao,
                "state": row["state"],
                "climate_region": row["climate_region"],
                "coastal_flag": row["coastal_flag"],
                "coastal_distance_km": f"{row['coastal_distance_km']:.3f}",
                "reason": reason or "",
                "evidence": evidence,
            }
        )
        if reason is not None:
            excluded_rows.append(
                {
                    "cli_station": row["cli_station"],
                    "icao": icao,
                    "label": row["label"],
                    "reason": reason,
                    "evidence": evidence,
                }
            )
            continue
        included_rows.append(
            {
                "cli_station": row["cli_station"],
                "icao": icao,
                "nbm_station_id": state.nbm_station_id or icao,
                "iana_tz": row["iana_tz"],
                "lat": row["lat"],
                "lon": row["lon"],
                "elevation_m": row["elevation_m"],
                "label": row["label"],
                "climate_region": row["climate_region"],
                "coastal_flag": "true" if row["coastal_flag"] else "false",
            }
        )

    included_rows.sort(key=lambda r: r["icao"])
    excluded_rows.sort(key=lambda r: r["icao"])
    probe_result_rows.sort(key=lambda r: r["icao"])

    with SEED_REGISTRY.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cli_station", "icao", "nbm_station_id", "iana_tz", "lat", "lon",
                "elevation_m", "label", "climate_region", "coastal_flag",
            ],  # fmt: skip
        )
        writer.writeheader()
        writer.writerows(included_rows)

    with SEED_EXCLUSIONS.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["cli_station", "icao", "label", "reason", "evidence"]
        )
        writer.writeheader()
        writer.writerows(excluded_rows)

    with PROBE_RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "icao", "state", "climate_region", "coastal_flag",
                "coastal_distance_km", "reason", "evidence",
            ],  # fmt: skip
        )
        writer.writeheader()
        writer.writerows(probe_result_rows)

    by_reason: dict[str, int] = {}
    for row in excluded_rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1

    print(f"listed={total_listed} conus={conus_count} included={len(included_rows)}")
    print(f"excluded_by_reason={by_reason}")
    print(f"live requests made this run={fetcher.requests_made}")


if __name__ == "__main__":
    main()
