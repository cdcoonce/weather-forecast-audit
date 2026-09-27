#!/usr/bin/env python3
"""Pin the NBS archive start date (issue #7, decision 6).

Method (build spec decision 6):
  a. Coarse: first-of-month coverage share, 2018-11-01 through 2026-09-01,
     on a deterministic stratified sample of 60 registry stations.
  b. Find the earliest month M after which the sample share stays >= 95%
     through the end of the coarse range.
  c. Fine: binary search the full registry at daily resolution between
     M-1 month and M; confirm full-registry share >= 95% at the pinned date
     and at 6 later spot dates (June 1, 2021-2026).

"Coverage" at date D = the share of stations with a non-null `txn` in the
canonical archived NBS run for D. Three cycle regimes are in play (build
spec decision 5's fix, plus the pre-existing, unchanged 2026-04-30
changeover), none of which depend on the archive-start date pinned here:
00/07/12/19 UTC (canonical 12Z) through 2020-02-25, 01/07/13/19 UTC
(canonical 13Z) 2020-02-26 through 2026-04-29, then back to a 12Z-canonical
regime from 2026-04-30 onward.

Requires `dbt/seeds/station_registry.csv` to already exist (run
`build_registry.py` first). Every HTTP response is cached under
`.cache/registry/`; `--offline` replays cached responses only.

Writes:
- dbt/seeds/nbs_archive.csv
- docs/spikes/2026-09-26-station-registry/coarse_coverage_curve.csv
- docs/spikes/2026-09-26-station-registry/fine_search_log.csv
- docs/spikes/2026-09-26-station-registry/README.md
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _cache import DEFAULT_CACHE_DIR, CachingFetcher  # noqa: E402

from weather_forecast_audit import registry_sources as rs  # noqa: E402
from weather_forecast_audit.iem.http import FetchError, UrllibFetcher  # noqa: E402
from weather_forecast_audit.registry import Station, load_registry  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[1]
SEED_REGISTRY = REPO_ROOT / "dbt" / "seeds" / "station_registry.csv"
SEED_EXCLUSIONS = REPO_ROOT / "dbt" / "seeds" / "station_exclusions.csv"
SEED_ARCHIVE = REPO_ROOT / "dbt" / "seeds" / "nbs_archive.csv"
EVIDENCE_DIR = REPO_ROOT / "docs" / "spikes" / "2026-09-26-station-registry"
COARSE_CSV = EVIDENCE_DIR / "coarse_coverage_curve.csv"
FINE_LOG_CSV = EVIDENCE_DIR / "fine_search_log.csv"
PROBE_RESULTS_CSV = EVIDENCE_DIR / "probe_results.csv"
README = EVIDENCE_DIR / "README.md"

MOS_API_URL = "https://mesonet.agron.iastate.edu/api/1/mos.json"
MOS_BATCH = 6

THRESHOLD = 0.95
SAMPLE_SIZE = 60
COARSE_START = date(2018, 11, 1)
COARSE_END = date(2026, 9, 1)
# Probed cycle-regime facts (build spec decision 5, plus the pre-existing,
# unchanged 2026-04-30 changeover from `nbs_cycle_regimes.csv`): 00/07/12/19
# UTC (canonical 12Z) through 2020-02-25 inclusive; 01/07/13/19 UTC
# (canonical 13Z) 2020-02-26 through 2026-04-29; 2026-04-30 onward reverts
# to a 12Z-canonical regime. None of these three boundary dates depend on
# the archive-start date this script is pinning.
FIRST_REGIME_END = date(2020, 2, 25)
SECOND_REGIME_END = date(2026, 4, 29)
CONFIRM_DATES = [date(year, 6, 1) for year in range(2021, 2027)]
# Bounds how many extra coarse months the fine search's fallback will widen
# into if the crossing month's full-registry coverage doesn't confirm the
# sample's crossing month outright. A handful of months is enough slack for
# sample-vs-full-registry noise near the true crossing; an unbounded widening
# loop is exactly what turned one bad request into an unbounded request
# storm (orchestrator redirect, issue #7).
MAX_FINE_SEARCH_MONTH_EXPANSIONS = 3

NWSCLI_CITATION = (
    "https://mesonet.agron.iastate.edu/geojson/network/NWSCLI.geojson"
)
COASTLINE_CITATION = (
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector"
    "@v5.1.2/geojson/ne_10m_coastline.geojson (pinned tag v5.1.2)"
)
CLIMATE_REGION_CITATION = (
    "NOAA NCEI's nine U.S. climate regions (Karl, T.R. and Koss, W.J., 1984: "
    '"Regional and National Monthly, Seasonal, and Annual Temperature '
    'Weighted by Area, 1895-1983," Historical Climatology Series 4-3, '
    "National Climatic Data Center), "
    "https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-climate-regions"
)


def canonical_hour_for_probe(d: date) -> int:
    """The archived cycle nearest 12Z, per the probed cycle-regime facts."""
    if d <= FIRST_REGIME_END:
        return 12
    if d <= SECOND_REGIME_END:
        return 13
    return 12


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class CoverageProber:
    """Coverage(date) backed by the bulk `/api/1/mos.json` NBS presence check.

    Caches per-(runtime, station) hits so the coarse sweep's sample and the
    fine search's full registry never re-query a station+runtime pair the
    other phase already fetched.
    """

    def __init__(self, fetcher: CachingFetcher) -> None:
        self.fetcher = fetcher
        self._hits: dict[tuple[str, str], bool] = {}
        self.call_log: list[dict] = []

    def _ensure(self, stations: list[str], runtime: str) -> None:
        missing = [s for s in stations if (runtime, s) not in self._hits]
        for batch in _chunks(missing, MOS_BATCH):
            query = "&".join(f"station={station}" for station in batch)
            url = f"{MOS_API_URL}?{query}&model=NBS&runtime={runtime}"
            try:
                response = self.fetcher.get(url)
            except FetchError as exc:
                # The API 404s (rather than 200-with-empty-data) when NONE of
                # the requested stations have any archived data at all for
                # this runtime -- the expected shape of "not yet archived"
                # for early dates. Any other status is a real failure.
                if exc.status != 404:
                    raise
                for station in batch:
                    self._hits[(runtime, station)] = False
                continue
            data = json.loads(response.body)["data"]
            present: set[str] = {
                row["station"] for row in data if row.get("txn") is not None
            }
            for station in batch:
                self._hits[(runtime, station)] = station in present

    def coverage(self, stations: list[str], d: date, *, phase: str) -> float:
        hour = canonical_hour_for_probe(d)
        runtime = f"{d.isoformat()}T{hour:02d}:00Z"
        self._ensure(stations, runtime)
        hits = sum(1 for station in stations if self._hits[(runtime, station)])
        share = hits / len(stations)
        self.call_log.append(
            {
                "phase": phase,
                "date": d.isoformat(),
                "runtime": runtime,
                "sample_size": len(stations),
                "hits": hits,
                "share": f"{share:.4f}",
            }
        )
        return share


def build_sample(stations: list[Station], total: int) -> list[Station]:
    return rs.select_stratified_sample(
        stations,
        region_of=lambda s: s.climate_region,
        icao_of=lambda s: s.icao,
        total=total,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stop(message: str, coarse_rows: list[dict]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(COARSE_CSV, ["date", "share", "sample_size"], coarse_rows)
    print(f"STOP: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if not SEED_REGISTRY.exists():
        msg = f"{SEED_REGISTRY} does not exist; run build_registry.py first"
        raise SystemExit(msg)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    inner = UrllibFetcher()
    fetcher = CachingFetcher(inner, cache_dir=DEFAULT_CACHE_DIR, offline=args.offline)
    prober = CoverageProber(fetcher)

    registry = load_registry(SEED_REGISTRY)
    all_stations = sorted(registry.values(), key=lambda s: s.icao)
    all_icaos = [s.nbm_station_id for s in all_stations]

    sample = build_sample(all_stations, SAMPLE_SIZE)
    sample_icaos = [s.nbm_station_id for s in sample]

    # -- a. coarse sweep ---------------------------------------------------
    coarse_dates = rs.first_of_month_range(COARSE_START, COARSE_END)
    coarse_shares: dict[date, float] = {
        d: prober.coverage(sample_icaos, d, phase="coarse") for d in coarse_dates
    }
    coarse_rows = [
        {
            "date": d.isoformat(),
            "share": f"{coarse_shares[d]:.4f}",
            "sample_size": len(sample_icaos),
        }
        for d in coarse_dates
    ]
    write_csv(COARSE_CSV, ["date", "share", "sample_size"], coarse_rows)

    try:
        crossing_month = rs.sustained_crossing_date(
            coarse_dates, lambda d: coarse_shares[d], threshold=THRESHOLD
        )
    except ValueError as exc:
        _stop(
            f"coarse sample coverage never reaches/sustains {THRESHOLD:.0%} "
            f"through {COARSE_END.isoformat()}: {exc}",
            coarse_rows,
        )
        return

    month_index = coarse_dates.index(crossing_month)
    low_bound = coarse_dates[month_index - 1] if month_index > 0 else crossing_month

    # -- c. fine binary search on the full registry -------------------------
    def full_coverage(d: date) -> float:
        return prober.coverage(all_icaos, d, phase="fine")

    pinned_date: date | None = None
    low = low_bound
    search_error: str | None = None
    candidate_highs = coarse_dates[
        month_index : month_index + 1 + MAX_FINE_SEARCH_MONTH_EXPANSIONS
    ]
    for high in candidate_highs:
        try:
            pinned_date = rs.bisect_crossing_date(low, high, full_coverage, THRESHOLD)
            break
        except ValueError as exc:
            search_error = str(exc)
            low = high
            continue

    if pinned_date is None:
        fine_log_rows = [
            entry for entry in prober.call_log if entry["phase"] == "fine"
        ]
        write_csv(
            FINE_LOG_CSV,
            ["phase", "date", "runtime", "sample_size", "hits", "share"],
            fine_log_rows,
        )
        _stop(
            "full-registry coverage never reaches/sustains "
            f"{THRESHOLD:.0%} within the searched range: {search_error}",
            coarse_rows,
        )
        return

    share_at_pinned = full_coverage(pinned_date)

    confirm_shares: dict[date, float] = {}
    for confirm_date in CONFIRM_DATES:
        confirm_shares[confirm_date] = full_coverage(confirm_date)

    confirmation_failures = [
        d for d, share in confirm_shares.items() if share < THRESHOLD
    ]
    if confirmation_failures:
        details = "; ".join(
            f"{d.isoformat()}={confirm_shares[d]:.4f}" for d in confirmation_failures
        )
        _stop(
            "full-registry share dropped back below "
            f"{THRESHOLD:.0%} at confirmation date(s): {details}",
            coarse_rows,
        )
        return

    # Re-write the fine log including the confirmation-phase calls too.
    fine_log_rows = [
        {
            "phase": entry["phase"],
            "date": entry["date"],
            "runtime": entry["runtime"],
            "sample_size": entry["sample_size"],
            "hits": entry["hits"],
            "share": entry["share"],
        }
        for entry in prober.call_log
        if entry["phase"] == "fine"
    ]
    write_csv(
        FINE_LOG_CSV,
        ["phase", "date", "runtime", "sample_size", "hits", "share"],
        fine_log_rows,
    )

    criterion = (
        "Earliest date D such that the canonical archived NBS run "
        f"(nearest 12Z) has a non-null txn for >= {THRESHOLD:.0%} of "
        "registry stations at D, and at every later monthly check date "
        "through the confirmation dates below (build spec #7 decision 6)."
    )
    confirm_evidence = "; ".join(
        f"{d.isoformat()}={confirm_shares[d]:.4f}" for d in sorted(confirm_shares)
    )
    evidence = (
        f"full_registry_share_at_pinned_date={share_at_pinned:.4f} "
        f"({len(all_icaos)} stations); coarse_sample_crossing_month="
        f"{crossing_month.isoformat()}; confirmation_spot_checks: "
        f"{confirm_evidence}"
    )
    write_csv(
        SEED_ARCHIVE,
        ["archive_start_date", "criterion", "evidence"],
        [
            {
                "archive_start_date": pinned_date.isoformat(),
                "criterion": criterion,
                "evidence": evidence,
            }
        ],
    )

    write_readme(
        pinned_date=pinned_date,
        crossing_month=crossing_month,
        share_at_pinned=share_at_pinned,
        confirm_shares=confirm_shares,
        sample=sample,
        all_stations=all_stations,
    )

    print(f"pinned archive start date: {pinned_date.isoformat()}")
    print(f"full-registry share at pinned date: {share_at_pinned:.4f}")
    print(f"coarse sample crossing month: {crossing_month.isoformat()}")
    print(f"live requests made this run: {fetcher.requests_made}")


def write_readme(
    *,
    pinned_date: date,
    crossing_month: date,
    share_at_pinned: float,
    confirm_shares: dict[date, float],
    sample: list[Station],
    all_stations: list[Station],
) -> None:
    if PROBE_RESULTS_CSV.exists():
        with PROBE_RESULTS_CSV.open(newline="") as handle:
            probe_rows = list(csv.DictReader(handle))
        conus_count = len(probe_rows)
        by_reason: dict[str, int] = {}
        for row in probe_rows:
            if row["reason"]:
                by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        excluded_total = sum(by_reason.values())
    else:
        conus_count = len(all_stations)
        by_reason = {}
        excluded_total = 0

    excluded_lines = "\n".join(
        f"- `{reason}`: {count}" for reason, count in sorted(by_reason.items())
    ) or "- (none)"

    confirm_lines = "\n".join(
        f"- {d.isoformat()}: {share:.4f}" for d, share in sorted(confirm_shares.items())
    )

    sample_lines = "\n".join(
        f"- {s.icao} ({s.climate_region})"
        for s in sorted(sample, key=lambda s: s.icao)
    )

    content = f"""# CONUS station registry + NBS archive start (issue #7)

## Sources

- **Station list:** {NWSCLI_CITATION}
- **Climate regions:** {CLIMATE_REGION_CITATION}. Washington DC is not
  listed in any of NCEI's nine region arrays; this project maps it to
  Northeast because DC is geographically enclosed by Maryland, which NCEI
  assigns to Northeast. This is this project's own convention (judgment
  call), not something the source states.
- **Coastal flag:** {COASTLINE_CITATION}. `coastal_flag` is true iff the
  great-circle distance from the station to the nearest coastline segment
  is <= 25 km.
- **Inclusion probes:** IEM's bulk `/api/1/mos.json` (NBS presence) and
  `asos.py` (ASOS + METAR 6-hour group presence), per build spec #7
  decision 4.
- **Archive start:** IEM's `/api/1/mos.json`, per build spec #7 decision 6.

## Station counts

- CONUS candidates (48 states + DC) after the NWSCLI filter: {conus_count}
- Excluded, by reason:
{excluded_lines}
- Included in `station_registry.csv`: {conus_count - excluded_total}

Per-station probe evidence: `probe_results.csv` (every CONUS candidate, its
climate region, coastal distance, and -- if excluded -- the first failing
reason with its probe counts).

## Archive start

**Pinned archive start date: {pinned_date.isoformat()}**

Criterion (build spec #7 decision 6): the earliest date D such that the
canonical archived NBS run (nearest 12Z, per the regime seed) has a
non-null `txn` for at least 95% of registry stations at D, and at every
later monthly check date.

- Coarse sweep (`coarse_coverage_curve.csv`): first-of-month share on a
  stratified 60-station sample, {COARSE_START.isoformat()} through
  {COARSE_END.isoformat()}. The sample stays at or above 95% coverage from
  **{crossing_month.isoformat()}** onward through the end of the range.
- Fine search (`fine_search_log.csv`): daily binary search on the **full**
  registry between the month before the crossing and the crossing month
  itself, landing on {pinned_date.isoformat()}.
- Full-registry share at the pinned date: **{share_at_pinned:.4f}**.
- Confirmation (full registry, June 1 of each year 2021-2026):
{confirm_lines}

### Stratified sample rule (judgment call)

The 60-station coarse sample is apportioned across the 9 NCEI climate
regions proportional to each region's station count in the included
registry (largest-remainder/Hamilton apportionment, ties broken
alphabetically by region name for determinism), then within each region
the stations are sorted by ICAO and evenly spaced indices are selected
(`weather_forecast_audit.registry_sources.select_stratified_sample`). The
build spec described "5 per region-ish"; since 60 / 9 regions is not an
integer and regions vary widely in station count, this project used
population-proportional apportionment rather than a flat 5-per-region so
the sample better represents where most stations actually are, while
remaining fully deterministic and reproducible from the registry alone.
The sample:
{sample_lines}

## Coastal spot checks

Expected true: KSFO, KBOS, KMIA, KSEA. Expected false: KPHX, KDEN, KORD.
Great Lakes shorelines must read false (KBUF, KCLE, KMKE) -- confirmed by
`build_registry.py`, which halts before writing any seed if the Natural
Earth coastline file turns out to include Great Lakes shoreline. See
stdout from that run, and `probe_results.csv` for every station's computed
distance, for the actual values.

## Reproducing this evidence

Both `build_registry.py` and `probe_archive_start.py` cache every HTTP
response under `.cache/registry/` (gitignored). Rerun either with
`--offline` to reproduce `dbt/seeds/station_registry.csv`,
`station_exclusions.csv`, `nbs_archive.csv`, and every file in this
directory byte-identically, with zero new network requests.
"""
    README.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
