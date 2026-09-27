# CONUS station registry + NBS archive start (issue #7)

## Sources

- **Station list:** https://mesonet.agron.iastate.edu/geojson/network/NWSCLI.geojson
- **Climate regions:** NOAA NCEI's nine U.S. climate regions (Karl, T.R. and Koss, W.J., 1984: "Regional and National Monthly, Seasonal, and Annual Temperature Weighted by Area, 1895-1983," Historical Climatology Series 4-3, National Climatic Data Center), https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-climate-regions. Washington DC is not
  listed in any of NCEI's nine region arrays; this project maps it to
  Northeast because DC is geographically enclosed by Maryland, which NCEI
  assigns to Northeast. This is this project's own convention (judgment
  call), not something the source states.
- **Coastal flag:** https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@v5.1.2/geojson/ne_10m_coastline.geojson (pinned tag v5.1.2). `coastal_flag` is true iff the
  great-circle distance from the station to the nearest coastline segment
  is <= 25 km.
- **Inclusion probes:** IEM's bulk `/api/1/mos.json` (NBS presence) and
  `asos.py` (ASOS + METAR 6-hour group presence), per build spec #7
  decision 4.
- **Archive start:** IEM's `/api/1/mos.json`, per build spec #7 decision 6.

## Station counts

- CONUS candidates (48 states + DC) after the NWSCLI filter: 610
- Excluded, by reason:
- `no_asos_observations`: 20
- `no_nbs_guidance`: 7
- `no_six_hour_groups`: 10
- Included in `station_registry.csv`: 573

Per-station probe evidence: `probe_results.csv` (every CONUS candidate, its
climate region, coastal distance, and -- if excluded -- the first failing
reason with its probe counts).

## Archive start

**Pinned archive start date: 2020-09-29**

Criterion (build spec #7 decision 6): the earliest date D such that the
canonical archived NBS run (nearest 12Z, per the regime seed) has a
non-null `txn` for at least 95% of registry stations at D, and at every
later monthly check date.

- Coarse sweep (`coarse_coverage_curve.csv`): first-of-month share on a
  stratified 60-station sample, 2018-11-01 through
  2026-09-01. The sample stays at or above 95% coverage from
  **2020-10-01** onward through the end of the range.
- Fine search (`fine_search_log.csv`): daily binary search on the **full**
  registry between the month before the crossing and the crossing month
  itself, landing on 2020-09-29.
- Full-registry share at the pinned date: **1.0000**.
- Confirmation (full registry, June 1 of each year 2021-2026):
- 2021-06-01: 1.0000
- 2022-06-01: 1.0000
- 2023-06-01: 1.0000
- 2024-06-01: 1.0000
- 2025-06-01: 1.0000
- 2026-06-01: 1.0000

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
- K1V4 (Northeast)
- K2WX (Northern Rockies and Plains)
- KAAF (Southeast)
- KAAT (West)
- KABI (South)
- KABQ (Southwest)
- KALO (Upper Midwest)
- KALW (Northwest)
- KARR (Ohio Valley)
- KBGR (Northeast)
- KBKV (Southeast)
- KBTM (Northern Rockies and Plains)
- KCDS (South)
- KCID (Upper Midwest)
- KCLE (Ohio Valley)
- KDAG (West)
- KDAN (Southeast)
- KDIK (Northern Rockies and Plains)
- KDPA (Ohio Valley)
- KDRO (Southwest)
- KELD (South)
- KELN (Northwest)
- KFLL (Southeast)
- KFVE (Northeast)
- KGPI (Northern Rockies and Plains)
- KGRR (Upper Midwest)
- KGUY (South)
- KIAD (Southeast)
- KIML (Northern Rockies and Plains)
- KIRK (Ohio Valley)
- KJBR (South)
- KLAX (West)
- KLMT (Northwest)
- KMDT (Northeast)
- KMDW (Ohio Valley)
- KMEI (South)
- KMHE (Northern Rockies and Plains)
- KMIW (Upper Midwest)
- KMOB (Southeast)
- KNYL (Southwest)
- KONT (West)
- KOTH (Northwest)
- KPHD (Ohio Valley)
- KPHP (Northern Rockies and Plains)
- KPIT (Northeast)
- KPSX (South)
- KROA (Southeast)
- KSAC (West)
- KSEA (Northwest)
- KSNA (West)
- KSTJ (Ohio Valley)
- KTVC (Upper Midwest)
- KVCT (South)
- KVEL (Southwest)
- KVSF (Northeast)
- KWAL (Southeast)
- KWVI (West)
- KXWA (Northern Rockies and Plains)
- KYKM (Northwest)
- KZZV (Ohio Valley)

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
