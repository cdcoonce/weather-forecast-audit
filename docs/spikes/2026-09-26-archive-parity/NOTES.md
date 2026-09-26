# NWS Forecast Calibration Audit — Archive Parity Spike Notes
Date run: 2026-09-26. Station: KPHX (Phoenix Sky Harbor), lat 33.4342 lon -112.0116, WFO=PSR, grid 161,57.

## 1. What api.weather.gov actually serves
- `/points/33.4342,-112.0116` -> gridId PSR, gridX 161, gridY 57, forecastGridData, forecast, forecastHourly URLs. (points_kphx.json)
- `/gridpoints/PSR/161,57` (raw grid, gridpoint_psr.json, 300KB): maxTemperature/minTemperature series in degC, e.g. maxTemperature validTime "2026-09-26T14:00:00+00:00/PT13H" = 37.78C = 100.0F.
- `/gridpoints/PSR/161,57/forecast` (text periods, forecast_psr.json): Period 1 "Today" temperature 100 F — matches the raw grid maxTemperature value exactly (37.78C->100F), confirming both endpoints are the SAME underlying NDFD-edited grid, just different renderings (raw grid vs. human-readable text periods).
- Per weather-gov/api docs (github.com/weather-gov/api/blob/master/gridpoints.md) and NCEI/MDL docs: api.weather.gov gridpoints are decoded **NDFD** (National Digital Forecast Database) grids — the forecaster-edited product, NOT raw NBM. NBM is the numerical guidance that seeds NDFD; local WFO forecasters can and do edit NDFD away from raw NBM.
- Cadence: updateTime in response ~hourly-ish (observed updateTime 2026-09-26T16:51:43Z); NDFD/AWS bucket updates "as often as once every half hour" per element/projection.
- Horizon: validTimes "2026-09-26T10:00:00+00:00/P7DT15H" → ~7.5 days out (day 1..7 lead times as the audit wants).
- Local-day definition caveat: maxTemperature validTime windows are NOT clean local calendar days — e.g. first max window is 14Z-03Z next day (~13h, 7am-8pm MST) and min windows span 02Z-16Z (~14h). This must be reconciled carefully against whatever "local calendar day" definition the observation source uses.

## 2. Archives found
### NDFD gridded archive (AWS `noaa-ndfd-pds`, keyless S3, us-east-1)
- registry: https://registry.opendata.aws/noaa-ndfd/ ; layout `wmo/<element>/<year>/<month>/<day>/<wmo-bulletin-filename>`.
- Probed via plain HTTPS S3 REST listing (`?list-type=2&prefix=...`), no docs stated depth so I measured it directly:
  - wmo/maxt/2021/01/01/ -> has files (YGAZ97_KWBN_2021...)
  - wmo/maxt/2020/06,09,11,12/01 -> has files
  - wmo/maxt/2020/03/01 -> EMPTY
  - wmo/maxt/2015,2018,2019 -> EMPTY
  - **Conclusion: real archive depth starts ~ Q2 2020 (roughly between Mar and Jun 2020), continuous to present (~6.3 years as of Sept 2026).** Not a rolling few-day window as the "1-2 days" note in one search summary implied — that note was about the *text* NBM guidance retention on a different system, not this bucket.
  - Format: raw WMO/GRIB bulletins mosaicked at CONUS/regional scale — extracting a single station's max/min requires GRIB decode + grid-cell lookup (no per-station API). High effort.

### National Blend of Models (NBM)
- AWS `noaa-nbm-grib2-pds` (registry: https://registry.opendata.aws/noaa-nbm/): GRIB2, back to ~May 2020 per search summary (not independently verified byte-for-byte here — see "unverified" below). Same high-effort GRIB extraction problem as NDFD.
- IEM pre-extracted per-station NBM guidance via `https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station=KPHX&model=NBS&sts=...&ets=...&format=json` — CONFIRMED WORKING. Pulled KPHX NBS guidance issued 2023-07-14T01:00Z for target date 2023-07-15:
  - `ftime=2023-07-15T00:00:00, tmp=114, txn=117.0` (txn = forecast day max)
  - **Observed ASOS max for KPHX 2023-07-15 = 117.0°F (exact match)** — see asos_daily_phx_20230715.csv.
  - This is a low-effort, pre-parsed, per-station JSON/CSV API — far better access than the GRIB archives.
  - IEM notes model=NBS archived at cycles 1,7,13,19 UTC only after 25-Feb-2020 (0,7,12,19 before). Implies IEM's NBM station archive starts ~2020ish; did not pin exact start date (see unverified).

### IEM MOS archive (legacy GFS MOS: MAV/MEX, model codes must be one of `AVN|ETA|GFS|LAV|MEX|NAM|NBE|NBS`)
- Same `mos.py` endpoint. Query for `model=AVN` KPHX 2023-07-14 returned 0 rows (either wrong station/date combo or that legacy model wasn't archived for this station on this date) — did NOT get a positive pull; web search claims archive "complete back to 3 Dec 2008" / "goes back to June 2000" for the fe.phtml UI — **unverified by direct pull**.

## 3. Observations for scoring
- **IEM CLI report archive** (`https://mesonet.agron.iastate.edu/json/cli.py?station=KPHX&year=YYYY`) — parses official NWS Daily Climatological Report (CLI) text products. Depth confirmed by direct probing:
  - year=2001: 0 results; year=2002: 309 results (partial;year=2003 on: full 365/366)
  - **Real depth: essentially complete from 2002/2003 to present (~24 years).**
  - CLI report = "official" daily max/min for the fixed Local Standard Time climate day, at NWS first-order/climate stations (KPHX is one). This is the station-of-record product NWS itself treats as ground truth for climate records and is the closest thing to what NWS verification systems use for max/min (per weather.gov CLI documentation and NWS Instruction 10-1302) — **the CLI-report basis is documented, but I did NOT find an explicit NWS statement that its automated forecast-verification systems (e.g. VSDB-successor) score against CLI rather than raw ASOS daily summary; treat "NWS verifies against CLI" as PLAUSIBLE, not confirmed. Labeled unverified below.**
- **IEM ASOS daily summary** (`/cgi-bin/request/daily.py?network=AZ_ASOS&stations=PHX&year1=...`) — CONFIRMED WORKING, pulled KPHX 2023-07-15: max_temp_f=117.0, min_temp_f=93.0 (see asos_daily_phx_20230715.csv). This is derived from raw 5-min ASOS obs (calendar-day midnight-midnight by default, not the fixed LST climate day used by CLI) — a subtly different "day" definition than CLI.

## 4. KPHX end-to-end field/unit check
- Current NDFD grid max (today, 2026-09-26): 37.78°C = 100.0°F; text /forecast period "Today" = 100°F. Units/conversion consistent.
- Archived NBM (NBS) forecast issued 2023-07-14 for target 2023-07-15: txn=117.0°F.
- Observed ASOS max 2023-07-15: 117.0°F. Same units (°F), values match exactly in this one sample.
- Could NOT pull an archived NDFD-grid (not NBM) value for a past date/station to compare directly against NBM/obs, because the NDFD AWS archive is GRIB/regional-mosaic only (no per-station pre-extraction) and doing a GRIB decode was out of scope for a keyless-curl spike. **This exact NDFD-vs-NBM-vs-obs three-way comparison for the same past date is UNVERIFIED — only NBM-vs-obs was directly confirmed.**

## Files in this directory (committed copies live in `samples/`)
- points_kphx.json — api.weather.gov /points response for KPHX coords
- gridpoint_psr.json — raw NDFD gridpoint data (PSR/161,57), maxTemperature/minTemperature series
- forecast_psr.json — text /forecast periods for same gridpoint
- cli_kphx_2026.json, cli_kphx_2000.json — IEM CLI report JSON pulls (depth probing)
- daily_py_help.txt — IEM daily.py page (returned HTML nav, not raw help — endpoint usage confirmed working directly instead)
- asos_daily_phx_20230715.csv — IEM ASOS daily summary for PHX 2023-07-15/16 (max=117.0F)
- nbs_kphx_20230714.json — IEM NBM (NBS) station guidance issued 2023-07-14 for KPHX
- avn_mos_kphx_20230714.json — empty result for legacy GFS MOS (AVN) query, KPHX 2023-07-14
- mav_kphx_20230714.json — error response demonstrating valid `model` enum values
- ndfd_bucket_root.xml, ndfd_wmo_2021.xml — S3 listing probes of noaa-ndfd-pds bucket
- nbs_phx_20230715.txt — failed AFOS pil guess (NBSPHX not a valid pil), kept as evidence of dead end
- nwstext_help.txt, mos_help.txt — IEM backend doc pages (mos_help.txt contains the working CGI arg reference)

## Addendum (2026-09-26, follow-up probe): archived NBS cycle regime change
Probed `mos.py?station=KPHX&model=NBS` one day at a time to list distinct `runtime` hours:
- 2020-06-01, 2021-06-01, 2023-06-01, 2025-06-01 … 2026-04-15: cycles **01 07 13 19Z**
- 2026-05-01: **all 24 hourly cycles** (transition artifact)
- 2026-05-10 … 2026-09-25: cycles **00 06 12 18Z**

Consequence: there is no single "12Z run" across the record. The canonical daily run is the archived cycle nearest 12Z (13Z before the changeover, 12Z after). Record the cycle hour on every row and treat the changeover (between 2026-04-15 and 2026-05-10; exact date to be pinned) as an annotated regime boundary, like an NBM version upgrade. Whether it coincides with an NBM version change is unverified.

Note: the IEM help pages fetched during the spike (mos/daily/nwstext help) are not committed; `samples/` holds only the API responses listed above.
