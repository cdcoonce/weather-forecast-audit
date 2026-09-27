# Methodology: KPHX tracer (v1)

This is the audit's verification methodology as of the KPHX tracer slice
(issue #5): how a forecast is matched to what actually happened, and every
judgment call behind that match. It will grow as later slices add sources,
stations, and the bias-correction experiment.

## Verification windows are fixed UTC, not local time

NBM's daily max/min guidance (the `TXN` element of the NBS product) is
defined in UTC by the MDL NBM text card (v4.0-v5.0, verbatim):

> TXN = 18-hour maximum and minimum temperatures, degrees F. Min is between
> 00Z-18Z and reported at 12Z ... Max is between 12z(current
> day)-06Z(next day) and reported at 00z(following day).

Source: <https://vlab.noaa.gov/web/mdl/nbm-textcard-v4.2> (identical wording
at v5.0).

This project takes those windows literally, in UTC, for every station
regardless of its local time zone or daylight-saving transitions:

- **max window** for target date D: `[D 12:00Z, D+1 06:00Z)`
- **min window** for target date D: `[D 00:00Z, D 18:00Z)`

Both are always exactly 18 hours; nothing about the resolver reads a
station's time zone. This is deliberately different from the GFS MOS (MAV)
convention, which verifies against a fixed 0700-1900 **local standard time**
window (MDL TPB 05-03, <https://www.weather.gov/media/mdl/mdltpb05-03.pdf>).
MOS and NBM/NBS are different products; the 0700-1900 LST convention does
not apply here.

`XND` (the guidance's spread column) is the **standard deviation** of the
max/min forecast, not a max-minus-min range; it is carried through as
`spread_f` and not otherwise interpreted in v1.

## target_date and lead_day

- `target_date` is the UTC date the window starts on (`D` above). The two
  variables are reported at different offsets from their windows: a `max`
  is reported at `D+1 00Z` (`target_date = ftime.date() - 1 day`), a `min`
  at `D 12Z` (`target_date = ftime.date()`).
- `run_date` is the UTC calendar date of the guidance run's `runtime`.
- `lead_day = target_date - run_date`, in whole days.

## Scope: lead 1-3, no lead-0 max, no lead-3 max

v1 only publishes `lead_day` in `{1, 2, 3}` in the fact table. Two guidance
rows exist but are deliberately out of scope, though both are kept in the
raw and resolved tables for later slices to use:

- **Lead 0** only exists for the 12Z-regime same-day max (a 12Z run's max
  window starts at 12Z the same day, i.e. its own issuance hour). A 13Z run
  has already missed that day's max window's start, so it never carries a
  lead-0 max at all.
- **Lead-3 max never occurs in NBS.** It would be reported at
  `run_date + 4, 00Z`, beyond the short-range NBS horizon: the last `TXN` a
  13Z or 12Z run carries is the lead-3 min at `run_date + 3, 12Z`. So at
  lead 3 the audit grades minimums only. The resolver
  treats any `ftime` hour other than `00` or `12` as a parse error rather
  than silently accepting an unexpected shape.

## Cycle regimes: which archived run is canonical

IEM archives four NBS cycles a day, and which four changed in spring 2026.
The audit grades one run per day: the archived cycle nearest 12Z. Probed
evidence (KPHX and KORD identical):

- **2020-02-25 through 2026-04-29**: cycles `{1, 7, 13, 19}` UTC, so the
  canonical run is **13Z**. The 2020-02-25 start comes from IEM's own help
  text ("archived at 1, 7, 13, 19 UTC only after 25 Feb 2020") and was not
  independently probed.
- **2026-04-30 onward**: the canonical run is **12Z**. 2026-04-30 is the
  first day a 12Z run is archived. The transition days hold extra cycles
  (2026-04-30 has `{0, 1, 7, 8-23}`, 2026-05-01 through 2026-05-04 have all
  24 hours, 2026-05-05 has `{0-12, 18}`), and the archive settles to
  `{0, 6, 12, 18}` from 2026-05-06.

When the canonical run for a date is missing, the day gets a `missing_run`
gap record. The pipeline never falls back to a neighboring cycle, which
would mix issuance times within one lead bucket.

These dates and hours live only in `dbt/seeds/nbs_cycle_regimes.csv`, read
by `weather_forecast_audit.regimes`; nothing in the Python resolver hard-codes
a changeover date. A singular dbt test
(`dbt/tests/singular/cycle_hour_matches_regime.sql`) checks every fact row's
`cycle_hour` against this seed.

## Observed extremes

The observed max/min over a verification window comes from the METAR
6-hour maximum (`1snTTT`) and minimum (`2snTTT`) remark groups, not from
the hourly `tmpf` readings the KPHX tracer slice (issue #5) used. Issue #6
measured both sources (and the CLI daily report) across 12 stations' 2025
windows and chose the source by a rule pre-registered before the
measurement ran; see
[`docs/analysis/2026-09-26-extreme-source/PREREG.md`](analysis/2026-09-26-extreme-source/PREREG.md)
for the fixed decision rule and
[`docs/analysis/2026-09-26-extreme-source/RESULTS.md`](analysis/2026-09-26-extreme-source/RESULTS.md)
for the full measurement.

### Source: `metar_6h`, tiled by three synoptic periods

Each window is tiled by three consecutive 6-hour synoptic periods:

- **max window** `[D 12Z, D+1 06Z)` is tiled by the periods ending
  `D 18Z`, `D+1 00Z`, and `D+1 06Z`.
- **min window** `[D 00Z, D 18Z)` is tiled by the periods ending
  `D 06Z`, `D 12Z`, and `D 18Z`.

For the period ending at synoptic hour `H`, the group comes from the
latest qualifying report valid in `[H - 60 min, H)`; US synoptic reports
are issued at about `H - 9 min`, so a 60-minute lookback comfortably
covers the issuance report without reaching back far enough to catch the
previous one. Because of that 9-minute offset, the reports that actually
tile a window arrive a few minutes before each synoptic hour rather than
on it: a max window's three periods are typically covered by reports
around 17:51Z, 23:51Z, and 05:51Z, so in practice the groups that decide a
max window span about `[11:51Z, 05:51Z)`, not the window's own
`[12:00Z, 06:00Z)` boundary.

A window is **scorable** exactly when all three periods find a qualifying
report (`extreme_source = 'metar_6h'`); the observed value is the max/min
of the three found values, kept in °F to the tenths-of-°C precision of the
source and not rounded. When one or more periods cannot be tiled, the
window is **unscorable** (`extreme_source = 'none'`, `observed_f` is
null). There is no hourly fallback and no bias correction: PREREG's
fallback rule requires pooled tiling of at least 0.95 with every station
at or above 0.85 before treating an untiled window as simply unscorable,
and the measured sample cleared that bar (pooled tiling 0.973 for both
variables; the lowest station, KSLC, was 0.912). Mixing in hourly-derived
values at the lower-tiling stations would reintroduce exactly the
sampling bias `metar_6h` exists to avoid.

### Why `metar_6h` and not hourly

PREREG's decision rule adopts `metar_6h` if pooled tiling is at least 0.90
for both variables and `metar_6h` agrees with the CLI daily report better
than hourly does, for both variables. RESULTS.md's measurement, pooled
across the 2025 sample:

| variable | pooled tiling | mean(hourly - CLI) | mean(metar_6h - CLI) |
| --- | --- | --- | --- |
| max | 0.973 | -0.939 F | -0.209 F |
| min | 0.973 | +1.355 F | +0.806 F |

Both conditions hold clearly, so `metar_6h` is primary.

### The hourly sampling bias this replaces

An hourly reading is a snapshot at about :51 past the hour; it can only
miss a true peak or trough between readings, never exceed it, so the
hourly max reads low and the hourly min reads high. Pooled mean(hourly -
metar_6h) by season, on windows both sources could resolve:

| variable | DJF | MAM | JJA | SON |
| --- | --- | --- | --- | --- |
| max | -0.55 F | -0.79 F | -0.86 F | -0.71 F |
| min | +0.58 F | +0.56 F | +0.52 F | +0.54 F |

`fct_forecast_verification.hourly_observed_f` keeps the hourly value as a
diagnostic only, not used in `error_f`, so this artifact stays visible
rather than silently biasing the score. `scripts/sql/kphx_mean_error.sql`
reports the hourly-based error alongside the official one for the same
reason.

### The min-vs-CLI residual: an NBM-window effect, not a decoding error

Even `metar_6h` does not match CLI exactly: pooled mean(metar_6h - CLI) is
+0.81 F on min (max is close to zero, at -0.21 F). This residual is not
spread evenly across stations. It is largest in winter at continental
stations (KMSP +2.55 F, KBIS +2.08 F, KDEN +1.85 F, DJF) and close to
zero at mild, temperature-stable stations (KSEA, KPHX, KSFO). That pattern
points to a definitional mismatch, not a parsing error: CLI's day is
local-standard midnight to midnight, while the NBM min window is the
fixed UTC `[D 00Z, D 18Z)`. At a continental station in winter, the
coldest moment of the local day can fall after 18Z, which the CLI day
still counts but the NBM window has already closed; a mild coastal or
desert station's overnight low is far less likely to keep falling that
late. Consistent with this being a window-definition effect rather than a
source error, max agrees with CLI exactly on 98% of summer days pooled
(JJA, round(metar_6h) equal to CLI).

`fct_forecast_verification.cli_f` still carries the CLI value alongside
`observed_f` on every row; it remains the reference check used above, not
an input to `error_f`.

## Boundary convention

Windows are half-open: **`[start, end)`**. An observation exactly at the
window's start counts; one exactly at the window's end does not. The MDL
text card does not specify this either way, so this is the project's own
convention, applied consistently by the resolver and unit-tested at the
boundary (`tests/unit/test_resolver.py`).
