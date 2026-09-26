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

- `target_date` is the UTC calendar date the window covers (`D` above),
  taken from the window's own definition, not derived from `ftime` by a
  fixed offset in one direction: a `max` row's `ftime` lands at `D+1 00Z`,
  so `target_date = ftime.date() - 1 day`; a `min` row's `ftime` lands at
  `D 12Z`, so `target_date = ftime.date()`.
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
- **Lead-3 max never occurs in NBS.** For a 13Z run, the max window for
  `run_date + 3` would need an `ftime` at `run_date + 4, 00Z`, one hour past
  NBS's guidance horizon; NBS simply does not emit that row. The resolver
  treats any `ftime` hour other than `00` or `12` as a parse error rather
  than silently accepting an unexpected shape.

## Cycle regimes: which archived run is canonical

IEM's NBS archive keeps only the run "nearest 12Z" for older dates, plus
denser cadences right around a 2026-04-30 changeover. Probed evidence (KPHX
and KORD identical):

- **2020-02-25 through 2026-04-29**: only the **13Z** run is archived
  (`{1, 7, 13, 19}` UTC cycles exist; 13Z is nearest 12Z of those). The
  2020-02-25 start date is from IEM's own help text ("archived at 1, 7, 13,
  19 UTC only after 25 Feb 2020"), not independently probed.
- **2026-04-30 onward**: the archive gains denser cadences (2026-04-30 has
  `{0, 1, 7, 8-23}`; 2026-05-01 through 2026-05-04 have all 24 hours;
  2026-05-05 has `{0-12, 18}`) and settles back to a steady `{0, 6, 12, 18}`
  cadence from 2026-05-06. The canonical cycle nearest 12Z becomes **12Z**.

These dates and hours live only in `dbt/seeds/nbs_cycle_regimes.csv`, read
by `weather_forecast_audit.regimes`; nothing in the Python resolver hard-codes
a changeover date. A singular dbt test
(`dbt/tests/singular/cycle_hour_matches_regime.sql`) checks every fact row's
`cycle_hour` against this seed.

## Completeness threshold

An hourly observation window is **scorable** only when it has at least
`ceil(0.75 x 18) = 14` distinct UTC clock-hours with at least one
non-missing `tmpf` reading (routine or special obs both count); the
observed extreme is the max/min of every non-missing reading in the window,
not just one per hour. `MIN_HOUR_COVERAGE = 0.75` is a named constant in
`weather_forecast_audit.resolver`.

Rationale: 75% coverage tolerates a short outage (up to 4 missing hours out
of 18) without silently dropping a day, while still requiring enough of the
window's hours on file that the recorded extreme is unlikely to have missed
the true daily peak or trough. A stricter threshold drops more days at
stations with patchier ASOS coverage; issue #6 is expected to revisit this
methodology once more stations are in the audit.

## Boundary convention

Windows are half-open: **`[start, end)`**. An observation exactly at the
window's start counts; one exactly at the window's end does not. The MDL
text card does not specify this either way — this is this project's own
convention, applied consistently by the resolver and unit-tested at the
boundary (`tests/unit/test_resolver.py`).

## Hourly-sampling caveat

The hourly-ASOS-derived observed extreme is a lower bound on the true daily
extreme: it only sees the top-of-hour-ish routine and special reports IEM
archives, not the full 1-minute ASOS record NWS uses to compile the CLI
product. On the hand-checked KPHX days, the published CLI daily report was
**1°F more extreme than the hourly-derived value on both the max and the
min**, on both days:

| date | hourly max | CLI high | hourly min | CLI low |
| --- | --- | --- | --- | --- |
| 2023-07-14 | 115 | 116 | 94 | 93 |
| 2023-07-15 | 117 | 118 | 93 | 92 |

`fct_forecast_verification.cli_f` carries the CLI value alongside the
hourly-derived `observed_f` specifically so this gap is visible per row,
not just in this note. Issue #6 is expected to measure how often and by how
much this happens across more stations and more days.
