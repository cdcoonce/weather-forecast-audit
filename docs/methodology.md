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

- **2020-09-29 through 2026-04-29**: cycles `{1, 7, 13, 19}` UTC, so the
  canonical run is **13Z**. IEM's own help text says this cycle schedule
  applies "only after 25 Feb 2020", and it was archiving at `{1, 7, 13, 19}`
  UTC from then on, but this regime's `valid_from` is the pinned NBS
  archive start (issue #7 decision 6), 2020-09-29, not that earlier
  cycle-schedule date: before the NBM v4.0 rollout on 2020-09-29, the
  archive's daily max/min lived in a column named `n_x`, not `txn`, which
  this project does not read (NBM v3.x's N/X period definition is
  unverified and may be local-time like MOS, which our fixed-UTC
  verification windows cannot grade -- a documented follow-up, not built
  here). Earlier still, `{0, 7, 12, 19}` UTC cycles ran through 2020-02-25,
  entirely before the archive's usable (`txn`) start. See
  `dbt/seeds/nbs_archive.csv` and
  `docs/spikes/2026-09-26-station-registry/README.md` for the search that
  pinned 2020-09-29.
- **2026-04-30 onward**: the canonical run is **12Z**. 2026-04-30 is the
  first day a 12Z run is archived. The transition days hold extra cycles
  (2026-04-30 has `{0, 1, 7, 8-23}`, 2026-05-01 through 2026-05-04 have all
  24 hours, 2026-05-05 has `{0-12, 18}`), and the archive settles to
  `{0, 6, 12, 18}` from 2026-05-06.

When the canonical run for a date is missing, the day gets a `missing_run`
gap record; when the canonical run exists but none of its rows carry a
non-null `txn` (as with any pre-2020-09-29 date, if ever queried), the day
gets a `no_txn` gap record instead, so guidance that is present but
unusable is never mistaken for guidance that is simply absent. The pipeline
never falls back to a neighboring cycle, which would mix issuance times
within one lead bucket.

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

## Scoring and uncertainty

`weather_forecast_audit.scoring.score` (issue #8) turns
`fct_forecast_verification` rows into every headline number the site
publishes: bias, MAE, skill, and their uncertainty, sliced any way a page
needs. It is a pure function -- no I/O, no DuckDB, no dbt -- so it can be
unit-tested against hand-built and synthetic fixtures
(`tests/unit/test_scoring.py`).

### Input and slicing

Only `scorable = true` rows with a non-null `error_f` are scored; anything
else is dropped before any statistic is computed. A slice can combine
`station`, `month`, `season`, `lead_day`, and `variable`; `source` is
always an implicit grouping key on top of whatever else is requested,
because pooling error across `raw_nbm`, `baseline`, and `challenger` would
average away the exact comparison the audit exists to make.

`month` and `season` (meteorological: DJF/MAM/JJA/SON) are derived from
**`target_date`**, not `run_date`. "Highs run cold in July" is a statement
about the weather day being forecast, not the day guidance was issued: a
lead-3 run issued 2025-06-29 verifies 2025-07-02, and belongs in July's
bucket even though it was produced in June.

### Bias, MAE, and skill

`bias` is mean signed error (`mean(error_f)`); `mae` is mean absolute
error. `skill = 1 - MAE_source / MAE_raw_nbm` is computed on **matched
pairs only**: a row of a non-`raw_nbm` source counts toward skill only
when a scorable `raw_nbm` row exists for the same `(station, run_date,
lead_day, variable)`, and both MAEs in the ratio are taken over that same
matched set, not each source's full slice. Without this restriction, a
challenger that only issues guidance on a favorable subset of days could
look skillful by comparison to `raw_nbm`'s unrestricted MAE, which includes
harder days the challenger never attempted. Unmatched rows still count
toward that source's own `bias`/`mae` -- they are simply excluded from the
head-to-head skill ratio. `raw_nbm` is skill's reference by construction,
so its own skill is always exactly `0.0`. A slice with no matched pairs, or
whose matched `raw_nbm` MAE is exactly 0, reports `skill` (and its CI) as
null rather than dividing by zero.

### Why the bootstrap resamples calendar dates, not rows

A bad NBM cycle rarely misses at one station and hits at the rest. The
same model run and the same synoptic pattern drive every station's error
that day, so errors from different stations on the same issuance date
are correlated.

Resampling individual verification rows (a row-level bootstrap) treats
every row as an independent draw. With 40 stations on 60 dates it behaves
as if it had 2,400 independent errors, when the shared day-to-day
component has only 60 independent draws. Its standard error shrinks with
the row count instead of the date count, so the interval comes out far
too narrow and misses the true bias much more often than its nominal
rate.

The date-block bootstrap instead resamples **calendar blocks of issuance
dates** with replacement -- every row sharing a block moves together as
one unit, preserving whatever correlation exists within a block. Measured
on synthetic data built to have exactly this day-to-day correlation (a
day effect shared by all 40 synthetic stations plus independent
per-station noise, true bias `μ = 0.7`), 200 simulations of a 95% date-block
interval covered `μ` in 96.5% of simulations (the test requires 90-99%),
while row-level resampling on the identical data covered it in 30.5%.
(See
`test_date_block_coverage_beats_row_level` in `tests/unit/test_scoring.py`
for the exact generator and the measured numbers; the module's own
`_block_bootstrap` is reused for both arms, passing individual-row block
ids to get the row-level comparison, so the two paths differ only in what
"block" means.)

Concretely: `block_id = (run_date - 1970-01-01).days // block_days`, so
`block_days = 1` resamples one issuance date at a time and larger values
group consecutive calendar dates into one resampling unit. The default,
`BLOCK_DAYS = 14`, is measured (limitation 2).
The bootstrap itself operates on per-block sums (`Σ error`, `Σ
|error|`, and for skill, the matched-pair sums of `|error_source|` and
`|error_raw|`), not on materialized resampled rows, so it stays cheap even
at `N_BOOT = 2000` replicates. The CI is the ordinary bootstrap percentile
interval at the requested `ci_level`. A slice with fewer than two distinct
blocks cannot support a bootstrap at all; its CI is null and
`no_detectable_bias` defaults to `True`. That choice is conservative: a
slice whose interval could not be estimated is shown as "no detectable
bias", never as a detected bias.

Every slice's random draws come from a generator seeded from a stable hash
(`sha256` of the slice's own key, combined with a fixed base `SEED`), never
Python's built-in `hash()` (which is salted per process and would make
results irreproducible run to run). This makes a slice's CI bit-identical
across repeated calls and independent of what other, unrelated slices
happen to be present in the same call -- adding a second station's rows
cannot perturb the first station's confidence interval.

### `min_sample_flag` and `no_detectable_bias`

`min_sample_flag` fires when a slice has fewer than `MIN_SAMPLE_BLOCKS`
(21, measured in #37) **distinct bootstrap blocks** (`n_blocks`), not
dates and not rows. Blocks of issuance dates are the bootstrap's
independent resampling unit. A slice built from 30 stations reporting on
one date is not more trustworthy than one station reporting on 30 dates,
and counting rows would make it look so. Counting dates fails the same
way one level up: 180 consecutive dates are only 14 blocks, and a
percentile interval over 14 resampled units is too narrow (limitation 3).
`n_dates` is still reported, and it is what the site shows.

`no_detectable_bias` is `bias_lo <= 0 <= bias_hi` (both endpoints
inclusive: a CI that touches exactly zero still "spans" it). When the CI
is undefined, it defaults to `True` for the same conservative reason as
above.

### Limitation 1: pooled-lead slices understate correlation across leads

Blocks are keyed on **issuance date**, not target/weather date. When a
slice pools several lead days together (e.g. `by=("station",)` with no
`lead_day` restriction), the rows verifying the *same* weather day come
from *different* issuance dates -- a lead-1 and a lead-3 row that both
verify against 2025-07-02 were issued on 2025-07-01 and 2025-06-29
respectively, and land in different date blocks. That splits up
correlation that a same-weather-day view would otherwise capture as one
unit, so pooled-lead intervals are somewhat too narrow. The site's headline
numbers are always reported per lead day, so this only affects pooled
summaries and is not fixed here.

### Limitation 2 (resolved by #31): the block length is now measured

`BLOCK_DAYS = 1` treated consecutive issuance dates as independent of each
other. Forecast errors persist across multi-day weather regimes, since a
stuck upper-level pattern can bias guidance the same way for a week, so
per-date intervals are too narrow whenever that persistence is real.
`test_block_days_seven_beats_block_days_one_on_ar1_data` shows the knob
works: with an AR(1) day effect at ρ = 0.8, nominal 95% intervals cover the
true bias in 48% of simulations with 1-day blocks and 82% with 7-day blocks.
That proves the mechanism, not the number.

**The number: `BLOCK_DAYS = 14`**, chosen by a rule committed before any
data was fetched (`docs/analysis/2026-09-27-block-length/`, `PREREG.md`
then `RESULTS.md`). On #6's 12 stations over 2025 issuance dates, the
Politis-White (2004) circular-block estimator with the Patton, Politis and
White (2009) correction was run on the daily cross-station mean error for
each (variable, lead), and on each station's own series. The rule takes
the larger of the pooled estimate and the upper-median station estimate per
(variable, lead), then the ceiling of the maximum. The binding series was
max lead 2's pooled mean: 13.18, so 14.

| (variable, lead) | pooled | upper-median station | pooled, deseasonalized |
| --- | --- | --- | --- |
| max, 1 | 12.80 | 6.80 (KORD) | 8.32 |
| max, 2 | **13.18** | 7.12 (KORD) | 8.66 |
| min, 1 | 6.27 | 6.01 (KMIA) | 4.42 |
| min, 2 | 9.47 | 6.58 (KSEA) | 5.70 |
| min, 3 | 6.99 | 8.47 (KSLC) | 5.41 |

A coverage replay was the acceptance check. An AR(p) model (Yule-Walker,
p ≤ 7 by AIC) was fitted to each pooled series and to each upper-median
station's series, 400 years of each were simulated, and each was scored
through `scoring.score`. At 14 every one of the ten series covered at least
0.90 (worst 0.9075, pooled min lead 2). At the old default of 1, coverage
fell as low as 0.6475 (pooled max lead 2), so the intervals published
before #31 would have been far too narrow.

What the number does and does not rest on:

- **It lands exactly on the pre-registered cap of 14.** The cap fires only
  above 14, so 14 stands. A slightly more persistent year would have sent
  the choice back for review rather than raising the default.
- **About 4.5 of the 14 days are the seasonal cycle.** Removing a centered
  31-day mean lowers the largest estimate to 8.66. A slow seasonal swing in
  the mean error reads as persistence to a full-year estimator. That errs
  toward wider intervals, which is the safe side for a bias claim, but it
  means 14 is conservative for per-season slices, whose own window already
  holds the season roughly fixed.
- **The replay cannot represent the seasonal swing.** A low-order AR model
  captures day-to-day persistence only, so the 0.90 pass certifies the
  short-memory part. The seasonal part is covered by the estimator's
  conservatism above, not by the replay.
- **Single stations can persist longer than the median.** KSFO's own max
  estimates are 14.4 and 14.3 (marine-layer regimes). The upper median keeps
  one station from setting a national default, so a KSFO interval is
  somewhat narrower than its own persistence warrants.
- **Short slices are a separate problem** -- see limitation 3.

### Limitation 3 (resolved by #37): the sample floor counts blocks

The #31 replay certified `BLOCK_DAYS = 14` on full-year series. Replayed at
shorter lengths, nominal 95% intervals covered only 0.83–0.89 at 90 dates
(one season) and 0.73–0.80 at 30 dates, the old `MIN_SAMPLE_DATES`. No
block length fixed that. A percentile bootstrap over a handful of
resampled units is too narrow on its own, and
`test_date_block_coverage_beats_row_level` shows it even with no
persistence (0.870 on 60 iid dates at 14-day blocks, which is why that
test pins `block_days=1`).

**The floor: `MIN_SAMPLE_BLOCKS = 21`**, chosen by a replay pre-registered
before it ran (`docs/analysis/2026-09-27-sample-floor/`). The ten AR models
fitted in #31 were simulated 1000 times per slice shape at 14-day blocks.
The shapes were contiguous runs of 60–730 dates, and season-shaped slices
made of one 92-date JJA segment per year for 1–6 years. The floor is the
smallest block count at which every shape with at least that many blocks
kept all ten models at or above 0.90:

| shape | dates | blocks | worst of ten |
| --- | --- | --- | --- |
| contiguous | 180 | 14 | 0.879 |
| contiguous | 270 | **21** | **0.904** |
| contiguous | 365 | 27 | 0.911 |
| season, 2 years | 184 | 14 | 0.885 |
| season, 3 years | 276 | 22 | 0.913 |
| season, 6 years | 552 | 46 | 0.920 |

One shape (540 contiguous dates, 40 blocks, 0.894 on one model) was re-run
at 2000 sims under the pre-registered noise protocol and read 0.912.

What it means for the site. A per-season cell (`station`, `variable`,
`lead_day`, `season`) pools every archive year, so it needs **three years
of that season** before it can carry a bias claim. A single season, or two,
reads as "too few days". The national backfill (#11, archive from
2020-09-29) gives every station about six years. Until that backfill runs,
essentially every published cell is flagged, and that is the honest state.

## Walk-forward evaluation

`weather_forecast_audit.walkforward` (issue #17) is the harness a
bias-correction model is judged by: it walks a `Model` (a `fit`/`predict`
pair) forward through issuance dates and returns one row per prediction,
never letting the model see a verification outcome before that outcome
actually existed. It is a pure function over `fct_forecast_verification`
rows -- no I/O, no DuckDB, no dbt -- unit-tested against synthetic fixtures
(`tests/unit/test_walkforward.py`).

### The leak a run-date cutoff misses

The obvious cutoff -- train on every run issued before the run being
evaluated -- leaks. NBS carries lead 1-3 in one guidance run, so a single
run at issuance time *T* reports a `max`/`min` for several different
future weather days at once. Concretely, run *t-1*'s lead-3 `min` (target
`t+2`) verifies the *same weather day* as run *t*'s own lead-2 pair
(target `t+2`). If training data is sliced by `run_date < t`, that lead-3
row from run *t-1* is already in scope once run *t* is evaluated, even
though nothing about its outcome was actually known until its verification
window closed on `t+2` -- days after `t`'s own issuance. A model that
merely memorizes observed values it has trained on looks skillful for
exactly this reason, not because it corrects anything.

### The correct cutoff: the verification window's close time

The training set for the run issued at time *T* is exactly the pairs with
`scorable = true` and `window_end_utc < T` (strict), across every lead,
variable and station -- never `run_date` or `target_date`. Sorting the
scorable pairs once by `(window_end_utc, station, runtime_utc, lead_day,
variable)` turns this into a single `searchsorted` prefix slice per run
(O(N log N) overall), instead of re-filtering the full pairs table at
every issuance date.

`predict` is restricted to an allow-list of guidance-side columns
(`station, run_date, runtime_utc, cycle_hour, lead_day, variable,
target_date, forecast_f, spread_f, window_start_utc, window_end_utc`) and
never sees `observed_f`, `error_f`, or any other verification-derived
column -- those are the answers a model is being scored against.

### Measured: the cheat has teeth, and the guard closes it

`test_cheating_model_gains_nothing_but_leaky_cutoff_helps` builds a
`CheatingModel` that memorizes `observed_f` keyed by `(station,
target_date, variable)` from its training pairs and predicts the
memorized value when the key is present, else falls back to raw guidance.
On 60 days of synthetic multi-station, multi-lead data:

- Under the evaluator's `window_end_utc < T` cutoff, the cheat's
  predictions are bit-identical to `RawNbmModel`'s on every row (MAE
  1.302°F both), because no training pair's target date ever reaches as
  far forward as anything the model is asked to predict.
- Under a deliberately wrong `run_date < current run_date` cutoff (built
  independently in the test, not part of the evaluator), the same cheat's
  MAE drops to 0.529°F on the identical data -- a real, substantial
  reduction, confirming the leak is not merely theoretical.

### Retraining and `trained_through`

`retrain="daily"` refits before every evaluated run date; `retrain
="monthly"` refits only on the first evaluated run date of each calendar
month (including the first evaluated date overall, even if it falls
mid-month). Between refits, the model fitted at the last retrain
continues to predict; `fit` is called exactly once per retrain date, using
that date's own cutoff. Each output row's `retrained_on` names the run
date whose fit produced it, and `trained_through` is the maximum
`window_end_utc` in that fit's training set (null if the training set was
empty, which is always true on the first evaluated date given no earlier
history). `trained_through < runtime_utc` holds for every non-null
prediction by construction of the cutoff, not as a separately-imposed
constraint.

## Baseline correction (rolling bias)

`weather_forecast_audit.baseline.BaselineModel` (issue #18) is the
transparent floor an ML challenger must beat (PRD module 7): for each
(`station`, `lead_day`, `variable`) group it estimates a rolling mean of
signed error (`forecast_f - observed_f`) from the trailing window of
scorable training pairs the walk-forward evaluator's leakage guard (#17)
allows it to see at each retrain, and subtracts that estimate from the raw
forecast. It is driven only through `walk_forward`, never called directly
in production, so the same lead-aware cutoff applies to it as to any other
model.

### The rule, and W and k

`fit` computes `anchor = max(window_end_utc)` over the whole training set
it is given, then keeps the pairs with `window_end_utc > anchor -
window_days` (strictly greater, `window_days` as a timedelta in days). For
each (`station`, `lead_day`, `variable`) group among those kept pairs,
`bias = mean(forecast_f - observed_f)` and `n_pairs = count()`. Training
pairs are sorted ascending by `window_end_utc` (the evaluator's own
contract), so the window's start is a single `searchsorted`, not a scan of
the full training set. `predict_detail` then reports, per row, `forecast_f
= raw - bias` when the row's group has `n_pairs >= min_pairs`, else the raw
forecast unchanged and `fallback = true`.

`WINDOW_DAYS = 30` and `MIN_PAIRS = 15` are named, pre-registered
constants, fixed **before** any scoring and never tuned on backfill skill
-- doing so would fit the evaluation itself:

- **W = 30** is about one month: short enough to follow the seasonal drift
  the audit exists to show (biases differ by season at the same station),
  and long enough that the standard error of the mean error is about
  2.5°F/√30 ≈ 0.45°F under independence -- below the 0.5-1.5°F biases being
  corrected.
- **k = 15** is half the window: a station with patchy data is corrected
  only when its estimate has SE ≲ 0.65°F; otherwise it falls back to raw.

Sensitivity at W = 60 is reported in the PR as exploratory only and does not
change these defaults. W = 14 was also planned, but with k = 15 it is not a
valid configuration. A group gains at most one pair per day, so a window
shorter than k can never reach k, and every group would silently fall back to
raw. `BaselineModel` now rejects `window_days < min_pairs` outright.

### Fallback

A group falls back to the raw forecast -- flagged via `fallback = true` in
`predict_detail`'s output and in `raw.model_predictions` -- whenever it has
no training-time estimate at all (absent from the fitted window, or the
training set was empty), or its `n_pairs` is below `min_pairs`. This is a
per-group decision made independently at every retrain: a group can move
between corrected and fallback across the backfill as its trailing history
grows or thins.

### Storage: `int -> predictions -> fct`, no lineage cycle

Predictions are written by `wfa predict baseline` to a new
`raw.model_predictions` table (`weather_forecast_audit.warehouse.
load_predictions`, idempotent by `(source, run_date range)`) and read back
into `fct_forecast_verification` through a new intermediate model,
`int_raw_verification_pairs` -- the `raw_nbm`-only body
`fct_forecast_verification.sql` used to be, split out unchanged -- and a
new staging model, `stg_model_predictions`, over
`raw.model_predictions`. `fct_forecast_verification` now selects
`int_raw_verification_pairs` verbatim, `union all` each prediction row
joined back to its raw pair on (`station`, `run_date`, `lead_day`,
`variable`) to inherit every observation-side column (`observed_f`, the
verification windows, `scorable`, `extreme_source`, `nbm_version`,
`cycle_regime`, ...); its own values are `source`, the corrected
`forecast_f`, and `error_f = forecast_f - observed_f` when scorable.

The Python side (`wfa predict baseline`) reads `int_raw_verification_pairs`
only, never `fct_forecast_verification` itself: that keeps the lineage
`int_raw_verification_pairs -> raw.model_predictions ->
fct_forecast_verification` acyclic, which matters once a later slice (#20)
wires this into a Dagster asset graph. No Dagster asset exists for
predictions in this slice.

## Regime boundaries

Every `fct_forecast_verification` row is tagged with two independent
regime boundaries (issue #12), because both change bias in ways that read
as unexplained drift if left untagged: an NBM version upgrade changes the
underlying model, and the archived-cycle changeover changes which run
"nearest 12Z" means.

### NBM operational versions

`dbt/seeds/nbm_versions.csv` lists every NBM operational version from the
archive start (2020-09-29) onward. Each row cites the NWS Service Change
Notice (SCN) that announced the version, following any delay chain to its
final notice. The boundary itself does not come from the notice. It
comes from the archived bulletins, because an SCN date is a plan, not a
record.

Each NBS station block's header names the version that produced it
(`KPHX    NBM V4.3 NBS GUIDANCE    4/30/2026  1300 UTC`). AWS Open Data
keeps one NBS text file per hourly run, uniform in version, so
`scripts/probe_nbm_version_headers.py` reads the first 4 KB of every
hourly run around each announced date. The boundary is the **first stable
run**: the earliest run showing the new version after which no run
reverts. The rule was fixed before any data was read. The details, raw
CSVs and results are in `docs/analysis/2026-09-27-nbm-version-headers/`.

| version | announced (SCN) | first stable run (headers) | isolated flips before it |
| --- | --- | --- | --- |
| v4.0 | 2020-09-29 12Z ([SCN 20-78](https://www.weather.gov/media/notification/pdf2/scn20-78nbm_v4_aaa.pdf)) | 2020-09-29 12Z | 0 |
| v4.1 | 2023-01-17 12Z ([SCN 22-131](https://www.weather.gov/media/notification/pdf2/scn22-131_nbm_v4.1.pdf)) | 2023-01-17 12Z | 0 |
| v4.2 | 2024-05-15 12Z ([SCN 24-41](https://www.weather.gov/media/notification/pdf_2023_24/scn24-41_nbm_v4.2.pdf)) | 2024-05-15 **11Z** | 9 |
| v4.3 | 2025-05-27 12Z ([SCN 25-34](https://www.weather.gov/media/notification/pdf_2025/nbm_v4.3_scn_aaa.pdf)) | 2025-05-27 12Z | 5 |
| v5.0 | 2026-04-30 13Z ([SCN 26-24 AAC](https://www.weather.gov/media/notification/pdf_2026/scn26-24_Updated_NBM_V5.0_aac.pdf)) | **2026-05-05 12Z** | 8 |

Three things in this table are not in any notice:

- **v5.0 took over five days after its final SCN date.** Every bulletin
  from 2026-04-30 13Z through 2026-05-05 11Z is V4.3, apart from eight
  isolated V5.0 runs. The header date matches MDL's own announcement ("On
  May 5, 2026"). The IEM archive's 2026-04-30 cycle changeover
  (`nbs_cycle_regimes.csv`) is therefore an archive change, not the v5.0
  rollout; the two regime boundaries are independent.
- **The later upgrades alternated between versions for about a day before
  settling.** Isolated new-version runs reverted to the old version. None
  of them fell on this project's canonical cycle (13Z, or 12Z from
  2026-04-30), so each canonical run carries one unambiguous version.
- **v4.0's first stable run equals the independently probed archive
  start.** The day the archive first carries `txn` for every registry
  station (`nbs_archive.csv`) is the v4.0 rollout day.

`fct_forecast_verification.nbm_version` is joined by `runtime_utc` against
this seed's half-open `[valid_from_utc, valid_to_utc)`.

### Archived cycle regimes

`dbt/seeds/nbs_cycle_regimes.csv` (issue #5/#7) is which archived NBS
cycle counts as "nearest 12Z" for a run date; see that seed and
`src/weather_forecast_audit/regimes.py` for its own documentation.
`fct_forecast_verification.cycle_regime` is joined by `run_date` against
this seed's inclusive `[valid_from, valid_to]`.

### Combined annotations

`dim_regime_boundaries` unions both seeds into one `(kind, id, starts_at,
verified, citation)` boundary list, for a future export (issue #9) to
read as time-series annotations. It does not itself build an export or
change any export schema version.

## Orchestration and partitions (issue #10)

Ingestion runs as four daily-partitioned Dagster assets, all on one
`DailyPartitionsDefinition` anchored to the pinned NBS archive start
(`regimes.load_archive_start()`, currently 2020-09-29): `raw/nbs_guidance`,
`raw/asos_hourly`, `raw/cli_daily`, and `raw/resolved_windows`.

### Each asset owns exactly its own dates

Every asset's partition key means a specific date in a specific source's own
terms -- `raw/nbs_guidance` and `raw/resolved_windows` partition by the
guidance run date (UTC), `raw/asos_hourly` by the observation date (UTC), and
`raw/cli_daily` by the station's **local** climate date. `warehouse.load_*`
and `warehouse.load_gaps` delete-then-insert exactly that date's rows before
inserting the new ones, so re-materializing a date replaces that date's rows
and never touches another date's -- literally true, not just a design
intent, because each `fetch_*` client's own `[start, end]` range semantics
already line up with its partition's date meaning (a run-date range for
guidance, a UTC observation-date range for asos, a local-date range for
cli): the partition window passes straight through to the fetch call with
no boundary adaptation.

### Why a resolved partition can be incomplete

`raw/resolved_windows` depends on `raw/nbs_guidance` by the identity mapping
and on `raw/asos_hourly` through a `TimeWindowPartitionMapping(start_offset=
-OBS_LOOKBACK_DAYS, end_offset=OBS_LOOKAHEAD_DAYS)` (1 and 4, the same
constants `pipeline.ingest_station` pads its own fetch range by). A run on
date D carries guidance out to a lead-3 min/max whose verification window
can close as late as `D+4 06Z`, so `raw/asos_hourly` for `D+4` must exist
before D's windows are fully scorable. A resolved partition for a recent D
is therefore incomplete -- some of its windows are unscorable for lack of
observations that haven't happened yet, not because anything is broken --
until `D+4`'s asos partition is materialized, at which point D must be
re-materialized to pick up the now-available observations. Scheduling that
re-materialization is issue #16's job, not this one's.

### `first_seen`: how long a gap has been open

`raw.ingest_gaps` carries a `first_seen` timestamp per `(station, source,
expected, reason)`. A gap that is still present on a later ingest keeps its
original `first_seen`; a newly-appearing gap gets the ingest's current
clock; a gap that has disappeared (the data showed up, or the reason
changed) is deleted outright, not soft-closed with an end date. This makes
`first_seen` a simple "how long has this specific gap been open" signal --
`dbt/models/marts/gap_ledger.sql` exposes it directly -- without needing a
second table to track gap history.

### One writer: runs are serialized by the slot, steps by the executor

DuckDB allows one writer process per database file. rammingspeed's single run slot serializes *runs*, but it does not serialize the *steps inside* a run. Under Dagster's default multiprocess executor, `ingest_job`'s independent raw assets (guidance, ASOS and CLI) start in parallel subprocesses. Each opens the warehouse for writing, and all but the first fail on the file lock. That is what happened in the first host run (8e53a237, 2026-09-27), where only `asos_hourly` materialized. Every job therefore runs on `in_process_executor`, set once on `Definitions` so future jobs inherit it, and a test asserts it for each job.

The tests could not see this failure. `execute_in_process` always runs steps sequentially in one process, so only a structural assertion on the configured executor catches it.

## National backfill (issue #11)

`national_backfill_sensor` (`sensors.py`) drives `ingest_job` and
`transform_job` through the whole archive -- `ARCHIVE_START` through
yesterday -- one run at a time, on rammingspeed's single shared run slot,
without a human launching each run by hand.

### The run unit

The archive is planned (`backfill.plan_units`) into calendar-month x
station-chunk units: each unit is one `ingest_job` run over a whole
calendar month (clipped to the archive/end bounds) for 30 stations
(`BACKFILL_CHUNK_SIZE`) sorted from the registry. A month matches D2's
existing ingest batching (a run already receives up to a month and fetches
each station once over the whole range); the 30-station chunk exists
because the archive has ~573 stations and ingest costs ~12s/station-month
(IEM's rate limit is 1 req/s) -- a national month in one run would take
over 6800s, and `MAX_RUNTIME_SECONDS` (600) caps every run at 600s. 30
stations keeps a unit's ingest under ~360s, leaving headroom under the cap.
Units are submitted in month-major, chunk-minor order, so every station
gets a given month before the backfill moves on to the next.

### The blackout window, and why the 600s cap is what makes it safe

rammingspeed's other tenants (oura, waga) have schedules firing every 15
minutes from 06:00 through 07:00 America/Phoenix (no DST there, so this is
a fixed UTC offset). `backfill.BLACKOUT_WINDOWS` blocks a new unit's
submission from 05:45 to 07:15 local -- 15 minutes of padding either side
of that window.

The padding's size is not arbitrary: `in_blackout` is checked against
`[now, now + horizon_s]`, where `horizon_s = MAX_RUNTIME_SECONDS + 300`.
`MAX_RUNTIME_SECONDS` is the run's own hard cap (run monitoring kills
anything longer), so a run launched right before the blackout starts is
*guaranteed* to have finished (or been killed) within 600s, plus a 300s
margin for the time between "the sensor's `RunRequest` is picked up" and
"the run actually starts occupying the slot" (container/process start).
That is the sense in which the 600s cap is a safety property here, not
just a runtime budget: without it, a run could still be holding the slot
when oura/waga need it, and the backfill would be blocking production
schedules instead of yielding to them.

### Retry and halt

Each unit gets up to 3 attempts (`MAX_BACKFILL_ATTEMPTS`). A `FAILURE` or
`CANCELED` run retries the same unit with the attempt incremented; a third
failure halts the whole backfill rather than silently skipping a unit or
retrying forever. A halted sensor keeps returning the identical `Halt`
decision, and keeps writing back the identical (unchanged) cursor, on
every subsequent tick -- it will never resubmit on its own. Unhalting
requires a human to look at the failed run, fix whatever broke, and reset
the cursor (see below); there is no automatic recovery, by design, because
a `FAILURE` that recurs three times against the same station/month is
worth a look, not a fourth blind retry.

The transform job (`dbt build`, run once after every ingest unit succeeds)
follows the same shape: its `FAILURE` halts, its `SUCCESS` completes the
backfill.

A submitted run that never appears at all (`last_run_status` reads `None`
-- no run tagged with this unit/attempt/generation exists yet) also halts,
but only after `not_found_timeout_s` (15 minutes) has passed since
`submitted_at`, not immediately: a run can legitimately take a moment to
show up. Dagster's sensor daemon dedupes `RunRequest`s by `run_key`, scoped
to the sensor (`dagster/_daemon/sensor.py`'s `fetch_existing_runs` and
`_get_or_create_sensor_run`): once a run_key has been used, the daemon will
never mint a *second* run under it, even if the first one never actually
launched (a daemon crash between building the `RunRequest` and creating the
run, for instance). Without the timeout, a cursor stuck on a never-created
run would `Wait("run in flight")` forever, silently, with no run to look
at and no way to tell the difference from a normal in-progress run. The
halt message says so explicitly and points at the fix (below).

### The frozen plan end

`plan_units`'s `end` argument -- "yesterday" at the sensor's first
evaluation -- is computed once and stored in the cursor from then on, not
recomputed on every tick. Without freezing it, a national backfill running
for weeks would see its own plan grow by one day (and therefore its unit
count and every unit's index) every time the clock ticks past midnight
UTC, which would either resubmit units whose indices shifted or leave a
"complete" backfill perpetually one day short. Freezing the end date at
evaluation time means the plan is a fixed, finite list from the first tick
onward: N units, then the transform, then done.

### Starting, stopping, and resetting the sensor

The sensor's `default_status` is `STOPPED`: nothing runs until a human
starts `national_backfill_sensor` from the Dagster UI (Automation ->
Sensors) or `dagster sensor start national_backfill_sensor`. Stopping it
(`dagster sensor stop national_backfill_sensor`) simply pauses evaluation;
the cursor is untouched, so starting it again resumes exactly where it
left off, including replaying an unresolved `Wait` or a `Halt`.

To reset a halted (or otherwise stuck) backfill entirely, delete the
sensor's cursor from the Dagster UI (the sensor's page has a "Reset
cursor" action) or via `dagster instance` tooling. The next evaluation
then starts over from a fresh cursor: a newly frozen plan end, unit 0,
attempt 1, generation 0.

To retry a specific halted unit/attempt *without* restarting the whole
backfill, edit the cursor JSON by hand (the envelope is `{"plan_end": ...,
"state": {"next_index", "attempt", "last_run_id", "transform_requested",
"submitted_at", "generation"}}`): keep `next_index`, set `attempt` to `1`,
set `last_run_id` and `submitted_at` to `null`, and **increment
`generation`**. Set `attempt` back to 1 when the halt was a bug that has
since been fixed and deployed: the unit then gets its normal three tries,
whereas leaving it at 3 gives it one. (The sensor evaluates once a minute,
so the last tick can still show the old `HALTED` message for up to a minute
after the edit; check that the newest tick names the new generation before
concluding the edit did nothing.) The generation bump is not optional -- because the sensor
daemon dedupes by `run_key` (`f"{unit_id}-a{attempt}-g{generation}"`,
`f"transform-g{generation}"` for the transform), clearing `last_run_id`
alone would make the next tick recompute the *identical* run_key the
backfill was already stuck on, and the daemon would just find (or silently
skip past) that same never-materialized run again. This is a deliberate,
manual override; the sensor itself never bumps `generation` on its own.

### Risk: the transform run may exceed the 600s cap

`transform_job` is a full `dbt build` over the whole raw archive (D4), and
it runs under the same `dagster/max_runtime: "600"` tag as every other job
here -- run monitoring kills it at 600s exactly like an ingest unit. That
budget was sized against ingest (`~12s/station-month`, chunked to fit), not
against a `dbt build` over six years of national data on rammingspeed's
no-AVX2 host; at that scale it may legitimately take longer than 600s,
and there is currently no measurement of its actual duration to say
otherwise. If the transform run is killed for exceeding the cap, the
sensor halts (per the transform's own `FAILURE`/`CANCELED` rule above) --
it does not retry, because a killed `dbt build` is exactly the kind of
failure worth a human look, not a blind resubmission.

The transform job's duration at national scale is not yet known; issue #11
asks for it to be recorded once the backfill actually reaches that step.
Until then, treat a transform halt as expected rather than surprising, and
run it by hand instead (`dagster job execute -j transform_job`, or `wfa`'s
own `dbt build` invocation) **outside the blackout window**, since a
manual run occupies the same shared slot the sensor is respecting.

## Published numbers

`weather_forecast_audit.export` (issue #9) turns `scoring.score` output into
the site's per-city summary and stat table. A published bias number always
comes from a **lead-1** slice: the site's plain-language summary and its
"day-ahead forecasts" wording never draw from lead 2 or lead 3, even when a
longer lead's bias is larger, so the one number a visitor reads without
statistics training is the least confounded by limitation 1 above (pooled
cross-lead correlation).

A city's summary claims a direction and a number only for a slice whose
bias interval **excludes 0** (`no_detectable_bias = false`) with **enough
distinct bootstrap blocks** (`min_sample_flag = false`, limitation 3); the per-city stat
table applies the identical rule to every cell, never just the summary
sentence. Every other slice reads as "no detectable bias" or "too few
days," never as a number that happens to round toward zero.

Both rules -- lead-1 only, and significance gated on both flags -- describe
what the site is allowed to say. Both statistical inputs behind them are
now measured: the block length (`BLOCK_DAYS = 14`, limitation 2) and the
sample floor (`MIN_SAMPLE_BLOCKS = 21`, limitation 3). What remains open
before a public launch (#16) is operational: the backfill that gives
per-season cells their three years (#11), and the nightly schedule.
