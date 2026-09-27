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
`BLOCK_DAYS = 1` (the current default) resamples one issuance date at a
time; larger values group consecutive calendar dates into one resampling
unit. The bootstrap itself operates on per-block sums (`Σ error`, `Σ
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

`min_sample_flag` fires when a slice has fewer than `MIN_SAMPLE_DATES`
(default 30) **distinct issuance dates**, not rows. Dates, not
verification rows, are the bootstrap's independent unit -- a slice built
from 30 stations reporting on a single date is not more trustworthy than
one station reporting on 30 dates, and counting rows would make it look
that way.

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

### Limitation 2: `BLOCK_DAYS = 1` is not a measured default

`BLOCK_DAYS = 1` treats consecutive issuance dates as independent of each
other. Forecast errors persist across multi-day weather regimes, since a
stuck upper-level pattern can bias guidance the same way for a week, so
per-date intervals are too narrow whenever that persistence is real. How
much it matters is not small:
`test_block_days_seven_beats_block_days_one_on_ar1_data` gives the shared
day effect AR(1) persistence with ρ = 0.8, and nominal 95% intervals then
cover the true bias in 48% of simulations with 1-day blocks and 82% with
7-day blocks. That test proves the knob works, not that 7 is the right
number for real data. The default block length used for any published
bias claim must come from a **pre-registered measurement on real
verification rows**: the persistence of the daily cross-station mean
error, by variable and lead. That measurement is a follow-up issue and
blocks the public launch.

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

## Published numbers

`weather_forecast_audit.export` (issue #9) turns `scoring.score` output into
the site's per-city summary and stat table. A published bias number always
comes from a **lead-1** slice: the site's plain-language summary and its
"day-ahead forecasts" wording never draw from lead 2 or lead 3, even when a
longer lead's bias is larger, so the one number a visitor reads without
statistics training is the least confounded by limitation 1 above (pooled
cross-lead correlation) and by the still-unmeasured block length in
limitation 2.

A city's summary claims a direction and a number only for a slice whose
bias interval **excludes 0** (`no_detectable_bias = false`) with **enough
distinct issuance dates** (`min_sample_flag = false`); the per-city stat
table applies the identical rule to every cell, never just the summary
sentence. Every other slice reads as "no detectable bias" or "too few
days," never as a number that happens to round toward zero.

Both rules -- lead-1 only, and significance gated on both flags -- describe
what the site is allowed to say **today**, on `BLOCK_DAYS = 1`. That
default is not the pre-registered measurement limitation 2 calls for: the
persistence of the daily cross-station mean error is still unmeasured, and
until that measurement lands (issue #31), no bias claim from this export is
a validated public finding, only a conservative reading of an
under-characterized interval. Publishing the site (PRD milestone M4) is not
a substitute for that measurement.
