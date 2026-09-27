# Pre-registration: the bootstrap block length (#31)

Committed and pushed before any sample data was fetched. The rule below is fixed. The analysis reports against it and does not tune it.

## Question

How many consecutive issuance dates should one bootstrap block hold (`scoring.BLOCK_DAYS`) so that the published 95% bias intervals keep close to their nominal coverage on real NBM errors?

## What prompted this

`scoring.BLOCK_DAYS = 1` treats consecutive issuance dates as independent. #8's synthetic test shows what that costs under persistence. With an AR(1) day effect at ρ = 0.8, nominal 95% intervals cover the true bias in 48% of simulations with 1-day blocks and 82% with 7-day blocks. Whether real errors persist like that is unknown, and it is not a default to guess (`docs/methodology.md`, "Scoring and uncertainty", limitation 2).

## Sample (fixed)

- **Stations (12):** #6's set: KPHX, KSFO, KSEA, KDEN, KSLC, KBIS, KOKC, KMSP, KORD, KATL, KMIA, KBOS. A station that returns no data is reported and dropped, not replaced.
- **Issuance dates:** `run_date` 2025-01-01 through 2025-12-31. Observations are ingested through 2026-01-04, so every lead of the last run can verify.
- **Rows:** `fct_forecast_verification`, `source = 'raw_nbm'`, `scorable` true, `error_f` not null. Leads are whatever the mart carries: max leads 1–2 and min leads 1–3, since NBS never carries a lead-3 max. That gives **five (variable, lead) series**.
- **Build:** through the pipeline as it stands on `main` at the time of fetching (`wfa init-db`, `wfa ingest` per station, then `dbt build`), into a throwaway DuckDB file. The extracted rows are committed with the results so the analysis re-runs without the warehouse.
- **Known in-sample confounds, reported and not corrected:** the NBM v4.2 → v4.3 upgrade (2025-05-27 12Z) and the seasonal cycle. A level shift or a slow seasonal swing in the mean error raises the autocorrelation and so lengthens the estimated block. That errs toward wider intervals, the safe direction for a published bias claim.

## Series

1. **Pooled (primary, the issue's spec):** for each (variable, lead), the mean of `error_f` across stations per `run_date`. A date with fewer than 6 of the 12 stations scorable is missing. Missing dates are dropped and the remaining dates kept in order. The count of missing dates is reported per series, and more than 10% missing in any series is reported as a deviation.
2. **Per station (added protection):** for each (station, variable, lead), `error_f` per `run_date`, with missing dates dropped and counted the same way. A station series with fewer than 300 dates is excluded from the median below and reported.

**Why the per-station series binds as well.** City cards bootstrap one station's dates, not the cross-station mean. Averaging across regions can dilute a regional regime, such as a Southwest heat dome that Boston never sees, so a pooled estimate alone could under-state the persistence a single city's interval faces. Adding the per-station median can only lengthen blocks.

## Estimator

- **Block length:** Politis & White (2004) automatic block-length selection with the Patton, Politis & White (2009) correction, taking the **circular block bootstrap** estimate. It uses the `arch` package's `arch.bootstrap.optimal_block_length`, the reference implementation, and is not hand-rolled.
- **Scheme mismatch, stated up front:** `scoring.py` resamples non-overlapping calendar blocks (`epoch day // block_days`), not circular blocks. For a non-overlapping block bootstrap the asymptotically optimal length is about (2/3)^(1/3) ≈ 0.87 times the circular one (Lahiri 1999), so the circular estimate is slightly long, the conservative side. The coverage replay below tests the scheme actually used.
- **Secondary, reported and not binding:** the lag 1–14 sample autocorrelation of each pooled series, with ±1.96/√n reference bands.

## Decision rule

1. For each (variable, lead):
   - `b_pooled` is the pooled series' circular estimate.
   - `b_station` is the median of the per-station circular estimates. With an even count, it is the upper median.
   - `b(variable, lead) = max(b_pooled, b_station)`.
2. `BLOCK_DAYS = ceil(max over the five (variable, lead) of b)`, with a floor of 1.
3. **Cap at 14.** If the ceiling exceeds 14, the default is **not** silently capped. The result is reported, `BLOCK_DAYS` is left at 1, #16 stays blocked, and the choice goes back to Charles. The deseasonalized estimate below is reported to inform that revisit, but it does not decide it.
4. **Reported, not binding:** the same estimates on each pooled series after subtracting a centered 31-day rolling mean, which removes the seasonal swing. This shows how much of the block length the season causes.

## Coverage replay (the acceptance check)

For each of **ten replay series**:
- the five pooled series;
- for each (variable, lead), the station series whose estimate is the upper median in rule 1 (the station that sets `b_station`).

Steps for each replay series:

1. **Fit an AR(p) model** by Yule–Walker. Choose p from 0–7 by AIC; p = 0 is white noise.
2. **Simulate 400 series** from the fitted model. Each is as long as the observed series, with the observed mean as the true bias. Discard a burn-in of 100.
3. **Score each series** with `scoring.score`, using `by=()`, `n_boot=500`, the module's `CI_LEVEL` (0.95) and `block_days = BLOCK_DAYS`. Each simulated value is one row on consecutive `run_date`s starting 2025-01-01, so blocks align exactly as they would on real data.
4. **Coverage** is the share of the 400 intervals that contain the true bias.

**Pass:** coverage ≥ 0.90 on all ten replay series at the chosen `BLOCK_DAYS`.

**Fallback if it fails:** raise `BLOCK_DAYS` one day at a time and re-run the replay until all ten pass, up to 14. The first passing value is the default, and the gap between it and the rule's value is reported. If nothing up to 14 passes, handle it as the cap case in rule 3.

The same replay also runs at `BLOCK_DAYS = 1`, to report the coverage the current default actually has on these fitted models.

## Seeds and determinism

- Simulation seed `20260927`. Each replay series draws from its own stream: `numpy.random.SeedSequence(20260927).spawn(10)`, in the order listed above.
- The bootstrap seed is `scoring.SEED`.
- The analysis script is deterministic given the committed extract. Its SHA-256 is recorded in `RESULTS.md`.

## Outputs

- `RESULTS.md`: per-series `n`, missing dates, `b_pooled`, `b_station`, the per-station estimates, the ACF table, the deseasonalized estimates, the fitted AR orders and coefficients, coverage at the chosen value and at 1, and the chosen `BLOCK_DAYS` under the rule.
- `results/*.csv`: the numbers behind those tables.
- If the rule yields a value (rule 2 without the rule 3 cap), `scoring.BLOCK_DAYS` is set to it. A unit test pins the constant and cites this directory, and `docs/methodology.md` (limitation 2) is rewritten with the measured value.
