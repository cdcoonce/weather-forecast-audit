# Pre-registration: a block-aware minimum sample (#37)

Committed before the replay below was run. The rule is fixed, and the analysis reports against it without tuning it. No new data is fetched: the replay reuses the ten AR models fitted and committed in #31 (`docs/analysis/2026-09-27-block-length/results/ar_fits.csv`).

## Question

How many distinct bootstrap blocks must a slice hold before its nominal 95% bias interval covers the true bias at least 90% of the time? The answer replaces `scoring.MIN_SAMPLE_DATES = 30` as the threshold for `min_sample_flag`.

## What prompted this

#31 set `BLOCK_DAYS = 14` and certified it on full-year series (365 dates, 27 blocks), with every one of ten series covering at least 0.905. A post-hoc replay (`docs/analysis/2026-09-27-block-length/posthoc_short_slices.py`) found:
- a 90-date slice covers 0.83–0.89;
- a 30-date slice covers 0.73–0.80;
- an iid series of 60 dates at 14-day blocks covers 0.870.

A percentile bootstrap over only a few blocks is too narrow on its own, so the floor has to be counted in blocks, not dates. The export publishes `(station, variable, lead_day, season)` slices, and `season` pools every archive year present, so a season slice is several disjoint segments of about 90 dates.

## Options considered

1. **A block-count floor (chosen).** It is one flag change, and it reads the quantity that actually governs coverage.
2. **Pooling seasons across years.** This is not a separate change: `scoring.score` already derives `season` from `target_date` month and pools years. The national backfill (#11; archive from 2020-09-29) is what gives each season slice several years. The replay below includes that shape.
3. **A studentized or t interval for few-block slices (rejected for now).** It is a new interval method with its own coverage risk, and it is unnecessary once season slices span several years. It is revisited only if the chosen floor leaves most per-season cells unpublishable after the backfill.

## Replay (fixed)

- **Models:** the ten `ArFit`s in #31's `results/ar_fits.csv`, in file order: five pooled series, then five upper-median station series.
- **Block length:** `scoring.BLOCK_DAYS` = 14.
- **Shapes:**
  - **Contiguous:** n consecutive issuance dates starting 2025-01-01, with n ∈ {60, 90, 120, 180, 270, 365, 540, 730}.
  - **Season-shaped:** Y segments, one per year for Y ∈ {1, 2, 3, 4, 5, 6}. Each segment spans the JJA target dates, June 1 to August 31 (92 dates), in years 2021, 2022, and so on. Each segment is simulated as an independent draw from the model: segments sit about 270 days apart, and the fitted autocorrelations are negligible at that range. The dates are real calendar dates, so the block ids (`epoch day // 14`) split at segment edges exactly as real data would.
- **Per shape and model:** 1000 simulated slices, each scored with `scoring.score(by=(), n_boot=500, block_days=14)` and the module's `CI_LEVEL` and `SEED`. The shape's block count is the number of distinct block ids in the slice. It is deterministic given the dates, and it is recorded.
- **Coverage** is the share of the 1000 intervals containing the model's mean.
- **Seeds:** `numpy.random.SeedSequence(20260929)`, spawning one child per (shape, model) in the order the shapes are listed above (contiguous first, ascending n; then season, ascending Y), models in file order.

## Decision rule

1. **A shape passes** when all ten models cover at least 0.90 at that shape.
2. **Noise protocol, fixed now.** A shape with 1000 sims has a standard error of about 0.0095 near 0.90. Among many shapes, a lone failure that is a Monte Carlo artifact is expected. So a shape that fails while a shape with fewer blocks passes is re-run once for its failing models only. The re-run uses 2000 sims on a fresh seed child (`SeedSequence(20260930)`, spawned the same way), and the re-run's coverage replaces the original for that model. Both numbers are reported.
3. **`MIN_SAMPLE_BLOCKS`** is the smallest block count k such that every shape in the grid with at least k blocks passes, after step 2.
4. **If no k exists,** because the largest shape still fails after step 2, the flag is not changed. The result is reported, and #16 stays blocked.
5. **The flag becomes** `min_sample_flag = n_blocks < MIN_SAMPLE_BLOCKS`, where `n_blocks` is the slice's distinct block count, now an output column. `MIN_SAMPLE_DATES` is removed rather than kept as a second threshold, because two floors on different units would make the flag hard to explain.

## Reported, not binding

- Coverage at every shape, together with each shape's block count.
- After the rule is applied, the share of the live warehouse's `(station, variable, lead, season)` slices that would be flagged. Before the backfill (#11) this is expected to be nearly all of them, which is the honest state.

## Outputs

`RESULTS.md` (generated) plus a hand-written `INTERPRETATION.md` included verbatim, `results/coverage.csv`, and the replay script. If rule 3 yields a value, `scoring.MIN_SAMPLE_BLOCKS` is set to it. A test then pins it with this citation, and `docs/methodology.md` limitation 3 is rewritten.
