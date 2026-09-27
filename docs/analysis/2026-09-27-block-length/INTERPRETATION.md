**Verdict: `BLOCK_DAYS = 14`, under the rule as registered, with the coverage replay passing.** The binding series is the pooled NBM max lead-2 error, estimated at 13.18. Every one of the ten replay series covers at least 0.90 at 14; the worst is pooled min lead 2 at 0.9075. The old default of 1 covers as little as 0.6475 (pooled max lead 2), so the intervals the scoring module produced before this change were far too narrow for a public bias claim.

**Provenance and order.**
- `PREREG.md` was committed as `f328cc6` at 2026-09-27 11:53:52 −07:00 and pushed to `origin/feat/block-length-measurement` before any fetch.
- The fetch started at 18:54:11Z (11:54:11 −07:00). It ran from a detached worktree at `f328cc6`, so the pipeline code was `main` at `943686f` plus the pre-registration.
- All 12 stations returned data (`wfa ingest` for 2025-01-01 to 2026-01-04), and `dbt build` passed 82 of 82. The extract holds 21,900 rows, 1,825 per station.
- None of the five pooled series is missing a date. Per-station series run 333–364 dates, so none falls below the 300-date floor.
- The analysis script's fail-loud hardening (`ca2e32b`) was committed before the fetch finished and before any data was read. No pre-registered rule was changed after data arrived.

**What the number rests on.**
1. **It sits exactly on the cap.** Rule 3 fires only when the ceiling exceeds 14, so 14 stands. A slightly more persistent year would have produced "report and revisit", not a larger default.
2. **The seasonal cycle accounts for about 4.5 of the 14 days.** With a centered 31-day mean removed, the largest pooled estimate falls from 13.18 to 8.66. The pre-registration predicted this confound and its direction (longer blocks, wider intervals), and chose not to correct for it.
3. **The replay certifies only short-memory persistence.** The fitted models are AR(1) to AR(6), with lag-1 coefficients of 0.20–0.51, and cannot represent a seasonal swing. The seasonal part is covered by point 2's conservatism, not by the replay.
4. **Single stations can persist longer than the median.** KSFO's own max estimates are 14.4 (lead 1) and 14.3 (lead 2), probably marine-layer regimes. The upper median keeps one station from setting the national default, so a KSFO-only interval is somewhat narrower than that station's persistence warrants.

**Post-hoc, not pre-registered (`posthoc_short_slices.py`, `results/posthoc_short_slices.csv`): short slices under-cover at every block length.** The same ten fitted models were replayed at the slice lengths the site actually publishes. The table shows the range over the ten series.

| slice length | 1-day blocks | 7-day | 14-day |
| --- | --- | --- | --- |
| 365 dates | 0.68–0.89 | 0.89–0.95 | 0.905–0.95 |
| 90 dates (one season) | 0.67–0.88 | 0.85–0.91 | 0.83–0.89 |
| 30 dates (`MIN_SAMPLE_DATES`) | 0.65–0.82 | 0.78–0.84 | 0.73–0.80 |

- A percentile bootstrap over only a few blocks gives intervals that are too narrow whatever the block length. At 14 days, a 30-date slice holds three blocks.
- The existing iid-day-effect test (`test_date_block_coverage_beats_row_level`, 60 dates) shows the same thing with no persistence at all: 0.870 at 14-day blocks, against its 0.90 floor at 1-day blocks.
- So 14 is the right block length for full-year claims. The publishability floor, which counts dates and not blocks, and the interval method for season-length slices are a separate, open decision: #37, which blocks #16. This result informed nothing in the decision rule above, which was settled first.
