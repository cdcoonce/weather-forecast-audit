**Verdict: `MIN_SAMPLE_BLOCKS = 21`, under the rule as registered.** At 14-day blocks, 21 blocks is the smallest count at which every replayed shape with at least that many blocks keeps all ten fitted models at 0.90 coverage or better. The 270-date contiguous slice (21 blocks) is worst at 0.904, and the three-season slice (22 blocks) at 0.913. Below the floor, 180 contiguous dates (14 blocks) fall to 0.879 and two seasons (14 blocks) to 0.885.

**Rule 2 fired once, as designed.** The 540-date contiguous shape (40 blocks) read 0.894 on pooled max lead 2, while shapes with fewer blocks passed. That model was re-run at 2000 sims on the pre-registered fresh seed stream and read 0.912. Both numbers are in `results/coverage.csv`. Without the re-run the rule would have given 46, set by one Monte Carlo dip 0.006 under the bar, about 0.6 standard errors at 1000 sims. The noise protocol was written before the replay for exactly this case.

**Provenance and order.**
- `PREREG.md` was committed as `cec44f3` at 2026-09-27 12:49:27 −07:00 and pushed before `replay.py` existed.
- The replay reads only #31's committed `ar_fits.csv`, so no new data was fetched.
- The helpers are unit-tested, including a bit-identity test proving that the refactored `replay_coverage` reproduces #31's replay stream exactly, so #31's results stand unchanged.

**What the floor does to the site (reported, not binding; `flag_share.py`, `results/flag_share.csv`).**
- On the committed #31 extract (12 stations, one year), the export's cells are `(station, variable, lead_day, season)`, and each holds one season: 6–9 blocks.
- Under the old 30-date floor, none of the 240 cells was sample-flagged, and **52 of them, 19 of the 96 lead-1 cells, would have published a bias claim** from a single season's interval. The replay puts that interval's true coverage near 0.84–0.88.
- Under the new floor, all 240 are flagged.
- The host warehouse holds about six issuance dates (runs `3ec46c56` and `875f2cc5`; no schedules exist), so every cell there is flagged under either floor.
- A per-season cell needs three years of that season, which the national backfill (#11, archive from 2020-09-29) provides.

**Limits.**
- The replay inherits #31's models: ten AR(p) fits from one year and 12 stations, with no seasonal component. A station whose errors persist much longer than those fits (KSFO's own estimate was about 14.4 days) is under-protected by any floor derived from them.
- Season-shaped slices were simulated as independent yearly segments. At a 274-day gap the fitted autocorrelations are effectively zero, but a multi-year regime, such as a model-version bias that persists across seasons, is not represented.
- Option 3 (a studentized or t interval) was not tested. It becomes worth revisiting only if, after the backfill, the floor still leaves a large share of lead-1 cells unpublishable.
