# Challenger backfill skill (issue #19)

The LightGBM challenger, scored over the whole national backfill against raw NBM (the bar) and against the rolling-bias baseline, under the evaluation rule pre-registered in `preregistration.md` before any challenger skill was computed. Reported whatever the sign.

## Provenance

- Code: the branch head of this change (`wfa predict challenger`, constants as in `docs/methodology.md` "Challenger correction (LightGBM)", never tuned).
- Data: a byte copy of the production warehouse after the duplicate-key repair and the first green transform (1,491,349,504 bytes, sha256 `b7eaeb82...a0868b`, verified against the server file), 573 stations, run dates 2020-09-29 to 2026-09-26, 6,262,125 matched keys for raw NBM, baseline and challenger alike.
- Run: on a laptop (16 GB, load average above 150 from other sessions, single LightGBM thread), `wfa predict challenger --start 2020-09-29 --end 2026-09-26`: 73 monthly retrains, 71 fitted (the first two months are below the 20,000-pair floor), fit time 3,316.65 s in total, 54.9 s median, 99.6 s maximum, wall 3,798.9 s, peak resident memory 3.80 GB. Fallback share 1.5%, all of it in 2020-09 and 2020-10.
- Retrain on rammingspeed (2026-10-12, a throwaway container from the PR branch on a copy of the warehouse, `--memory=5g --cpus=2`, `nice 10`, otherwise idle host): **one monthly retrain at the largest size (the 2026-09-01 retrain, about 5.5 million training pairs) took 148.3 s to fit and 163.3 s wall including the data load, with a peak resident set of 3.96 GB**, inside the 600 s run cap and inside the 5 GiB limit. Its 74,490 predictions are bit-identical to the laptop run's for the same dates (count, sum and a hash of every forecast). **The full 73-retrain walk did not fit that limit:** the container was OOM-killed after 5,898 s at 5.15 GB resident with nothing written (the CLI writes at the end), so the whole-archive backfill evaluation was run on a laptop, not on the server. The live path (one retrain a month, one predict a day) is the single-retrain case measured here. Where the walk accumulates memory was not diagnosed.

## Result against raw NBM (headline, all dates)

| variable | lead | n | challenger MAE | raw NBM MAE | skill (95% CI) | verdict |
|---|---|---|---|---|---|---|
| max | 1 | 1,124,572 | 2.077 | 2.146 | 0.032 [0.026, 0.039] | positive |
| max | 2 | 1,084,852 | 2.385 | 2.444 | 0.024 [0.018, 0.030] | positive |
| min | 1 | 1,165,899 | 2.296 | 2.335 | 0.016 [0.013, 0.020] | positive |
| min | 2 | 1,125,598 | 2.475 | 2.509 | 0.014 [0.010, 0.017] | positive |
| min | 3 | 1,085,834 | 2.695 | 2.722 | 0.010 [0.007, 0.013] | positive |

Date-block bootstrap, 14-day blocks, 2,000 resamples. Skill is 1 - sum|err| / sum|err_raw| on identical matched keys. Against the rolling-bias baseline the challenger's skill is 0.037 to 0.062 per cell (`score-vs-baseline.md`); the baseline itself loses to raw NBM in every cell (-0.021 to -0.039, reported on #18).

## What the headline does not say

- **The first 12 months are not distinguishable from raw NBM.** On run dates before 2021-09-29 the skill is +0.011 and +0.009 on highs (leads 1 and 2, intervals include 0) and -0.009, -0.006 and -0.007 on lows (intervals include 0). After the warm-up every cell is clearly positive: highs 0.037 and 0.027, lows 0.022, 0.018 and 0.013 (`extras-output.txt`). The headline above includes the warm-up, as pre-registered, so it understates the later skill.
- **The gain is small: 1 to 3% of MAE.** Highs gain most (3.2% at lead 1) and the gain shrinks with lead on lows.
- **The challenger is warm-biased on highs: +0.23 F (CI [0.16, 0.30])**, against raw NBM's -0.09 F. The objective is L1 (the metric is MAE), so the model targets the conditional median error, not the mean. Lows are near zero (+0.02 F, interval includes 0). This does not change what the audit says about NBM; it matters for anyone who publishes the challenger's own bias.
- **Lagged-error features assume an observed value is known when its window closes.** The backfill treats a pair as known at `window_end_utc`; a CLI-derived value that publishes later would not be. The same assumption sits under the baseline.
- **Exploratory, not pre-registered:** by calendar year the sign holds in every year from 2021 (2021 highs have a lower bound at -0.0005 and -0.0023; 2026 is a partial year flagged below the sample floor). See `extras-output.txt`.
- **Absent by design:** dewpoint, wind and sky are not ingested, so they are not features (re-fetching the archive for them is out of scope).

## Files

- `preregistration.md`: the rules fixed before any skill was read.
- `score-vs-raw_nbm.md`, `score-vs-baseline.md`: `scripts/score_sources.py` output (default reference and `--reference baseline`).
- `extras.py`, `extras-output.txt`: the pre-registered warm-up split and monthly fallback share, plus the by-year table labelled exploratory.
