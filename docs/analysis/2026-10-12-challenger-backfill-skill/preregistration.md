# Challenger pre-registration (written 2026-10-12, before any challenger or full-backfill baseline skill is computed)

Status: the rules below are fixed BEFORE the full-backfill baseline and challenger evaluations are run. Nothing here may be changed after reading a skill number. A change after the fact is a new, labelled, exploratory analysis, not a replacement.

## Pre-registered model constants (never tuned on evaluation skill)
- Model: LightGBM regressor of the signed error `forecast_f - observed_f` (units F). Prediction = `forecast_f - predicted_error`.
- Objective `regression_l1` (the reported metric, skill, is an MAE ratio; the objective matches it).
- n_estimators 200, learning_rate 0.05, num_leaves 31, min_data_in_leaf 500, feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambda_l2 10.0, max_bin 63.
- Reproducibility: deterministic=True, force_row_wise=True, num_threads=1, seed=42 (the #4 platform-proof settings).
- Retrain: monthly, expanding window, all scorable pairs with window_end_utc < issuance (the evaluator's guard).
- Minimum training rows: 20,000 scorable pairs; below it every row falls back to raw NBM, flagged `fallback`.
- Features at issuance (no dewpoint, wind or sky: not ingested; re-fetching the archive for them is out of scope): forecast_f (txn), spread_f (xnd), tmp_f, cycle_hour, lead_day, variable (categorical), day-of-year sine and cosine of target_date, station latitude, longitude, elevation_m, coastal_flag, climate_region (categorical), and lagged errors for the same (station, lead_day, variable) known at issuance: latest error, mean of the last 3 and the last 7 errors, count of pairs in the last 30 days.

## Evaluation rule (pre-registered)
- Window: every run date with a resolved verification pair, 2020-09-29 to the data end, scored on `scorable` pairs.
- Comparators: raw NBM (the bar) and the rolling-bias baseline (W=30, k=15, the merged constants), both over the same matched keys.
- Primary statistic: skill = 1 - sum|err_challenger| / sum|err_raw|, per variable x lead (5 cells), lead-1 slice is the only one the published site may quote; date-block bootstrap CI with BLOCK_DAYS = 14, n_boot = 2000, seed as in `scripts/score_sources.py`.
- Reported verdict per cell: positive (CI excludes 0 and skill > 0), indistinguishable (CI includes 0), negative (CI excludes 0 and skill < 0); `min_sample_flag` cells are reported as too few days. The result is reported whatever the sign.
- Secondary: skill against the baseline with the same CI machinery; challenger fallback share by month.
- Warm-up exclusion: the first 12 months after 2020-09-29 are reported separately (the model has little to train on) and are included in the headline too; both numbers are shown, the headline is the all-dates one.
- Exploratory only, never used to change any constant above: a 60-day baseline window, a larger n_estimators, per-lead models.

## What would be a defect, not a result
- A skill computed on keys where challenger and reference have different row sets (the evaluation must pair on identical matched keys).
- Any feature built from a pair whose window_end_utc is at or after the run's issuance time.
- A run that mixes rows from two builds of the model constants.
