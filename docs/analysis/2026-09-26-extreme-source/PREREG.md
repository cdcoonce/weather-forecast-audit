# Pre-registration: choosing the observed-extreme source (#6)

Committed before any multi-station data was fetched. The decision rule below is fixed; the analysis reports against it and does not tune it.

## Question

Which observation source should define the observed max/min over NBM's UTC verification windows (max `[D 12Z, D+1 06Z)`, min `[D 00Z, D 18Z)`)?

- **(a) hourly**: max/min of routine and special `tmpf` readings in the window (the tracer's v1 method).
- **(b) metar_6h**: the METAR 6-hour maximum (`1snTTT`) and minimum (`2snTTT`) remark groups from the synoptic reports. Three consecutive groups tile each window: max uses the periods ending 18Z D, 00Z D+1 and 06Z D+1; min uses the periods ending 06Z, 12Z and 18Z of D.
- **(c) CLI**: the official daily high/low. This is the reference only, because its day is local-standard midnight to midnight and not the NBM window.

## What prompted this

On KPHX 2023-07-15, (b) gave 118.0°F max and 91.9°F min; CLI gave 118 and 92; (a) gave 117 and 93. This is one station-day and decides nothing on its own.

## Sample (fixed)

- **Stations (12):** KPHX, KSFO, KSEA, KDEN, KSLC, KBIS, KOKC, KMSP, KORD, KATL, KMIA, KBOS. They cover desert, Pacific coast, the Pacific Northwest, mountain and high-basin, northern and southern plains, the upper Midwest, the Great Lakes, the humid Southeast, subtropical coast, and the Northeast coast.
- **Period:** windows with target dates 2025-01-01 through 2025-12-31.
- A station that returns no data is reported and dropped, not replaced.

## Metrics

For each station × variable (max, min) × meteorological season (DJF, MAM, JJA, SON):

1. **Tiling rate T:** the share of windows with all three required groups present.
2. **Hourly sampling error** on tiled windows: mean, median, and 5th/95th percentiles of (a − b) in °F. The expected sign is negative for max and positive for min.
3. **Agreement with CLI** on tiled windows: mean(x − CLI) and the share of days with |round(x) − CLI| = 0 and ≤ 1, for x ∈ {a, b}. The window mismatch adds noise to both sources equally, so the comparison between (a) and (b) is the informative part.
4. **Outliers:** the count of windows with |a − b| > 10°F, listed for inspection and not dropped.

## Decision rule

1. **Primary source.** Adopt (b) `metar_6h` if pooled T ≥ 0.90 for both variables **and** |mean(b − CLI)| < |mean(a − CLI)| for both variables. Otherwise (a) stays primary and a measured correction is designed in a follow-up. No bias numbers are published until then.
2. **Fallback for windows (b) cannot tile**, when (b) is primary:
   - If pooled T ≥ 0.95 and every station has T ≥ 0.85, the window is **unscorable**, with `extreme_source = 'none'`. Mixing sources would reintroduce the sampling bias exactly at the stations with patchy reports.
   - Otherwise, the fallback is hourly plus an additive correction. The correction is mean(b − a) per variable × season, estimated on tiled windows, and recorded as `extreme_source = 'hourly_corrected'`. This applies only if its leave-one-station-out MAE against (b) is ≤ 0.75°F. If it is higher, the window is unscorable, and stations with T < 0.85 are flagged for exclusion review.
3. **Precision.** Observed values are kept in °F to the precision of the source (tenths of °C) and not rounded. Rounding is unbiased on average and would only add quantization noise.

## Matching rules (fixed before seeing data)

- The group for the period ending at synoptic hour H is taken from a report valid in `[H − 60 min, H)` that carries the group. If more than one report qualifies, the latest wins. US synoptic reports are issued at about H−9 min.
- Groups are parsed only from the remarks section (after `RMK`), as whole 5-character tokens `1snTTT` / `2snTTT`, where s ∈ {0, 1} is the sign and TTT is tenths of °C.
