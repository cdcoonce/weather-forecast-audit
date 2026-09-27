# NBM NBS header-version probe — results

Method: `probe_nbm_version_headers.py`, ranged GET (`bytes=0-4095`) against
`blend.YYYYMMDD/HH/text/blend_nbstx.tHHz`, one request per hourly run,
0.2 s delay, 15 s timeout, 3 retries. All 1,056 requests succeeded on the
first try (no retries, no missing runs) — see `err_*.log` (all empty).
Every window used the pre-registered rule: retrieve every hourly run from
00Z (SCN date − 2 days) through 23Z (SCN date + 7 days); "first stable"
= earliest new-version run after which every later run in the window is
also the new version. The v4.3→v5.0 window used the separately-specified
2026-05-04T00Z–2026-05-07T23Z reconfirm range instead of the ±2/+7 rule.

Canonical-run convention (given, not derived here): the archived cycle
nearest 12Z is **13Z** for 2020-09-29 through 2026-04-29, and **12Z** from
2026-04-30 onward. "Canonical first-tagged run" below is the first
canonical-hour run (13Z or 12Z, per that convention) whose header shows
the new version.

## v3.2 → v4.0

- SCN claim: 2020-09-29, 12Z
- Window: 2020-09-27T00Z – 2020-10-06T23Z (240/240 runs retrieved)
- Last v3.2 run: **2020-09-29T11:00Z**
- First stable v4.0 run: **2020-09-29T12:00Z**
- Isolated flips: **none**
- Matches SCN: **yes**, exactly
- Canonical run (13Z era) first tagged v4.0: **2020-09-29** (13Z that day is v4.0, part of the stable run)

## v4.0 → v4.1

- SCN claim: 2023-01-17, 12Z
- Window: 2023-01-15T00Z – 2023-01-24T23Z (240/240 runs retrieved)
- Last v4.0 run: **2023-01-17T11:00Z**
- First stable v4.1 run: **2023-01-17T12:00Z**
- Isolated flips: **none**
- Matches SCN: **yes**, exactly
- Canonical run (13Z era) first tagged v4.1: **2023-01-17** (13Z that day is v4.1)

## v4.1 → v4.2

- SCN claim: 2024-05-15, 12Z
- Window: 2024-05-13T00Z – 2024-05-22T23Z (240/240 runs retrieved)
- Last v4.1 run: **2024-05-15T10:00Z**
- First stable v4.2 run: **2024-05-15T11:00Z** (1 hour before the SCN-claimed 12Z)
- Isolated flips (v4.2 runs that later reverted to v4.1), 9 total:
  2024-05-14T22Z, 2024-05-15T00Z, 01Z, 02Z, 04Z, 05Z, 06Z, 07Z, 08Z
- Matches SCN: **close but not exact** — stabilization landed at 11Z, one hour ahead of the claimed 12Z, after a day and a half of intermittent flips starting 2024-05-14T22Z
- Canonical run (13Z era) first tagged v4.2: **2024-05-15** (13Z that day; the 05-14T13Z and earlier canonical runs were still v4.1 — none of the pre-stabilization flips landed on the 13Z canonical hour)

## v4.2 → v4.3

- SCN claim: 2025-05-27, 12Z
- Window: 2025-05-25T00Z – 2025-06-03T23Z (240/240 runs retrieved)
- Last v4.2 run: **2025-05-27T11:00Z**
- First stable v4.3 run: **2025-05-27T12:00Z**
- Isolated flips (v4.3 runs that later reverted to v4.2), 5 total:
  2025-05-26T16Z, 19Z, 20Z, 21Z, 2025-05-27T07Z
- Matches SCN: **yes**, exactly, despite a day of intermittent flips beforehand
- Canonical run (13Z era) first tagged v4.3: **2025-05-27** (13Z that day; 2025-05-26T13Z was not among the flip hours, so it was still v4.2)

## v4.3 → v5.0 (reconfirm of prior full-file probe)

- SCN claim: 2026-04-30, 13Z (already known from the prior probe to be wrong)
- Window (as specified): 2026-05-04T00Z – 2026-05-07T23Z (96/96 runs retrieved)
- Last v4.3 run: **2026-05-05T11:00Z**
- First stable v5.0 run: **2026-05-05T12:00Z**
- Isolated flips (v5.0 runs that later reverted to v4.3), 8 total:
  2026-05-04T13Z, 15Z, 17Z, 19Z, 21Z, 2026-05-05T03Z, 05Z, 07Z
- Matches SCN: **no** — confirms the prior probe's finding of a ~5-day-late, non-monotonic cutover
- Canonical run (12Z era, from 2026-04-30 on) first tagged v5.0: **2026-05-05** (12Z that day; no earlier 12Z canonical run flipped — all the flips above land on off-canonical hours)
- **Cross-check against the prior full-file probe**: for every hour the earlier probe actually downloaded in full (04-30 13/14/19Z, 05-03 13Z, 05-04 12/18/20/22Z, 05-05 00/03/06/09/10/11/12/13/19Z, 05-06 00/06/12Z), this ranged probe's header read agrees exactly. The discrepancy is not a disagreement between methods — it's that the earlier probe's sparse hour selection happened to skip every one of the odd-hour flip runs (05-04T13, 15, 17, 19 and 21Z, 05-05T05 and 07Z; only 05-05T03Z was in its sample) and so under-reported "one isolated flip" when the true count in this window is 8. The ranged method, by covering every hour, is strictly more complete at no meaningful extra cost.

## Summary table

| Boundary | SCN claim | First stable run (UTC) | Match | Flips | Canonical first-tagged date |
|---|---|---|---|---|---|
| v3.2→v4.0 | 2020-09-29 12Z | 2020-09-29 12Z | yes | 0 | 2020-09-29 |
| v4.0→v4.1 | 2023-01-17 12Z | 2023-01-17 12Z | yes | 0 | 2023-01-17 |
| v4.1→v4.2 | 2024-05-15 12Z | 2024-05-15 11Z | close (−1h) | 9 | 2024-05-15 |
| v4.2→v4.3 | 2025-05-27 12Z | 2025-05-27 12Z | yes | 5 | 2025-05-27 |
| v4.3→v5.0 | 2026-04-30 13Z | 2026-05-05 12Z | no | 8 | 2026-05-05 |

No boundary was "not obtainable" — the `text/` listing at `blend.YYYYMMDD/13/text/` (or `/12/text/` for the reconfirm window) returned `blend_nbstx.tHHz` with a normal size (~29–30 MB) for every SCN date checked, and every hourly run in every window existed and returned a valid header on the first ranged request.

## Data/bytes

- 1,056 ranged HTTP requests total (240 × 4 boundaries + 96 reconfirm), each capped at 4,096 bytes → **4,325,376 bytes (≈4.13 MB) downloaded**, vs. ~590 MB for the ~20 full files fetched in the prior probe.
