# Brainstorm: ML portfolio project — NWS forecast calibration audit (2026-09-26)

**Problem.** Build a public, always-running portfolio project that proves predictive/ML depth end to end — historical backfill plus a live feed, a model whose forecasts are graded against reality, and visuals of that track record — filling the gap left by WAGA (engineering on mock data) and housing-commute-analysis (descriptive econometrics).

**Appetite.** Ongoing living project: built once, runs indefinitely on rammingspeed (home server, Dagster OSS in Docker), refreshing on a schedule; value compounds as the scored record accumulates. Stack: Dagster + dbt, publishing to charleslikesdata.com.

**Direction.** A public, continuously updated **calibration audit of National Weather Service forecasts**: for every ASOS (airport) station, measure where, when, and by how much NWS forecasts (high/low temperature first) are systematically wrong, and run a bias-correction model as an experiment *inside* the audit. "Bias" means signed error (forecast − observed) sliced by station × season × lead time. Both the raw NWS forecast and the corrected forecast are scored daily against observed ASOS actuals; the raw NWS forecast is the baseline the correction must beat (persistence/climatology are sanity floors, not the opponent). Bets that a meta-question — "how wrong is the expert, and where?" — is more novel and shareable than a raw forecast. The audit (bias map, per-city reliability) is the product; the correction model is an experiment within it, so "could not beat NWS" is a reportable finding, not a failure.

**Required changes (conditions of the commit).**
1. Product is the audit, not "beat NWS" — public framing, README, and site copy say so.
2. Archive-parity spike — **DONE 2026-09-26, PASS by grading NBM.** The graded product is **National Blend of Models station guidance (NBS)** via IEM `cgi-bin/request/mos.py?model=NBS`: one per-station endpoint serves both backfill (≥2020, exact start unpinned) and live (KPHX 18Z run present at 19:19Z same day, ~80 min latency). Backfill and live therefore score one product by construction. NBM is the machine guidance NWS forecasters start from; the forecaster-edited NDFD grid that api.weather.gov serves is a *different* product (archive: AWS `noaa-ndfd-pds`, CONUS GRIB mosaics from ~Jun 2020, no per-station extraction) and is NOT the graded product. Public framing must say "auditing the NWS's core forecast guidance," not "the forecast on weather.gov." Scoring obs: IEM CLI climate reports (`json/cli.py`, ~2002→present) or IEM ASOS daily summaries (`request/daily.py`). Spike notes and sample payloads: `docs/spikes/2026-09-26-archive-parity/`. **Cycle-regime finding:** IEM's archived NBS cycles were 01/07/13/19Z through at least 2026-04-15 and 00/06/12/18Z from at least 2026-05-10 (2026-05-01 holds all 24 hourly cycles), so the canonical daily run must be "the archived cycle nearest 12Z," with the cycle hour recorded per row and the changeover treated as an annotated regime boundary.
3. Visual contract: an animated seasonal bias map of the US, plus a per-city reliability card (e.g. "Phoenix highs run X°F cold in monsoon season").

**Criteria.** (frozen) Novelty; visual payoff; predictions graded against external reality; backfill available on day one plus a live feed; free public data.

**Killed options.**
- *Deepen WAGA's mock-fleet realism* — bet that realism inside a simulator shows modeling skill; lost because a model scored against self-generated data proves nothing. Requirement: the target must be a truth you don't control. (Remains a WAGA backlog item.)
- *EIA-930 grid demand vs. the grid operator's day-ahead forecast* — bet that a real benchmark beats novelty; lost on novelty (canonical energy-forecasting portfolio project; reads as "WAGA again"). Requirement: must look like a different project.
- *Wikipedia pageview/attention forecasting* — bet that big clean behavioral data beats geography; lost on visuals (no spatial dimension) and novelty (Kaggle 2017 Web Traffic competition). Requirement: needs a spatial dimension.
- *GTFS-realtime transit delay prediction* — bet that a local animated map is the showcase; lost because most agencies have no backfill and reliability trackers exist. Requirement: history must exist before code does.
- *eBird migration-arrival forecasting* (runner-up) — bet that an unusual domain and the best visuals win; lost because arrival targets resolve once a year (scorecard stays one row long) and observer-effort bias demands occupancy modeling with a high chance of an embarrassing miss; BirdCast/eBird Status & Trends already occupy the space.

**Premortem risks (winner).**
1. Null result — NWS output is already MOS/NBM-corrected; mitigated by change 1.
2. Archive/live product mismatch invalidates the backfill; mitigated by change 2.
3. Audience doesn't get "a model of another model's error"; the visual contract and plain-language city cards must carry it.
4. ASOS siting — airports aren't where people live; observation gaps.
5. api.weather.gov flakiness (errors, rate limits) leaves gaps in the live record; the pipeline must tolerate and log missing days rather than silently drop them.

**Open questions (deferred to PRD/implementation).** Local-day convention for max/min: NBM/NDFD max windows are ~12Z–00Z-ish (not calendar days) and CLI uses a fixed local-standard-time climate day, while ASOS daily summaries use local midnight — pick one and document it. "As-of" issuance convention for lead-time buckets (IEM archives four NBS cycles/day: 00/06/12/18Z). Phase-2 option: add NDFD (live-accumulated, optionally GRIB-backfilled from AWS) to ask "do human forecasters add value over the blend?" — a strong novelty hook, but a second product with high extraction cost. Variables beyond max/min temperature (precip probability, wind); lead times scored (day 1 vs. day 1–7); station universe (all ASOS vs. a curated set); correction model family; publishing surface (static site vs. app) and refresh cadence; public repo vs. private repo with a public site.

**Routing.** `write-a-prd` — this is a new product with a committed direction; the PRD interview should start from the archive-parity spike as its first gated milestone.
