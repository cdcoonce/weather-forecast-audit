# Pre-registration: when did each NBM version actually take over?

**Provenance:** the rule below was written into the probe's instructions before any bulletin was fetched. This file commits it afterwards, together with the results. The first probe covered v5.0 only; after it contradicted the Service Change Notice, the same rule was applied unchanged to every other boundary.

## Question

NWS Service Change Notices (SCNs) announce a date on which each NBM version will begin ("on or about"). Did the archived NBS guidance actually switch version at those runs?

## Instrument

Every NBS station block's header line names the version, for example `KPHX    NBM V4.3 NBS GUIDANCE    4/30/2026  1300 UTC`. The AWS Open Data archive holds one text file per hourly run at `noaa-nbm-grib2-pds/blend.YYYYMMDD/HH/text/blend_nbstx.tHHz`, and a file carries a single version throughout. `scripts/probe_nbm_version_headers.py` reads the first 4 KB of each hourly run with an HTTP Range request and records the first header's version.

## Decision rule

1. **Window:** for each SCN boundary, read every hourly run from 00Z on the SCN date − 2 days through 23Z on the SCN date + 7 days.
2. **First stable run of the new version:** the earliest run whose header shows the new version such that every later run in the window also shows it. The boundary is set at this run.
3. **Reported alongside:** the last old-version run, and every isolated flip (a new-version run later followed by an old-version run).
4. **Missing product:** if the text product is absent for a window, report "not obtainable" and do not infer the version from anything else.
5. **No stabilization:** if the new version has not stabilized by the end of the window, extend the window once by 7 days.

## Consequence for the project

`dbt/seeds/nbm_versions.csv` sets `valid_from_utc` to the first stable run and sets `verified = true` only for boundaries this rule confirms. The SCN stays as the `source_url`, citing what was announced.
