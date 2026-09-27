#!/usr/bin/env python3
"""probe_nbm_version_headers.py

Ranged-read probe of NBM NBS text-bulletin header lines on AWS Open Data,
to determine when each NBM version actually appears in the archived
station-text guidance, independent of NWS Service Change Notice (SCN)
claims.

Each hourly run of the NBS bulletin
(https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.YYYYMMDD/HH/text/blend_nbstx.tHHz)
is a plain-text file (~30 MB) containing one block per station, all
stations in a single file sharing one NBM version. Because the version
is uniform within a file, only the first few KB (first station's header
line) need to be read, via an HTTP Range request -- the full file is
never downloaded.

Pre-registered decision rule (fixed before fetching; must not change
after seeing data):

  For each SCN version-boundary, retrieve the header version for every
  hourly run that exists in the bucket from 00Z on (SCN date - 2 days)
  through 23Z on (SCN date + 7 days).

    * "First stable run of the new version" = the earliest run whose
      header shows the new version such that EVERY later run retrieved
      in the window also shows the new version (i.e. no later reversion
      to the old version).
    * Report every isolated flip: a new-version run followed by any
      later old-version run (a reversion).
    * Report the last old-version run.
    * If the text/ product does not exist for the window at all, check
      the bucket listing for a differently-named product under
      blend.YYYYMMDD/13/text/ (?list-type=2&prefix=...) and report
      "not obtainable" plus what the listing contained -- never infer
      version from anything else (e.g. grib2 file presence).
    * If the new version has not stabilized by the end of the window,
      report that and extend the window by 7 more days, once.

This script only fetches data and writes the raw per-run CSV; the
window-by-window judgment above (first stable run, flips, "not
obtainable" calls) is applied afterward by the caller and recorded in
RESULTS.md -- it is not computed here.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta

BASE_URL = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
RANGE_BYTES = "bytes=0-4095"
HEADER_RE = re.compile(r"NBM\s+V(\d+\.\d+)\s+NBS\s+GUIDANCE")

REQUEST_TIMEOUT_S = 15
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5
POLITE_DELAY_S = 0.2


def parse_utc(s: str) -> datetime:
    """Parse a UTC datetime from common forms, e.g. '2020-09-27T00',
    '2020-09-27 00', '2020-09-27T00:00', '2020-09-27T00:00:00Z'."""
    s = s.strip().rstrip("Z")
    s = s.replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)  # noqa: DTZ007 (tz attached below)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Could not parse UTC datetime: {s!r}")


def run_url(run_dt: datetime) -> str:
    ymd = run_dt.strftime("%Y%m%d")
    hh = run_dt.strftime("%H")
    return f"{BASE_URL}/blend.{ymd}/{hh}/text/blend_nbstx.t{hh}z"


def fetch_header(run_dt: datetime) -> tuple[bool, str, str]:
    """Return (exists, header_version, header_line) for one hourly run.

    exists=False means the object was not found (HTTP 404) or could not
    be read after retries; header_version/header_line are '' in that case.
    """
    url = run_url(run_dt)
    req = urllib.request.Request(url, headers={"Range": RANGE_BYTES})

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                raw = resp.read()
            text = raw.decode("latin-1", errors="replace")
            for line in text.splitlines():
                m = HEADER_RE.search(line)
                if m:
                    return True, m.group(1), line.strip()
            # Object exists but no header match found in the fetched
            # bytes -- still "exists", version unknown.
            return True, "", ""
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, "", ""
            last_exc = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exc = e
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_S * attempt)

    print(
        f"warning: giving up on {url} after {MAX_RETRIES} attempts: {last_exc}",
        file=sys.stderr,
    )
    return False, "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--start", required=True, help="UTC start datetime, e.g. 2020-09-27T00"
    )
    ap.add_argument(
        "--end",
        required=True,
        help="UTC end datetime (inclusive hour), e.g. 2020-10-06T23",
    )
    args = ap.parse_args()

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    writer = csv.writer(sys.stdout)
    writer.writerow(["run_utc", "exists", "header_version", "header_line"])
    sys.stdout.flush()

    run = start
    first = True
    while run <= end:
        if not first:
            time.sleep(POLITE_DELAY_S)
        first = False
        exists, version, line = fetch_header(run)
        writer.writerow([run.strftime("%Y-%m-%dT%H:00Z"), exists, version, line])
        sys.stdout.flush()
        run += timedelta(hours=1)


if __name__ == "__main__":
    main()
