"""IEM `mos.py` client for archived NBS (NBM station guidance) rows.

The column set drifts by date (IEM inserted an `slv` column before `i06`
sometime between the 2023 and 2026 fixtures), so parsing is strictly by
header name, never position. Rows are kept only when their runtime falls on
the canonical archived cycle for that runtime date (build spec fact 2); the
regime lookup is the only place a changeover date is read from.
"""

import csv
import io
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from weather_forecast_audit.gaps import FetchResult, GapRecord
from weather_forecast_audit.iem._chunking import date_range, month_chunks
from weather_forecast_audit.iem.http import Fetcher, FetchError
from weather_forecast_audit.regimes import CycleRegime, canonical_cycle_hour

REQUIRED_COLUMNS = ("runtime", "ftime", "station", "txn", "xnd", "tmp")
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"


@dataclass(frozen=True)
class GuidanceRow:
    station: str
    runtime: datetime
    ftime: datetime
    cycle_hour: int
    txn: float | None
    xnd: float | None
    tmp: float | None


def _parse_naive_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def _parse_optional_float(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def parse_nbs_csv(body: bytes) -> list[dict[str, str]]:
    """Parse an IEM mos.py CSV body into header-keyed rows.

    Raises ValueError naming the missing column(s) if any required column
    (runtime, ftime, station, txn, xnd, tmp) is absent from the header.
    """
    reader = csv.DictReader(io.StringIO(body.decode("utf-8")))
    fieldnames = reader.fieldnames or []
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        msg = f"NBS CSV missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    return list(reader)


def _guidance_url(station: str, chunk_start: date, chunk_end: date) -> str:
    sts = f"{chunk_start.isoformat()}T00:00Z"
    ets = f"{(chunk_end + timedelta(days=1)).isoformat()}T00:00Z"
    return f"{BASE_URL}?station={station}&model=NBS&sts={sts}&ets={ets}&format=csv"


def fetch_guidance(
    station: str,
    start: date,
    end: date,
    fetcher: Fetcher,
    regimes: list[CycleRegime],
) -> FetchResult[GuidanceRow]:
    """Fetch canonical-cycle NBS guidance for `station` over [start, end].

    One request per calendar-month chunk. A `missing_run` gap is emitted for
    every run date in range with zero canonical-cycle rows; a `no_txn` gap
    is emitted when the canonical run has rows but none carry a non-null
    `txn` (e.g. pre-NBM-v4.0 archives, which use `n_x` for the daily
    max/min instead -- issue #7); a fetch failure for a chunk emits
    `http_error:*` gaps for every run date in that chunk only, so other
    chunks still complete.
    """
    rows: list[GuidanceRow] = []
    gaps: list[GapRecord] = []

    for chunk_start, chunk_end in month_chunks(start, end):
        chunk_dates = date_range(chunk_start, chunk_end)
        url = _guidance_url(station, chunk_start, chunk_end)
        try:
            response = fetcher.get(url)
        except FetchError as exc:
            status = exc.status if exc.status is not None else exc.reason
            reason = f"http_error:{status}"
            for run_date in chunk_dates:
                hour = canonical_cycle_hour(run_date, regimes)
                expected = f"{run_date.isoformat()}T{hour:02d}:00Z"
                gaps.append(
                    GapRecord(
                        station=station, source="nbs", expected=expected, reason=reason
                    )
                )
            continue

        canonical_dates_seen: set[date] = set()
        canonical_dates_with_txn: set[date] = set()
        for raw in parse_nbs_csv(response.body):
            runtime = _parse_naive_utc(raw["runtime"])
            run_date = runtime.date()
            hour = canonical_cycle_hour(run_date, regimes)
            if runtime.hour != hour:
                continue
            txn = _parse_optional_float(raw["txn"])
            rows.append(
                GuidanceRow(
                    station=raw["station"],
                    runtime=runtime,
                    ftime=_parse_naive_utc(raw["ftime"]),
                    cycle_hour=hour,
                    txn=txn,
                    xnd=_parse_optional_float(raw["xnd"]),
                    tmp=_parse_optional_float(raw["tmp"]),
                )
            )
            canonical_dates_seen.add(run_date)
            if txn is not None:
                canonical_dates_with_txn.add(run_date)

        for run_date in chunk_dates:
            hour = canonical_cycle_hour(run_date, regimes)
            expected = f"{run_date.isoformat()}T{hour:02d}:00Z"
            if run_date not in canonical_dates_seen:
                gaps.append(
                    GapRecord(
                        station=station,
                        source="nbs",
                        expected=expected,
                        reason="missing_run",
                    )
                )
            elif run_date not in canonical_dates_with_txn:
                # The canonical run exists (rows were returned) but none of
                # them carry a non-null txn -- e.g. pre-NBM-v4.0 archives,
                # which named the daily max/min column `n_x` instead.
                # Treating this as "no gap" would silently hide guidance
                # that is present but unusable.
                gaps.append(
                    GapRecord(
                        station=station,
                        source="nbs",
                        expected=expected,
                        reason="no_txn",
                    )
                )

    return FetchResult(rows=rows, gaps=gaps)
