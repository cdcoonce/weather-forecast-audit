"""IEM `asos.py` and `cli.py` clients: hourly obs and CLI daily reports.

ASOS station params use the 3-letter code, which drops a CONUS ICAO's
leading `K` (KPHX -> PHX). A live check confirmed `station=PHX` with no
`network` param resolves without ambiguity (build spec fact 5), so no
`iem_asos_network` registry column is added.
"""

import csv
import io
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from weather_forecast_audit.gaps import FetchResult, GapRecord
from weather_forecast_audit.iem._chunking import date_range, month_chunks
from weather_forecast_audit.iem.http import Fetcher, FetchError
from weather_forecast_audit.iem.metar import parse_six_hour_groups
from weather_forecast_audit.registry import Station

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py"

# The raw metar column must be present; other columns are still read
# positionally by name (raw["valid"], raw["tmpf"]) rather than validated here,
# matching the existing behavior this module had before metar parsing.
REQUIRED_ASOS_COLUMNS = ("valid", "tmpf", "metar")


@dataclass(frozen=True)
class HourlyObservation:
    station: str  # ICAO, e.g. KPHX (not the 3-letter ASOS request code)
    valid_utc: datetime
    tmpf: float | None
    max_6h_f: float | None
    min_6h_f: float | None


@dataclass(frozen=True)
class CliDaily:
    station: str
    local_date: date
    high_f: int | None
    low_f: int | None


def _asos_station_code(icao: str) -> str:
    return icao[1:] if len(icao) == 4 and icao.startswith("K") else icao


def _parse_optional_float(value: str) -> float | None:
    return None if value in ("", "M", None) else float(value)


def _parse_optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().upper() == "M":
        return None
    return int(value)


def _parse_asos_valid(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)


def _c_to_f(value_c: float) -> float:
    return value_c * 9 / 5 + 32


def _asos_url(icao: str, chunk_start: date, chunk_end: date) -> str:
    code = _asos_station_code(icao)
    sts = f"{chunk_start.isoformat()}T00:00Z"
    ets = f"{(chunk_end + timedelta(days=1)).isoformat()}T00:00Z"
    return (
        f"{ASOS_URL}?station={code}&data=tmpf&data=metar&sts={sts}&ets={ets}"
        "&tz=Etc/UTC&format=onlycomma&missing=M"
        "&report_type=3&report_type=4&latlon=no"
    )


def parse_asos_csv(body: bytes) -> list[dict[str, str]]:
    """Parse an IEM asos.py onlycomma body into header-keyed rows.

    Raises ValueError naming the missing column(s) if any required column
    (valid, tmpf, metar) is absent from the header, the same column-by-name
    check `iem.guidance.parse_nbs_csv` does for the NBS client.
    """
    reader = csv.DictReader(io.StringIO(body.decode("utf-8")))
    fieldnames = reader.fieldnames or []
    missing = [column for column in REQUIRED_ASOS_COLUMNS if column not in fieldnames]
    if missing:
        msg = f"ASOS CSV missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    return list(reader)


def fetch_hourly(
    station: Station, start: date, end: date, fetcher: Fetcher
) -> FetchResult[HourlyObservation]:
    """Fetch hourly tmpf observations for `station` over [start, end].

    One request per calendar-month chunk. A `missing_observations` gap is
    emitted for every UTC date in range with zero non-missing tmpf readings.
    """
    icao = station.icao
    rows: list[HourlyObservation] = []
    gaps: list[GapRecord] = []

    for chunk_start, chunk_end in month_chunks(start, end):
        url = _asos_url(icao, chunk_start, chunk_end)
        try:
            response = fetcher.get(url)
        except FetchError as exc:
            status = exc.status if exc.status is not None else exc.reason
            reason = f"http_error:{status}"
            for day in date_range(chunk_start, chunk_end):
                gaps.append(
                    GapRecord(
                        station=icao,
                        source="asos",
                        expected=day.isoformat(),
                        reason=reason,
                    )
                )
            continue

        dates_with_data: set[date] = set()
        for raw in parse_asos_csv(response.body):
            valid = _parse_asos_valid(raw["valid"])
            tmpf = _parse_optional_float(raw["tmpf"])
            groups = parse_six_hour_groups(raw["metar"])
            max_6h_f = _c_to_f(groups.max_c) if groups.max_c is not None else None
            min_6h_f = _c_to_f(groups.min_c) if groups.min_c is not None else None
            rows.append(
                HourlyObservation(
                    station=icao,
                    valid_utc=valid,
                    tmpf=tmpf,
                    max_6h_f=max_6h_f,
                    min_6h_f=min_6h_f,
                )
            )
            if tmpf is not None:
                dates_with_data.add(valid.date())

        for day in date_range(chunk_start, chunk_end):
            if day not in dates_with_data:
                gaps.append(
                    GapRecord(
                        station=icao,
                        source="asos",
                        expected=day.isoformat(),
                        reason="missing_observations",
                    )
                )

    return FetchResult(rows=rows, gaps=gaps)


def fetch_cli(
    station: Station, start: date, end: date, fetcher: Fetcher
) -> FetchResult[CliDaily]:
    """Fetch CLI daily high/low reports for `station` over [start, end].

    One request per year. A `missing_report` gap is emitted for every local
    date in range with no report, or with both high and low missing ("M").

    Rows carry the station's ICAO (not the CLI product id, used only to
    build the request) so every raw table's `station` column is the same
    join key back to the station registry.
    """
    icao = station.icao
    by_date: dict[date, CliDaily] = {}
    gaps: list[GapRecord] = []

    for year in range(start.year, end.year + 1):
        url = f"{CLI_URL}?station={station.cli_station}&year={year}"
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year, 12, 31))
        try:
            response = fetcher.get(url)
        except FetchError as exc:
            status = exc.status if exc.status is not None else exc.reason
            reason = f"http_error:{status}"
            for day in date_range(year_start, year_end):
                gaps.append(
                    GapRecord(
                        station=icao, source="cli", expected=day.isoformat(),
                        reason=reason,
                    )
                )
            continue

        payload = json.loads(response.body)
        for record in payload.get("results", []):
            local_date = date.fromisoformat(record["valid"])
            if not (start <= local_date <= end):
                continue
            by_date[local_date] = CliDaily(
                station=icao,
                local_date=local_date,
                high_f=_parse_optional_int(record.get("high")),
                low_f=_parse_optional_int(record.get("low")),
            )

    rows: list[CliDaily] = []
    for day in date_range(start, end):
        daily = by_date.get(day)
        if daily is None or (daily.high_f is None and daily.low_f is None):
            gaps.append(
                GapRecord(
                    station=icao,
                    source="cli",
                    expected=day.isoformat(),
                    reason="missing_report",
                )
            )
        else:
            rows.append(daily)

    return FetchResult(rows=rows, gaps=gaps)
