"""ASOS hourly and CLI daily clients: parsing, M-handling, gaps."""

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from weather_forecast_audit.iem.http import FetchError, HttpResponse
from weather_forecast_audit.iem.observations import fetch_cli, fetch_hourly
from weather_forecast_audit.registry import load_registry

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"
STATION = load_registry()["KPHX"]


class FakeFetcher:
    def __init__(self, routes: list[tuple[Callable[[str], bool], object]]) -> None:
        self._routes = routes
        self.requested_urls: list[str] = []

    def get(self, url: str) -> HttpResponse:
        self.requested_urls.append(url)
        for predicate, outcome in self._routes:
            if predicate(url):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        msg = f"FakeFetcher: no route matched {url}"
        raise AssertionError(msg)


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


# -- ASOS hourly ---------------------------------------------------------


def test_fetch_hourly_parses_fixture_uses_asos_3letter_code() -> None:
    fetcher = FakeFetcher(
        [
            (
                lambda u: "station=PHX" in u,
                HttpResponse(200, _read("asos_kphx_2023-07-13_2023-07-17.csv")),
            ),
        ]
    )
    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 17), fetcher)

    assert len(result.rows) == 108
    assert all(row.station == "KPHX" for row in result.rows)
    assert not result.gaps


def test_fetch_hourly_missing_tmpf_parses_as_none() -> None:
    body = b"station,valid,tmpf\nPHX,2023-07-13 00:51,M\nPHX,2023-07-13 01:51,80.00\n"
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)

    tmpfs = {row.valid_utc.hour: row.tmpf for row in result.rows}
    assert tmpfs[0] is None
    assert tmpfs[1] == 80.0
    assert not result.gaps  # one non-missing hour is enough for the date


def test_fetch_hourly_all_missing_emits_missing_observations_gap() -> None:
    body = b"station,valid,tmpf\nPHX,2023-07-13 00:51,M\n"
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)

    assert len(result.gaps) == 1
    assert result.gaps[0].reason == "missing_observations"
    assert result.gaps[0].expected == "2023-07-13"


def test_fetch_hourly_http_error_emits_gap_per_day() -> None:
    fetcher = FakeFetcher([(lambda _u: True, FetchError(status=503, reason="x"))])

    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 15), fetcher)

    assert result.rows == []
    assert len(result.gaps) == 3
    assert all(g.reason == "http_error:503" for g in result.gaps)


# -- CLI daily -------------------------------------------------------------


def test_fetch_cli_parses_known_answer_days() -> None:
    fetcher = FakeFetcher(
        [(lambda u: "year=2023" in u, HttpResponse(200, _read("cli_kphx_2023.json")))]
    )
    result = fetch_cli(STATION, date(2023, 7, 14), date(2023, 7, 15), fetcher)

    by_date = {row.local_date: row for row in result.rows}
    assert by_date[date(2023, 7, 14)].high_f == 116
    assert by_date[date(2023, 7, 14)].low_f == 93
    assert by_date[date(2023, 7, 15)].high_f == 118
    assert by_date[date(2023, 7, 15)].low_f == 92
    assert not result.gaps


def test_fetch_cli_m_values_treated_as_missing_report() -> None:
    payload = json.dumps(
        {"results": [{"valid": "2023-07-14", "high": "M", "low": "M"}]}
    ).encode("utf-8")
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, payload))])

    result = fetch_cli(STATION, date(2023, 7, 14), date(2023, 7, 14), fetcher)

    assert result.rows == []
    assert len(result.gaps) == 1
    assert result.gaps[0].reason == "missing_report"


def test_fetch_cli_absent_date_emits_missing_report_gap() -> None:
    payload = json.dumps({"results": []}).encode("utf-8")
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, payload))])

    result = fetch_cli(STATION, date(2023, 7, 14), date(2023, 7, 14), fetcher)

    assert result.rows == []
    assert result.gaps[0].reason == "missing_report"
    assert result.gaps[0].expected == "2023-07-14"


def test_fetch_cli_one_request_per_year() -> None:
    fetcher = FakeFetcher(
        [
            (lambda u: True, HttpResponse(200, json.dumps({"results": []}).encode())),
        ]
    )
    fetch_cli(STATION, date(2022, 12, 30), date(2023, 1, 2), fetcher)
    assert len(fetcher.requested_urls) == 2
    assert "year=2022" in fetcher.requested_urls[0]
    assert "year=2023" in fetcher.requested_urls[1]
