"""ASOS hourly and CLI daily clients: parsing, M-handling, gaps."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from weather_forecast_audit import warehouse
from weather_forecast_audit.gaps import FetchResult, GapRecord
from weather_forecast_audit.iem.http import FetchError, HttpResponse
from weather_forecast_audit.iem.observations import (
    HourlyObservation,
    _c_to_f,
    fetch_cli,
    fetch_hourly,
)
from weather_forecast_audit.registry import load_registry

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"
STATION = load_registry()["KPHX"]
SECOND_STATION = load_registry()["KSEA"]


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


def test_fetch_hourly_requests_metar_alongside_tmpf() -> None:
    fetcher = FakeFetcher(
        [
            (
                lambda u: "station=PHX" in u,
                HttpResponse(200, _read("asos_kphx_2023-07-13_2023-07-17.csv")),
            ),
        ]
    )
    fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 17), fetcher)

    assert "data=tmpf" in fetcher.requested_urls[0]
    assert "data=metar" in fetcher.requested_urls[0]


def test_fetch_hourly_populates_six_hour_groups_on_synoptic_rows_only() -> None:
    fetcher = FakeFetcher(
        [(lambda _u: True, HttpResponse(200, _read(
            "asos_kphx_2023-07-13_2023-07-17.csv"
        )))]
    )
    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 17), fetcher)

    by_valid = {row.valid_utc: row for row in result.rows}
    synoptic = by_valid[datetime(2023, 7, 15, 23, 51, tzinfo=UTC)]
    assert synoptic.max_6h_f == pytest.approx(47.8 * 9 / 5 + 32)
    assert synoptic.min_6h_f == pytest.approx(41.7 * 9 / 5 + 32)

    non_synoptic = by_valid[datetime(2023, 7, 15, 0, 51, tzinfo=UTC)]
    assert non_synoptic.max_6h_f is None
    assert non_synoptic.min_6h_f is None


def test_fetch_hourly_missing_metar_column_raises() -> None:
    body = b"station,valid,tmpf\nPHX,2023-07-13 00:51,80.00\n"
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    with pytest.raises(ValueError, match="metar"):
        fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)


KCAK = load_registry()["KCAK"]

# The real production report that halted the national backfill (KCAK,
# 2020-11-23 17:51 UTC): two different 6-hour max groups in one remark.
_KCAK_AMBIGUOUS_MAX = (
    b"station,valid,tmpf,metar\n"
    b"CAK,2020-11-23 17:51,41.00,"
    b"KCAK 231751Z RMK AO2 T00410006 10056 10083 20034\n"
)


def _kcak_fetch(
    body: bytes, end: date = date(2020, 11, 23), start: date = date(2020, 11, 23)
) -> FetchResult[HourlyObservation]:
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])
    return fetch_hourly(KCAK, start, end, fetcher)


def test_fetch_hourly_ambiguous_max_drops_only_the_max_and_records_a_gap() -> None:
    result = _kcak_fetch(_KCAK_AMBIGUOUS_MAX)

    (row,) = result.rows
    assert row.station == "KCAK"
    assert row.tmpf == 41.0
    assert row.max_6h_f is None
    assert row.min_6h_f == _c_to_f(3.4)
    assert result.gaps == [
        GapRecord(
            station="KCAK",
            source="asos",
            expected="2020-11-23",
            reason="ambiguous_six_hour_max@17:51",
        )
    ]


def test_fetch_hourly_ambiguous_min_drops_only_the_min_and_records_a_gap() -> None:
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 17:51,41.00,"
        b"KCAK 231751Z RMK AO2 T00410006 10056 20034 20012\n"
    )
    result = _kcak_fetch(body)

    (row,) = result.rows
    assert row.max_6h_f == _c_to_f(5.6)
    assert row.min_6h_f is None
    assert row.tmpf == 41.0
    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_min@17:51")
    ]


def test_fetch_hourly_ambiguous_max_and_min_record_one_gap_each() -> None:
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 17:51,41.00,"
        b"KCAK 231751Z RMK AO2 10056 10083 20034 20012\n"
    )
    result = _kcak_fetch(body)

    (row,) = result.rows
    assert (row.max_6h_f, row.min_6h_f) == (None, None)
    assert [gap.reason for gap in result.gaps] == [
        "ambiguous_six_hour_max@17:51",
        "ambiguous_six_hour_min@17:51",
    ]


def test_fetch_hourly_ambiguous_gap_names_the_station_being_fetched() -> None:
    # A second station, so the gap cannot be satisfied by a hardcoded KCAK.
    assert SECOND_STATION.icao != KCAK.icao
    body = (
        b"station,valid,tmpf,metar\n"
        b"SEA,2023-07-13 05:53,60.00,"
        b"KSEA 130553Z RMK AO2 T01560106 10461 10462 20144\n"
    )
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    result = fetch_hourly(SECOND_STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)

    assert result.gaps == [
        GapRecord("KSEA", "asos", "2023-07-13", "ambiguous_six_hour_max@05:53")
    ]


def test_fetch_hourly_ambiguous_gap_leaves_missing_day_gaps_untouched() -> None:
    # 11-23 has data (one ambiguous report), 11-24 has none, 11-25 has only a
    # missing tmpf: the ambiguity adds a gap without hiding or doubling the
    # two missing-day gaps.
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 17:51,41.00,"
        b"KCAK 231751Z RMK AO2 T00410006 10056 10083 20034\n"
        b"CAK,2020-11-25 00:51,M,M\n"
    )
    result = _kcak_fetch(body, end=date(2020, 11, 25))

    assert len(result.rows) == 2
    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@17:51"),
        GapRecord("KCAK", "asos", "2020-11-24", "missing_observations"),
        GapRecord("KCAK", "asos", "2020-11-25", "missing_observations"),
    ]


def test_fetch_hourly_ambiguous_only_report_without_tmpf_keeps_both_gaps() -> None:
    # The day's only report is ambiguous and has no tmpf: the day still has
    # zero non-missing readings, so it keeps its missing_observations gap next
    # to the ambiguous one.
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 17:51,M,KCAK 231751Z RMK AO2 10056 10083 20034\n"
    )
    result = _kcak_fetch(body)

    (row,) = result.rows
    assert row.tmpf is None
    assert row.max_6h_f is None
    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@17:51"),
        GapRecord("KCAK", "asos", "2020-11-23", "missing_observations"),
    ]


def test_fetch_hourly_ambiguous_gap_is_emitted_when_tmpf_is_missing() -> None:
    # Another day has data, so the only gap is the ambiguous one, and it must
    # not depend on the ambiguous report itself carrying a tmpf.
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 05:51,40.00,M\n"
        b"CAK,2020-11-23 17:51,M,KCAK 231751Z RMK AO2 10056 10083 20034\n"
    )
    result = _kcak_fetch(body)

    assert [row.tmpf for row in result.rows] == [40.0, None]
    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@17:51")
    ]


def test_fetch_hourly_ambiguous_gap_expected_is_the_reports_own_date() -> None:
    # The report is on the second day of the range, not its first.
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-22 12:51,40.00,M\n"
        b"CAK,2020-11-23 17:51,41.00,KCAK 231751Z RMK AO2 10056 10083 20034\n"
    )
    result = _kcak_fetch(body, start=date(2020, 11, 22))

    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-23", "ambiguous_six_hour_max@17:51")
    ]


def test_fetch_hourly_ambiguous_gap_expected_across_a_month_boundary() -> None:
    # Two chunks (Oct 30-31, Nov 1-2); the report is in the second chunk and
    # not on its first day, so neither chunk start nor range start is right.
    october = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-10-30 12:51,50.00,M\n"
        b"CAK,2020-10-31 12:51,51.00,M\n"
    )
    november = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-01 12:51,42.00,M\n"
        b"CAK,2020-11-02 05:51,43.00,KCAK 020551Z RMK AO2 10056 10083 20034\n"
    )
    fetcher = FakeFetcher(
        [
            (lambda u: "sts=2020-10-30T" in u, HttpResponse(200, october)),
            (lambda u: "sts=2020-11-01T" in u, HttpResponse(200, november)),
        ]
    )

    result = fetch_hourly(KCAK, date(2020, 10, 30), date(2020, 11, 2), fetcher)

    assert len(fetcher.requested_urls) == 2
    assert result.gaps == [
        GapRecord("KCAK", "asos", "2020-11-02", "ambiguous_six_hour_max@05:51")
    ]


def test_fetch_hourly_repeated_ambiguous_report_records_one_gap() -> None:
    # The same valid minute twice (e.g. a METAR and a SPECI): one gap, so the
    # loaded ledger keeps one row per (station, date, reason).
    row = (
        b"CAK,2020-11-23 17:51,41.00,"
        b"KCAK 231751Z RMK AO2 T00410006 10056 10083 20034\n"
    )
    result = _kcak_fetch(b"station,valid,tmpf,metar\n" + row + row)

    assert len(result.rows) == 2
    assert len(result.gaps) == 1


def test_fetch_hourly_ambiguous_gaps_load_idempotently_and_stay_unique_per_day(
    tmp_path: Path,
) -> None:
    # Two ambiguous max reports on the same UTC day must not collapse to one
    # (station, kind, gap_date, reason) key: gap_ledger tests that key unique.
    body = (
        b"station,valid,tmpf,metar\n"
        b"CAK,2020-11-23 05:51,41.00,KCAK 230551Z RMK AO2 10056 10083 20034\n"
        b"CAK,2020-11-23 17:51,41.00,KCAK 231751Z RMK AO2 10056 10083 20034\n"
    )
    result = _kcak_fetch(body)
    conn = duckdb.connect(str(tmp_path / "warehouse.duckdb"))
    warehouse.init_db(conn)
    now = datetime(2024, 1, 1)  # noqa: DTZ001 - load_gaps takes naive UTC
    start = end = date(2020, 11, 23)

    warehouse.load_gaps(conn, "KCAK", "asos", start, end, result.gaps, now=now)
    warehouse.load_gaps(conn, "KCAK", "asos", start, end, result.gaps, now=now)

    assert conn.execute("select count(*) from raw.ingest_gaps").fetchone() == (2,)
    assert conn.execute(
        "select count(distinct (cast(substr(expected, 1, 10) as date), "
        "station, source, reason)) from raw.ingest_gaps"
    ).fetchone() == (2,)


def test_fetch_hourly_accepts_identically_duplicated_group() -> None:
    body = (
        b"station,valid,tmpf,metar\n"
        b"PHX,2023-07-13 05:51,90.00,"
        b"KPHX 130551Z RMK AO2 T04110106 10206 10206 20344\n"
    )
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)

    assert len(result.rows) == 1
    assert result.rows[0].max_6h_f == _c_to_f(20.6)
    assert result.rows[0].min_6h_f == _c_to_f(34.4)


def test_fetch_hourly_missing_tmpf_parses_as_none() -> None:
    body = (
        b"station,valid,tmpf,metar\n"
        b"PHX,2023-07-13 00:51,M,M\n"
        b"PHX,2023-07-13 01:51,80.00,M\n"
    )
    fetcher = FakeFetcher([(lambda _u: True, HttpResponse(200, body))])

    result = fetch_hourly(STATION, date(2023, 7, 13), date(2023, 7, 13), fetcher)

    tmpfs = {row.valid_utc.hour: row.tmpf for row in result.rows}
    assert tmpfs[0] is None
    assert tmpfs[1] == 80.0
    assert not result.gaps  # one non-missing hour is enough for the date


def test_fetch_hourly_all_missing_emits_missing_observations_gap() -> None:
    body = b"station,valid,tmpf,metar\nPHX,2023-07-13 00:51,M,M\n"
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
