"""NBS guidance client: header-name parsing, canonical-cycle filtering, gaps."""

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from weather_forecast_audit.iem.guidance import fetch_guidance, parse_nbs_csv
from weather_forecast_audit.iem.http import FetchError, HttpResponse
from weather_forecast_audit.regimes import load_cycle_regimes

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"
REGIMES = load_cycle_regimes()


class FakeFetcher:
    """Maps URL predicates to a response (or an exception), in order."""

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


def test_parse_nbs_csv_reads_43_column_2023_fixture() -> None:
    rows = parse_nbs_csv(_read("nbs_kphx_2023-07-14.csv"))
    assert len(rows) == 92
    assert rows[0]["station"] == "KPHX"


def test_parse_nbs_csv_reads_44_column_2026_fixture() -> None:
    rows = parse_nbs_csv(_read("nbs_kphx_2026-04-29.csv"))
    assert len(rows) > 0
    assert all(row["station"] == "KPHX" for row in rows)


def test_parse_nbs_csv_empty_body_returns_no_rows() -> None:
    assert parse_nbs_csv(_read("nbs_kphx_empty.csv")) == []


def test_parse_nbs_csv_bogus_station_returns_no_rows() -> None:
    assert parse_nbs_csv(_read("nbs_bogus_station.csv")) == []


def test_parse_nbs_csv_missing_required_column_raises_clear_error() -> None:
    header = b"runtime,ftime,model,tmp,station,xnd\n"
    with pytest.raises(ValueError, match="txn"):
        parse_nbs_csv(header)


def test_fetch_guidance_2026_04_29_keeps_only_13z_canonical_rows() -> None:
    fetcher = FakeFetcher(
        [
            (
                lambda u: "2026-04" in u,
                HttpResponse(200, _read("nbs_kphx_2026-04-29.csv")),
            )
        ]
    )
    result = fetch_guidance(
        "KPHX", date(2026, 4, 29), date(2026, 4, 29), fetcher, REGIMES
    )
    assert result.rows
    assert {row.runtime.hour for row in result.rows} == {13}
    assert {row.cycle_hour for row in result.rows} == {13}
    assert not result.gaps


def test_fetch_guidance_2026_05_06_keeps_only_12z_canonical_rows() -> None:
    fetcher = FakeFetcher(
        [
            (
                lambda u: "2026-05" in u,
                HttpResponse(200, _read("nbs_kphx_2026-05-06.csv")),
            )
        ]
    )
    result = fetch_guidance(
        "KPHX", date(2026, 5, 6), date(2026, 5, 6), fetcher, REGIMES
    )
    assert result.rows
    assert {row.runtime.hour for row in result.rows} == {12}
    assert {row.cycle_hour for row in result.rows} == {12}
    assert not result.gaps


def test_fetch_guidance_missing_run_gap_when_canonical_rows_absent() -> None:
    body = _read("nbs_kphx_2023-07-14.csv")
    lines = body.decode("utf-8").splitlines(keepends=True)
    header, data_lines = lines[0], lines[1:]
    # Strip the 13Z (canonical) run, keep the off-cycle 01Z/07Z/19Z runs.
    kept = [line for line in data_lines if not line.startswith("2023-07-14 13:00:00")]
    stripped_body = (header + "".join(kept)).encode("utf-8")

    fetcher = FakeFetcher(
        [(lambda u: "2023-07" in u, HttpResponse(200, stripped_body))]
    )
    result = fetch_guidance(
        "KPHX", date(2023, 7, 14), date(2023, 7, 14), fetcher, REGIMES
    )

    assert result.rows == []
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.reason == "missing_run"
    assert gap.station == "KPHX"
    assert gap.expected == "2023-07-14T13:00Z"


def test_fetch_guidance_no_txn_gap_when_canonical_run_has_rows_but_no_txn() -> None:
    # Pre-v4.0 NBM (before autumn 2020) carries the daily max/min in a column
    # named `n_x`, not `txn` (build spec #7 orchestrator redirect); a
    # canonical run with rows but an all-null txn column must not pass
    # silently as if it had usable guidance.
    body = _read("nbs_kphx_2023-07-14.csv")
    lines = body.decode("utf-8").splitlines(keepends=True)
    header, data_lines = lines[0], lines[1:]
    txn_index = header.rstrip("\n").split(",").index("txn")

    def _blank_txn(line: str) -> str:
        if not line.startswith("2023-07-14 13:00:00"):
            return line
        cols = line.rstrip("\n").split(",")
        cols[txn_index] = ""
        return ",".join(cols) + "\n"

    blanked_body = (header + "".join(_blank_txn(line) for line in data_lines)).encode(
        "utf-8"
    )

    fetcher = FakeFetcher(
        [(lambda u: "2023-07" in u, HttpResponse(200, blanked_body))]
    )
    result = fetch_guidance(
        "KPHX", date(2023, 7, 14), date(2023, 7, 14), fetcher, REGIMES
    )

    assert result.rows  # the canonical run's rows are still kept
    assert all(row.txn is None for row in result.rows)
    assert len(result.gaps) == 1
    gap = result.gaps[0]
    assert gap.reason == "no_txn"
    assert gap.station == "KPHX"
    assert gap.expected == "2023-07-14T13:00Z"


def test_fetch_guidance_empty_body_emits_missing_run_for_every_day() -> None:
    fetcher = FakeFetcher(
        [(lambda _u: True, HttpResponse(200, _read("nbs_kphx_empty.csv")))]
    )
    result = fetch_guidance(
        "KPHX", date(2023, 7, 14), date(2023, 7, 16), fetcher, REGIMES
    )
    assert result.rows == []
    assert len(result.gaps) == 3
    assert {gap.reason for gap in result.gaps} == {"missing_run"}


def test_fetch_guidance_fetch_error_emits_http_error_gap_for_failed_chunk_only() -> (
    None
):
    fetcher = FakeFetcher(
        [
            (lambda u: "2023-07" in u, FetchError(status=503, reason="http_error:503")),
            (lambda u: "2023-08" in u, HttpResponse(200, _read("nbs_kphx_empty.csv"))),
        ]
    )
    result = fetch_guidance(
        "KPHX", date(2023, 7, 30), date(2023, 8, 2), fetcher, REGIMES
    )

    july_gaps = [g for g in result.gaps if g.expected.startswith("2023-07")]
    august_gaps = [g for g in result.gaps if g.expected.startswith("2023-08")]
    assert len(july_gaps) == 2  # 07-30, 07-31
    assert all(g.reason == "http_error:503" for g in july_gaps)
    assert len(august_gaps) == 2  # 08-01, 08-02
    assert all(g.reason == "missing_run" for g in august_gaps)


def test_fetch_guidance_requests_one_call_per_calendar_month() -> None:
    fetcher = FakeFetcher(
        [(lambda _u: True, HttpResponse(200, _read("nbs_kphx_empty.csv")))]
    )
    fetch_guidance("KPHX", date(2023, 6, 15), date(2023, 8, 5), fetcher, REGIMES)
    assert len(fetcher.requested_urls) == 3
    assert "sts=2023-06-15T00:00Z" in fetcher.requested_urls[0]
    assert "ets=2023-07-01T00:00Z" in fetcher.requested_urls[0]
    assert "sts=2023-08-01T00:00Z" in fetcher.requested_urls[2]
    assert "ets=2023-08-06T00:00Z" in fetcher.requested_urls[2]
