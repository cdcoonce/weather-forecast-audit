"""Shared pytest fixtures.

`DagsterFixtureFetcher` serves the recorded KPHX/KORD IEM payloads (build
spec #10) for the Dagster asset tests under `tests/dagster/`. It is
deliberately separate from `tests/integration/test_tracer_pipeline.py`'s own
`FixtureFetcher`/fixture set (per the build spec: "do not import it from the
integration test"), even though both read from `tests/fixtures/iem/`.
"""

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from weather_forecast_audit.iem.http import HttpResponse
from weather_forecast_audit.resources import IemResource

pytest_plugins = ["pytester"]

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "iem"

_NBS_FILES = {
    ("KPHX", "2023-07-14"): "nbs_kphx_2023-07-14.csv",
    ("KORD", "2023-07-14"): "nbs_kord_2023-07-14.csv",
}

_ASOS_RANGE_FILES = {
    "PHX": "asos_kphx_2023-07-13_2023-07-18.csv",
    "ORD": "asos_kord_2023-07-13_2023-07-18.csv",
}

_CLI_FILES = {
    "KPHX": "cli_kphx_2023.json",
    "KORD": "cli_kord_2023.json",
}


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _clip_asos_to_range(body: bytes, sts: datetime, ets: datetime) -> bytes:
    """The recorded ASOS fixture clipped to [sts, ets), as the service does.

    Each single-day partition materialization requests a different, narrower
    [sts, ets) than the fixture's own recorded range, so this must actually
    clip rather than return the whole file -- otherwise a too-narrow fetch
    would go undetected (mirrors
    tests/integration/test_tracer_pipeline.py's `_asos_in_requested_range`).
    """
    header, *rows = body.decode().splitlines()
    kept = [
        row
        for row in rows
        if sts
        <= datetime.strptime(row.split(",")[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        < ets
    ]
    return ("\n".join([header, *kept]) + "\n").encode()


class DagsterFixtureFetcher:
    """Routes NBS/ASOS/CLI requests to the recorded KPHX/KORD fixtures."""

    def get(self, url: str) -> HttpResponse:
        query = parse_qs(urlparse(url).query)
        if "mos.py" in url:
            station = query["station"][0]
            run_date = query["sts"][0][:10]
            key = (station, run_date)
            if key not in _NBS_FILES:
                raise AssertionError(f"unexpected mos.py request: {url}")
            return HttpResponse(200, _read(_NBS_FILES[key]))
        if "asos.py" in url:
            code = query["station"][0]
            if code not in _ASOS_RANGE_FILES:
                raise AssertionError(f"unexpected asos.py request: {url}")
            sts = datetime.fromisoformat(query["sts"][0].replace("Z", "+00:00"))
            ets = datetime.fromisoformat(query["ets"][0].replace("Z", "+00:00"))
            body = _read(_ASOS_RANGE_FILES[code])
            return HttpResponse(200, _clip_asos_to_range(body, sts, ets))
        if "cli.py" in url:
            station = query["station"][0]
            if station not in _CLI_FILES:
                raise AssertionError(f"unexpected cli.py request: {url}")
            return HttpResponse(200, _read(_CLI_FILES[station]))
        raise AssertionError(f"unexpected request: {url}")


class FixtureIemResource(IemResource):
    """An `IemResource` whose `fetcher()` returns the fixture fetcher (D5)."""

    def fetcher(self) -> DagsterFixtureFetcher:
        return DagsterFixtureFetcher()
