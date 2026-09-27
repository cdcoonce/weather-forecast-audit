"""Shared IEM fixture fetcher for DB-backed tests.

Routes NBS/ASOS/CLI requests to the recorded fixtures in `tests/fixtures/iem`
by URL shape, the way `tests/integration/test_tracer_pipeline.py` originally
defined it. `tests/integration/test_export_tracer.py` reuses the same
fetcher (imported, not copied) so both tests build the warehouse from one
recipe.
"""

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from weather_forecast_audit.iem.http import HttpResponse

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"

ASOS_HEADER_ONLY = b"station,valid,tmpf,metar\n"
CLI_EMPTY = b'{"results": []}'


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _asos_in_requested_range(url: str) -> bytes:
    """The ASOS fixture clipped to the request's [sts, ets), as the service does.

    Serving the whole file regardless of range would hide a pipeline that
    fetches too narrow a span: its lead-2/3 windows would still find their
    observations here, though the live service would not return them.
    """
    query = parse_qs(urlparse(url).query)
    sts = datetime.fromisoformat(query["sts"][0].replace("Z", "+00:00"))
    ets = datetime.fromisoformat(query["ets"][0].replace("Z", "+00:00"))
    header, *rows = _read("asos_kphx_2023-07-13_2023-07-17.csv").decode().splitlines()
    kept = [
        row
        for row in rows
        if sts
        <= datetime.strptime(row.split(",")[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        < ets
    ]
    return ("\n".join([header, *kept]) + "\n").encode()


class FixtureFetcher:
    """Routes NBS/ASOS/CLI requests to the recorded fixtures by URL shape.

    ASOS/CLI serve the 2023 fixture for 2023 requests and an empty payload
    for 2026 requests, matching the build spec's integration-test recipe.
    """

    def get(self, url: str) -> HttpResponse:
        if "mos.py" in url:
            if "sts=2023-07" in url:
                return HttpResponse(200, _read("nbs_kphx_2023-07-14.csv"))
            if "sts=2026-04" in url:
                return HttpResponse(200, _read("nbs_kphx_2026-04-29.csv"))
            if "sts=2026-05" in url:
                return HttpResponse(200, _read("nbs_kphx_2026-05-06.csv"))
            raise AssertionError(f"unexpected mos.py request: {url}")
        if "asos.py" in url:
            if "sts=2023-07" in url:
                return HttpResponse(200, _asos_in_requested_range(url))
            if "sts=2026" in url:
                return HttpResponse(200, ASOS_HEADER_ONLY)
            raise AssertionError(f"unexpected asos.py request: {url}")
        if "cli.py" in url:
            if "year=2023" in url:
                return HttpResponse(200, _read("cli_kphx_2023.json"))
            if "year=2026" in url:
                return HttpResponse(200, CLI_EMPTY)
            raise AssertionError(f"unexpected cli.py request: {url}")
        raise AssertionError(f"unexpected request: {url}")
