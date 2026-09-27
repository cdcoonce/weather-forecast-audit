"""Shared pytest fixtures.

`DagsterFixtureFetcher` serves the recorded KPHX/KORD IEM payloads (build
spec #10) for the Dagster asset tests under `tests/dagster/`. It is
deliberately separate from `tests/integration/test_tracer_pipeline.py`'s own
`FixtureFetcher`/fixture set (per the build spec: "do not import it from the
integration test"), even though both read from `tests/fixtures/iem/`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dagster import Definitions

from weather_forecast_audit.assets import (
    FRESHNESS_CHECKS,
    RAW_INGEST_ASSETS,
    ingest_gaps_spec,
)
from weather_forecast_audit.iem.http import HttpResponse
from weather_forecast_audit.resources import (
    ClockResource,
    IemResource,
    StationsResource,
    WarehouseResource,
)

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


class OneStationMissingGuidanceFetcher(DagsterFixtureFetcher):
    """As `DagsterFixtureFetcher`, but KORD's NBS run is missing for 07-14.

    Engineers a real gap (`missing_run`) for exactly one of two stations, so
    tests can assert both "gaps are recorded where the fixtures lack data"
    and a >5% gap rate (1 of 2 stations = 50%), per the build spec's own
    suggested example ("one station with a missing run").
    """

    def get(self, url: str) -> HttpResponse:
        if "mos.py" in url and "station=KORD" in url:
            return HttpResponse(200, _read("nbs_kphx_empty.csv"))
        return super().get(url)


class OneStationMissingGuidanceIemResource(FixtureIemResource):
    def fetcher(self) -> OneStationMissingGuidanceFetcher:
        return OneStationMissingGuidanceFetcher()


class OneStationMissingCliReportFetcher(DagsterFixtureFetcher):
    """As `DagsterFixtureFetcher`, but KORD's CLI report is missing for 07-14.

    Unlike a missing guidance run, a missing CLI report doesn't cascade:
    `pipeline.resolve_station` never reads `raw.cli_daily`, so both stations
    still get full guidance/asos/resolved rows -- letting a materialize test
    show both "gaps recorded where fixtures lack data" (this gap) and "rows
    exist for both stations in every raw table" (nothing else is missing) at
    once, and letting a downstream dbt-build test still see both stations in
    fct_forecast_verification.
    """

    def get(self, url: str) -> HttpResponse:
        if "cli.py" in url and "station=KORD" in url:
            return HttpResponse(200, json.dumps({"results": []}).encode())
        return super().get(url)


class OneStationMissingCliReportIemResource(FixtureIemResource):
    def fetcher(self) -> OneStationMissingCliReportFetcher:
        return OneStationMissingCliReportFetcher()


def build_ingest_test_defs(
    duckdb_path: str, only: list[str], iem_resource: IemResource
) -> Definitions:
    """A standalone `Definitions` binding the real raw ingest assets to
    test-scoped resources (a temp DuckDB path, a fixture fetcher, a station
    subset). `defs.resolve_job_def(...)`'s returned job is already bound to
    the *production* resources, and dagster refuses a conflicting `resources=`
    override at `execute_in_process` time (resource identity must match by
    reference) -- so tests build their own `Definitions` from the same
    asset/job objects instead.
    """
    from weather_forecast_audit.definitions import ingest_job

    return Definitions(
        assets=[*RAW_INGEST_ASSETS, ingest_gaps_spec],
        asset_checks=FRESHNESS_CHECKS,
        jobs=[ingest_job],
        resources={
            "warehouse_resource": WarehouseResource(duckdb_path=duckdb_path),
            "iem": iem_resource,
            "stations": StationsResource(only=only),
            "clock": ClockResource(),
        },
    )
