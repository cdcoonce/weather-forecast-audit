"""End-to-end: fixture-served ingest -> dbt build -> `wfa predict baseline` ->
dbt build -> baseline rows in `fct_forecast_verification` (build spec #18,
test 8).

Reuses `tests/integration/test_tracer_pipeline.py`'s three recorded KPHX
fixture windows and its `FixtureFetcher` routing pattern (duplicated here,
not imported, matching that test's own relationship to
`tests/conftest.py`'s separate Dagster fixture fetcher) so history is sparse
enough that most predictions fall back -- which is fine, and expected, for
this slice.
"""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import pytest
from dbt.cli.main import dbtRunner

from weather_forecast_audit import cli, warehouse
from weather_forecast_audit.iem.http import HttpResponse
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry

pytestmark = [pytest.mark.integration, pytest.mark.io, pytest.mark.dbt]

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"
DBT_DIR = REPO / "dbt"

ASOS_HEADER_ONLY = b"station,valid,tmpf,metar\n"
CLI_EMPTY = json.dumps({"results": []}).encode("utf-8")


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _asos_in_requested_range(url: str) -> bytes:
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
    """Routes NBS/ASOS/CLI requests to the recorded KPHX fixtures by URL shape."""

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


def _run_ingest(db_path: Path) -> None:
    station = load_registry()["KPHX"]
    regimes = load_cycle_regimes()
    fetcher = FixtureFetcher()
    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        for start, end in [
            (date(2023, 7, 14), date(2023, 7, 14)),
            (date(2026, 4, 29), date(2026, 4, 29)),
            (date(2026, 5, 6), date(2026, 5, 6)),
        ]:
            ingest_station(conn, station, start, end, fetcher, regimes)


def _dbt_build(tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("DBT_TARGET_PATH", str(tmp_path / "target"))
    monkeypatch.setenv("DBT_LOG_PATH", str(tmp_path / "logs"))
    for name in [n for n in os.environ if n.startswith("WFA_SNOWFLAKE_")]:
        monkeypatch.delenv(name)
    result = dbtRunner().invoke(
        ["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    )
    assert result.success, result.exception


def test_baseline_predictions_land_in_fct_forecast_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "baseline_pipeline.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    cli.main(
        ["predict", "baseline", "--start", "2023-07-14", "--end", "2026-05-06"]
    )

    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        raw_count = conn.execute(
            "select count(*) from fct_forecast_verification where source = 'raw_nbm'"
        ).fetchone()[0]
        baseline_count = conn.execute(
            "select count(*) from fct_forecast_verification where source = 'baseline'"
        ).fetchone()[0]
        # `fallback` lives on stg_model_predictions (raw.model_predictions),
        # not on fct_forecast_verification's own column set (build spec #18
        # D3.3): fct only carries the corrected forecast_f/error_f.
        fallback_mismatches = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'baseline' and fallback "
            "and forecast_f != raw_forecast_f"
        ).fetchone()[0]
        # Sanity: with only three sparse ingested run dates, most groups
        # never reach MIN_PAIRS, so this asserts the expected regime rather
        # than a coincidence.
        fallback_count = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'baseline' and fallback"
        ).fetchone()[0]
        dupes = conn.execute(
            "select station, run_date, lead_day, variable, source, count(*) "
            "from fct_forecast_verification "
            "group by station, run_date, lead_day, variable, source "
            "having count(*) > 1"
        ).fetchall()

    assert raw_count > 0
    assert baseline_count == raw_count
    assert fallback_mismatches == 0
    assert fallback_count > 0
    assert dupes == []
