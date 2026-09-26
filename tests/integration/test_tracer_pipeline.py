"""End-to-end tracer: fixture-served ingest -> dbt build -> known answers.

FakeFetcher-served fixtures stand in for the three live runs the tracer
script makes for real (`scripts/tracer_kphx.sh`): the hand-checked KPHX
2023-07-14 run, plus one run either side of the 2026-04-30 cycle changeover.
"""

import json
import os
from datetime import date
from pathlib import Path

import duckdb
import pytest
from dbt.cli.main import dbtRunner

from weather_forecast_audit import warehouse
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

KNOWN_ANSWERS = [
    # target_date, lead, variable, forecast, observed, error, hours_covered
    (date(2023, 7, 15), 1, "min", 93.0, 93.0, 0.0, 18),
    (date(2023, 7, 15), 1, "max", 118.0, 117.0, 1.0, 18),
    (date(2023, 7, 16), 2, "min", 93.0, 94.0, -1.0, 18),
    (date(2023, 7, 16), 2, "max", 118.0, 113.0, 5.0, 18),
    (date(2023, 7, 17), 3, "min", 94.0, None, None, 12),
]


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


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
                return HttpResponse(200, _read("asos_kphx_2023-07-13_2023-07-17.csv"))
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


def test_tracer_pipeline_known_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        rows = conn.execute(
            "select target_date, lead_day, variable, forecast_f, observed_f, "
            "error_f, hours_covered, scorable "
            "from fct_forecast_verification "
            "where station = 'KPHX' and run_date = '2023-07-14'"
        ).fetchall()

    by_key = {(r[0], r[1], r[2]): r for r in rows}
    assert len(by_key) == len(KNOWN_ANSWERS)
    for target, lead, variable, forecast, observed, error, hours in KNOWN_ANSWERS:
        row = by_key[(target, lead, variable)]
        assert row[3] == forecast
        assert row[4] == observed
        assert row[5] == error
        assert row[6] == hours
        assert row[7] == (observed is not None)


def test_tracer_pipeline_2026_rows_carry_cycle_hour_and_are_unscorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        rows_0429 = conn.execute(
            "select cycle_hour, scorable, error_f from fct_forecast_verification "
            "where run_date = '2026-04-29'"
        ).fetchall()
        rows_0506 = conn.execute(
            "select cycle_hour, scorable, error_f from fct_forecast_verification "
            "where run_date = '2026-05-06'"
        ).fetchall()

    assert rows_0429
    assert all(row[0] == 13 for row in rows_0429)
    assert all(row[1] is False and row[2] is None for row in rows_0429)

    assert rows_0506
    assert all(row[0] == 12 for row in rows_0506)
    assert all(row[1] is False and row[2] is None for row in rows_0506)


def test_tracer_pipeline_gaps_has_missing_observations_for_2026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        gaps = conn.execute(
            "select expected, reason from stg_ingest_gaps "
            "where source = 'asos' and reason = 'missing_observations' "
            "and expected like '2026%'"
        ).fetchall()

    assert gaps


def test_ingest_twice_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "tracer.duckdb"
    tables = [
        "nbs_guidance",
        "asos_hourly",
        "cli_daily",
        "ingest_gaps",
        "resolved_windows",
    ]

    _run_ingest(db_path)
    with duckdb.connect(str(db_path)) as conn:
        counts_first = {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()
            for table in tables
        }

    _run_ingest(db_path)
    with duckdb.connect(str(db_path)) as conn:
        counts_second = {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()
            for table in tables
        }

    assert counts_first == counts_second
    assert all(count[0] > 0 for count in counts_first.values())
