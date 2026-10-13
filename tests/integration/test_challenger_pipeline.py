"""End-to-end: fixture-served ingest -> dbt build -> `wfa predict challenger`
-> dbt build -> challenger rows in `fct_forecast_verification` (issue #19).

Reuses the three recorded KPHX fixture windows and the `FixtureFetcher`
routing pattern of `test_baseline_pipeline.py` (duplicated, not imported, as
that file does with the tracer test). History is far below the 20,000-row
minimum, so by default every prediction falls back to raw NBM, which is the
expected regime for this slice; a second test lowers the minimum through the
module constant the CLI reads, to drive the real LightGBM fit through the CLI.
"""

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import lightgbm
import pytest
from dbt.cli.main import dbtRunner

from weather_forecast_audit import challenger, cli, warehouse
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


def _predict_and_rebuild(
    tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    cli.main(["predict", "challenger", "--start", "2023-07-14", "--end", "2026-05-06"])
    _dbt_build(tmp_path, db_path, monkeypatch)


def test_challenger_predictions_land_in_fct_forecast_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "challenger_pipeline.duckdb"
    _predict_and_rebuild(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        raw_count = conn.execute(
            "select count(*) from fct_forecast_verification where source = 'raw_nbm'"
        ).fetchone()[0]
        challenger_count = conn.execute(
            "select count(*) from fct_forecast_verification "
            "where source = 'challenger'"
        ).fetchone()[0]
        params_rows = conn.execute(
            "select distinct params from raw.model_predictions "
            "where source = 'challenger'"
        ).fetchall()
        sources = conn.execute(
            "select distinct source from raw.model_predictions"
        ).fetchall()
        # Below the minimum row count every prediction is raw NBM, flagged.
        not_fallback = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'challenger' and not fallback"
        ).fetchone()[0]
        mismatched_forecast = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'challenger' and forecast_f != raw_forecast_f"
        ).fetchone()[0]
        # Same observation for the same pair under both sources.
        paired = conn.execute(
            "select count(*) from fct_forecast_verification as c "
            "inner join fct_forecast_verification as r "
            "on c.station = r.station and c.run_date = r.run_date "
            "and c.lead_day = r.lead_day and c.variable = r.variable "
            "where c.source = 'challenger' and r.source = 'raw_nbm'"
        ).fetchone()[0]
        different_observation = conn.execute(
            "select count(*) from fct_forecast_verification as c "
            "inner join fct_forecast_verification as r "
            "on c.station = r.station and c.run_date = r.run_date "
            "and c.lead_day = r.lead_day and c.variable = r.variable "
            "where c.source = 'challenger' and r.source = 'raw_nbm' "
            "and (c.observed_f is distinct from r.observed_f "
            "or c.tmp_f is distinct from r.tmp_f "
            "or c.scorable is distinct from r.scorable)"
        ).fetchone()[0]
        non_null_tmp = conn.execute(
            "select count(*) from fct_forecast_verification "
            "where source = 'challenger' and tmp_f is not null"
        ).fetchone()[0]
        dupes = conn.execute(
            "select station, run_date, lead_day, variable, source, count(*) "
            "from fct_forecast_verification "
            "group by station, run_date, lead_day, variable, source "
            "having count(*) > 1"
        ).fetchall()

    assert raw_count > 0
    assert challenger_count == raw_count
    assert sorted(row[0] for row in sources) == ["challenger"]
    assert not_fallback == 0
    assert mismatched_forecast == 0
    assert paired == raw_count
    assert different_observation == 0
    assert non_null_tmp > 0
    assert dupes == []

    assert len(params_rows) == 1
    params = json.loads(params_rows[0][0])
    assert params["n_estimators"] == challenger.N_ESTIMATORS == 200
    assert params["learning_rate"] == 0.05
    assert params["num_leaves"] == 31
    assert params["min_data_in_leaf"] == 500
    assert params["feature_fraction"] == 0.8
    assert params["bagging_fraction"] == 0.8
    assert params["bagging_freq"] == 1
    assert params["lambda_l2"] == 10.0
    assert params["max_bin"] == 63
    assert params["seed"] == 42
    assert params["objective"] == "regression_l1"
    assert params["min_train_rows"] == 20_000
    assert params["lightgbm_version"] == lightgbm.__version__

    out = capsys.readouterr().out
    assert "wfa predict challenger 2023-07-14..2026-05-06" in out
    assert "fallback_share=1.000" in out
    assert "retrains=" in out
    assert "Run `dbt build` next" in out


def test_cli_drives_a_real_fit_when_the_minimum_is_lowered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 1)
    db_path = tmp_path / "challenger_fit.duckdb"
    _predict_and_rebuild(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        trained = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'challenger' and not fallback"
        ).fetchone()[0]
        nulls = conn.execute(
            "select count(*) from stg_model_predictions "
            "where source = 'challenger' and forecast_f is null"
        ).fetchone()[0]
        recorded = json.loads(
            conn.execute(
                "select distinct params from raw.model_predictions "
                "where source = 'challenger'"
            ).fetchone()[0]
        )

    assert trained > 0
    assert nulls == 0
    assert recorded["min_train_rows"] == 1
