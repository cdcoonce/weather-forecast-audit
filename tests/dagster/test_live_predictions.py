"""`live_predictions`: one run date issued into the live log, as an asset."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import dagster as dg
import duckdb
import polars as pl
import pytest
from support.synthetic_pairs import make_pairs, runtime_of

from weather_forecast_audit import challenger, warehouse
from weather_forecast_audit.assets import live_predictions
from weather_forecast_audit.definitions import defs
from weather_forecast_audit.prediction_log import verify_immutability
from weather_forecast_audit.registry import load_registry
from weather_forecast_audit.resources import ClockResource, WarehouseResource

pytestmark = [pytest.mark.dagster, pytest.mark.io]

RUN_DATE = "2026-03-10"
RUN_DAY = datetime.fromisoformat(RUN_DATE).date()


def _db(tmp_path: Path) -> Path:
    stations = sorted(load_registry())[:3]
    days = [RUN_DAY - timedelta(days=k) for k in range(20, -1, -1)]
    pairs: pl.DataFrame = make_pairs(days, stations)  # noqa: F841
    path = tmp_path / "wfa.duckdb"
    with duckdb.connect(str(path)) as conn:
        warehouse.init_db(conn)
        conn.execute("create table int_raw_verification_pairs as select * from pairs")
    return path


def _materialize(path: Path, now: datetime) -> dg.ExecuteInProcessResult:
    return dg.materialize(
        [live_predictions],
        partition_key=RUN_DATE,
        resources={
            "warehouse_resource": WarehouseResource(duckdb_path=str(path)),
            "clock": ClockResource(now_override=now.isoformat()),
        },
        raise_on_error=False,
    )


def test_materializing_a_partition_issues_and_seals_its_run_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 50)
    path = _db(tmp_path)
    now = runtime_of(RUN_DAY).replace(tzinfo=UTC) + timedelta(hours=1)

    result = _materialize(path, now)

    assert result.success
    [event] = result.get_asset_materialization_events()
    metadata = event.materialization.metadata
    assert metadata["late_rejected"].value == 0
    assert metadata["retrained"].value is True
    assert metadata["sealed"].value is True
    with duckdb.connect(str(path)) as conn:
        n_rows = conn.execute(
            "select count(*) from live.prediction_log where run_date = ?", [RUN_DAY]
        ).fetchone()
        assert verify_immutability(conn).ok
    assert n_rows is not None
    assert metadata["rows_appended"].value == n_rows[0] > 0
    with duckdb.connect(str(path)) as conn:
        chain = conn.execute(
            "select chain_sha256 from live.partition_seals where run_date = ?",
            [RUN_DAY],
        ).fetchone()
        stored = conn.execute(
            "select retrained_on, fit_seconds from live.challenger_models"
        ).fetchone()
    assert chain is not None
    assert stored is not None
    assert metadata["run_date"].value == RUN_DATE
    assert metadata["chain_sha256"].value == chain[0]
    assert metadata["challenger_retrained_on"].value == str(stored[0]) == RUN_DATE
    assert metadata["fit_seconds"].value == stored[1] > 0.0
    assert metadata["history_rows"].value > 0
    assert metadata["wall_seconds"].value > 0.0


def test_late_rows_fail_the_run_without_retries_and_leave_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 50)
    path = _db(tmp_path)
    # 19:00 the next day: the lead-1 min windows (18:00) have closed.
    now = runtime_of(RUN_DAY).replace(tzinfo=UTC) + timedelta(hours=30)

    result = _materialize(path, now)

    assert not result.success
    [failure_event] = result.get_step_failure_events()
    failure = failure_event.step_failure_data.error
    assert failure is not None
    assert "rejected as late" in str(failure)
    with duckdb.connect(str(path)) as conn:
        sealed = conn.execute(
            "select n_late_rejected from live.partition_seals where run_date = ?",
            [RUN_DAY],
        ).fetchone()
        assert verify_immutability(conn).ok
    assert sealed is not None
    assert sealed[0] == 6


def test_the_late_failure_is_not_retryable_and_names_the_late_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 50)
    path = _db(tmp_path)
    now = runtime_of(RUN_DAY).replace(tzinfo=UTC) + timedelta(hours=30)

    with pytest.raises(dg.Failure) as raised:
        live_predictions(
            dg.build_asset_context(partition_key=RUN_DATE),
            WarehouseResource(duckdb_path=str(path)),
            ClockResource(now_override=now.isoformat()),
        )

    assert raised.value.allow_retries is False
    assert len(raised.value.metadata["late_keys"].value) == 6


def test_the_issuing_job_selects_only_the_live_predictions_asset() -> None:
    job = defs.resolve_job_def("issue_predictions_job")

    assert {key.to_user_string() for key in job.asset_layer.executable_asset_keys} == {
        "live_predictions"
    }
