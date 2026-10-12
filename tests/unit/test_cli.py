"""wfa CLI: env var handling and command wiring (no network)."""

import json
import types
from datetime import date, datetime, time, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from weather_forecast_audit import challenger, cli, warehouse
from weather_forecast_audit.registry import load_registry

pytestmark = pytest.mark.unit


def test_main_requires_wfa_duckdb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DUCKDB_PATH", raising=False)
    with pytest.raises(SystemExit, match="WFA_DUCKDB_PATH"):
        cli.main(["init-db"])


def test_init_db_creates_raw_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "wfa.duckdb"
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    cli.main(["init-db"])

    with duckdb.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'raw'"
            ).fetchall()
        }
    assert "nbs_guidance" in tables


def test_export_requires_wfa_duckdb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DUCKDB_PATH", raising=False)
    with pytest.raises(SystemExit, match="WFA_DUCKDB_PATH"):
        cli.main(["export", "--out", "/tmp/wfa-export-does-not-run"])


def test_export_parser_requires_out_flag() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["export"])


def test_ingest_unknown_station_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(tmp_path / "wfa.duckdb"))

    with pytest.raises(SystemExit, match="unknown station"):
        cli.main(
            [
                "ingest",
                "--station",
                "ZZZZ",
                "--start",
                "2023-07-14",
                "--end",
                "2023-07-14",
            ]
        )


def test_predict_challenger_requires_start_and_end() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "challenger"])
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "challenger", "--start", "2023-07-14"])


def test_predict_challenger_has_no_model_constant_overrides() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["predict", "challenger", "--start", "2023-07-14", "--end", "2023-07-15"]
    )
    assert vars(args).keys() == {"command", "predict_model", "start", "end", "func"}
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "predict",
                "challenger",
                "--start",
                "2023-07-14",
                "--end",
                "2023-07-15",
                "--n-estimators",
                "5",
            ]
        )


# -- `wfa predict challenger`, driven for real over a synthetic pairs table -----


def _synthetic_pairs(stations: list[str], first: date, n_days: int) -> pl.DataFrame:
    """`int_raw_verification_pairs`-shaped rows (naive UTC, like raw.*).

    The NBS error is +3 F when its own `tmp_f` is warm and -3 F when cool, so
    a model that is handed `tmp_f` can learn it and one that is not cannot.
    """
    rng = np.random.default_rng(5)
    leads = {"max": (1, 2), "min": (1, 2, 3)}
    records = []
    for offset in range(n_days):
        run = first + timedelta(days=offset)
        runtime = datetime.combine(run, time(13, 0))
        for station in stations:
            for variable, lead_days in leads.items():
                for lead in lead_days:
                    target = run + timedelta(days=lead)
                    start = datetime.combine(target, time(12, 0))
                    end = datetime.combine(target + timedelta(days=1), time(6, 0))
                    if variable == "min":
                        start = datetime.combine(target, time(0, 0))
                        end = datetime.combine(target, time(18, 0))
                    tmp = float(rng.uniform(30.0, 90.0))
                    observed = float(rng.normal(60.0, 10.0))
                    error = (3.0 if tmp > 60.0 else -3.0) + float(rng.normal(0.0, 0.4))
                    records.append(
                        {
                            "station": station,
                            "run_date": run,
                            "runtime_utc": runtime,
                            "cycle_hour": 13,
                            "lead_day": lead,
                            "variable": variable,
                            "target_date": target,
                            "forecast_f": observed + error,
                            "spread_f": 3.0,
                            "tmp_f": tmp,
                            "window_start_utc": start,
                            "window_end_utc": end,
                            "observed_f": observed,
                            "scorable": True,
                        }
                    )
    return pl.DataFrame(records)


def _predictions(db_path: Path, where: str = "true") -> pl.DataFrame:
    with duckdb.connect(str(db_path)) as conn:
        return conn.execute(
            f"select * from raw.model_predictions where {where}"  # noqa: S608
        ).pl()


def test_predict_challenger_runs_monthly_over_the_range_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stations = sorted(load_registry())[:8]
    pairs = _synthetic_pairs(stations, date(2026, 1, 1), 105)
    db_path = tmp_path / "wfa.duckdb"
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 500)
    # Pairs beyond the end date exist in the table and must not matter.
    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        conn.execute("create table int_raw_verification_pairs as select * from pairs")
        # Another source's rows, in range and out of range, and a stale
        # challenger row in range plus one out of range.
        for source, run_date_, params in [
            ("baseline", "2026-02-10", "keep"),
            ("baseline", "2026-04-20", "keep"),
            ("challenger", "2026-02-12", "stale"),
            ("challenger", "2026-04-25", "outside"),
        ]:
            conn.execute(
                "insert into raw.model_predictions values "
                "('KPHX', ?, '2026-01-01 13:00:00', 1, 'max', '2026-01-02', ?, "
                "70.0, 70.0, null, null, true, ?, '2026-01-01 00:00:00')",
                [run_date_, source, params],
            )

    # A clock whose successive readings are cubes, so each fit's duration,
    # their sum, max, min, mean and median are all different numbers.
    ticks = iter(range(1, 1000))
    clock = types.SimpleNamespace(monotonic=lambda: float(next(ticks)) ** 3)
    monkeypatch.setattr(challenger, "time", clock)
    monkeypatch.setattr(cli, "time", clock)

    argv = ["predict", "challenger", "--start", "2026-01-05", "--end", "2026-04-10"]
    cli.main(argv)
    out = capsys.readouterr().out

    result = _predictions(db_path, "source = 'challenger' and params != 'outside'")
    n_days = (date(2026, 4, 10) - date(2026, 1, 5)).days + 1
    assert result.height == n_days * len(stations) * 5
    assert result["run_date"].min() == date(2026, 1, 5)
    assert result["run_date"].max() == date(2026, 4, 10)
    assert (
        result.select(
            pl.struct("station", "run_date", "lead_day", "variable").n_unique()
        ).item()
        == result.height
    )

    # Other sources and out-of-range challenger rows are untouched; the stale
    # in-range challenger row was replaced.
    assert _predictions(db_path, "source = 'baseline'").height == 2
    assert _predictions(db_path, "params = 'outside'").height == 1
    assert _predictions(db_path, "params = 'stale'").height == 0

    # Four calendar months retrain once each; January has too few pairs.
    fallback_share = float(result["fallback"].mean())
    assert 0.0 < fallback_share < 1.0
    assert (
        f"rows={result.height} fallback_share={fallback_share:.3f} "
        "retrains=4 fitted=3 "
        "fit_seconds_total=297.00 fit_seconds_max=169.00 fit_seconds_median=91.00 "
        "wall_time=728.00s"
    ) in out

    recorded = json.loads(result["params"][0])
    assert recorded["retrain"] == "monthly"
    assert recorded["min_train_rows"] == 500
    assert recorded["n_estimators"] == 200
    assert recorded["seed"] == 42
    assert recorded["objective"] == "regression_l1"
    assert result["params"].n_unique() == 1

    # It really used tmp_f and the registry: the corrections learn the +-3 F
    # error in the last month, where a fit existed.
    late = result.filter(pl.col("run_date") >= date(2026, 3, 5))
    truth = pairs.select("station", "run_date", "lead_day", "variable", "observed_f")
    joined = late.join(truth, on=["station", "run_date", "lead_day", "variable"])
    raw_mae = float((joined["raw_forecast_f"] - joined["observed_f"]).abs().mean())
    fixed_mae = float((joined["forecast_f"] - joined["observed_f"]).abs().mean())
    assert fixed_mae < 0.5 * raw_mae

    # Idempotent: a second identical run replaces, never duplicates.
    ticks = iter(range(1, 1000))
    cli.main(argv)
    again = _predictions(db_path, "source = 'challenger' and params != 'outside'")
    assert again.height == result.height
    assert _predictions(db_path, "source = 'baseline'").height == 2


def test_challenger_station_features_come_from_the_registry() -> None:
    registry = load_registry()
    frame = cli._station_features(registry)  # noqa: SLF001

    assert frame.height == len(registry)
    for row in frame.filter(
        pl.col("station").is_in(["KPHX", "KSEA", "KMIA", "KDEN"])
    ).iter_rows(named=True):
        station = registry[row["station"]]
        assert row["latitude"] == station.lat
        assert row["longitude"] == station.lon
        assert row["elevation_m"] == station.elevation_m
        assert row["coastal_flag"] == station.coastal_flag
        assert row["climate_region"] == station.climate_region
    assert frame.filter(pl.col("station") == "KSEA")["coastal_flag"].item() is True
    assert frame.filter(pl.col("station") == "KPHX")["coastal_flag"].item() is False
