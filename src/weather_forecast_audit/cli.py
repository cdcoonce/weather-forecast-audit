"""`wfa` command-line interface: `init-db`, `ingest`, `predict baseline`,
`predict challenger`, `export`, `log verify`, and `log issue`.

The DuckDB path always comes from `WFA_DUCKDB_PATH`, never a flag: one
database per environment (local dev, CI, the tracer script), set once.
"""

import argparse
import json
import os
import statistics
import time
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import polars as pl

from weather_forecast_audit import challenger as challenger_module
from weather_forecast_audit import export as export_module
from weather_forecast_audit import issuance, prediction_log, warehouse
from weather_forecast_audit.baseline import MIN_PAIRS, WINDOW_DAYS, BaselineModel
from weather_forecast_audit.challenger import (
    PAIR_COLUMNS,
    ChallengerModel,
    station_feature_frame,
)
from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import Station, load_registry
from weather_forecast_audit.walkforward import walk_forward


def _db_path() -> str:
    path = os.environ.get("WFA_DUCKDB_PATH")
    if not path:
        msg = "WFA_DUCKDB_PATH must be set to the DuckDB database file path"
        raise SystemExit(msg)
    return path


def _cmd_init_db(args: argparse.Namespace) -> None:
    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        warehouse.init_db(conn)
    print(f"wfa init-db: raw schema ready in {db_path}")


def _cmd_ingest(args: argparse.Namespace) -> None:
    db_path = _db_path()
    registry = load_registry()
    station = registry.get(args.station)
    if station is None:
        known = ", ".join(sorted(registry))
        msg = f"unknown station {args.station!r}; known stations: {known}"
        raise SystemExit(msg)

    regimes = load_cycle_regimes()
    fetcher = UrllibFetcher()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    with duckdb.connect(db_path) as conn:
        warehouse.init_db(conn)
        summary = ingest_station(conn, station, start, end, fetcher, regimes)

    gaps = ", ".join(
        f"{reason}={count}" for reason, count in sorted(summary.gaps_by_reason.items())
    )
    print(
        f"wfa ingest {summary.station} {start}..{end}: "
        f"guidance={summary.guidance_rows} asos={summary.asos_rows} "
        f"cli={summary.cli_rows} resolved={summary.resolved_rows} "
        f"gaps=[{gaps or 'none'}]"
    )


def _cmd_predict_baseline(args: argparse.Namespace) -> None:
    db_path = _db_path()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    model = BaselineModel(window_days=args.window_days, min_pairs=args.min_pairs)

    started = time.monotonic()
    with duckdb.connect(db_path) as conn:
        warehouse.init_db(conn)
        # "All rows whose window_end_utc is before the end date's issuance"
        # is simplest as all rows: the evaluator's own leakage guard (#17)
        # already restricts what each retrain actually trains on.
        pairs = conn.execute("select * from int_raw_verification_pairs").pl()
        result = walk_forward(pairs, model, start, end, retrain="daily")
        params = json.dumps(
            {"window_days": model.window_days, "min_pairs": model.min_pairs}
        )
        rows_frame = result.with_columns(pl.lit(params).alias("params"))
        warehouse.load_predictions(
            conn,
            model.name,
            start,
            end,
            rows_frame,
            now=datetime.now(UTC).replace(tzinfo=None),
        )
    wall_time = time.monotonic() - started

    n = result.height
    fallback_share = float(result["fallback"].mean()) if n > 0 else 0.0
    print(
        f"wfa predict baseline {start}..{end}: rows={n} "
        f"fallback_share={fallback_share:.3f} wall_time={wall_time:.2f}s"
    )
    print("Run `dbt build` next to refresh fct_forecast_verification.")


def _station_features(registry: dict[str, Station]) -> pl.DataFrame:
    return station_feature_frame(registry)


def _cmd_predict_challenger(args: argparse.Namespace) -> None:
    db_path = _db_path()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    # The minimum is read from the module at call time (not bound as a
    # default argument) so a test can lower it; there is deliberately no CLI
    # override of any model constant: a run is one set of pre-registered
    # constants, recorded in `params`.
    model = ChallengerModel(
        _station_features(load_registry()),
        min_train_rows=challenger_module.MIN_TRAIN_ROWS,
    )

    started = time.monotonic()
    with duckdb.connect(db_path) as conn:
        warehouse.init_db(conn)
        # Rows from runs after `end` can neither be predicted nor reach a
        # training set (their windows close after `end`'s issuance), so they
        # are not read at all.
        columns = ", ".join(PAIR_COLUMNS)
        pairs = conn.execute(
            f"select {columns} from int_raw_verification_pairs where run_date <= ?",
            [end],
        ).pl()
        result = walk_forward(pairs, model, start, end, retrain="monthly")
        del pairs
        params = json.dumps(model.describe_params())
        rows_frame = result.with_columns(pl.lit(params).alias("params"))
        warehouse.load_predictions(
            conn,
            model.name,
            start,
            end,
            rows_frame,
            now=datetime.now(UTC).replace(tzinfo=None),
        )
    wall_time = time.monotonic() - started

    n = result.height
    fallback_share = float(result["fallback"].mean()) if n > 0 else 0.0
    fit_log = model.fit_seconds_log
    print(
        f"wfa predict challenger {start}..{end}: rows={n} "
        f"fallback_share={fallback_share:.3f} retrains={model.fit_count} "
        f"fitted={len(fit_log)} "
        f"fit_seconds_total={sum(fit_log):.2f} "
        f"fit_seconds_max={max(fit_log, default=0.0):.2f} "
        f"fit_seconds_median={statistics.median(fit_log) if fit_log else 0.0:.2f} "
        f"wall_time={wall_time:.2f}s"
    )
    print("Run `dbt build` next to refresh fct_forecast_verification.")


def _cmd_export(args: argparse.Namespace) -> None:
    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        summary = export_module.export(conn, Path(args.out))
    print(
        f"wfa export: {summary.station_count} stations, "
        f"data_through={summary.data_through}, {len(summary.files)} files "
        f"-> {summary.out_dir}"
    )


def _cmd_log_verify(args: argparse.Namespace) -> None:
    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        # An old database has no `live` schema: that is an empty log, which
        # verifies, not a failure to find a table.
        prediction_log.init_log(conn)
        report = prediction_log.verify_immutability(conn)
    head = report.head_chain_sha256[:12]
    if report.ok:
        print(f"OK partitions={report.partitions} rows={report.rows} head={head}")
        return
    print(
        f"FAIL partitions={report.partitions} rows={report.rows} head={head} "
        f"problems={len(report.problems)}"
    )
    for problem in report.problems:
        print(problem)
    raise SystemExit(1)


def _cmd_log_issue(args: argparse.Namespace) -> None:
    db_path = _db_path()
    run_date = date.fromisoformat(args.run_date)
    # Read at call time (not a default argument) so a test can lower it; like
    # `predict challenger` there is no override of any model constant.
    features = _station_features(load_registry())

    with duckdb.connect(db_path) as conn:
        warehouse.init_db(conn)
        result = issuance.issue_predictions(
            conn,
            run_date,
            now=datetime.now(UTC),
            station_features=features,
            min_train_rows=challenger_module.MIN_TRAIN_ROWS,
        )

    appended = result.append
    print(
        f"wfa log issue {run_date}: appended={appended.n_appended} "
        f"late_rejected={appended.n_late_rejected} "
        f"retrained={'yes' if result.retrained else 'no'} "
        f"challenger_retrained_on={result.challenger_retrained_on} "
        f"history_rows={result.n_history_rows} "
        f"fit_seconds={result.fit_seconds:.2f} "
        f"wall_seconds={result.wall_seconds:.2f}"
    )
    for station, lead_day, variable, source in appended.late_keys:
        print(f"late: {station} lead={lead_day} {variable} {source}")
    if appended.n_late_rejected > 0:
        # The schedule ran after those windows closed: the strict late rule
        # refused them, so the run is not a clean one.
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wfa")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db_parser = subparsers.add_parser(
        "init-db", help="Create the raw schema and tables (idempotent)"
    )
    init_db_parser.set_defaults(func=_cmd_init_db)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Fetch, load, and resolve one station's guidance and obs"
    )
    ingest_parser.add_argument("--station", required=True, help="ICAO, e.g. KPHX")
    ingest_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    ingest_parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    ingest_parser.set_defaults(func=_cmd_ingest)

    predict_parser = subparsers.add_parser(
        "predict", help="Write model prediction rows via the walk-forward evaluator"
    )
    predict_subparsers = predict_parser.add_subparsers(
        dest="predict_model", required=True
    )
    baseline_parser = predict_subparsers.add_parser(
        "baseline", help="Rolling-bias baseline corrector (issue #18)"
    )
    baseline_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    baseline_parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    baseline_parser.add_argument(
        "--window-days", type=int, default=WINDOW_DAYS, dest="window_days"
    )
    baseline_parser.add_argument(
        "--min-pairs", type=int, default=MIN_PAIRS, dest="min_pairs"
    )
    baseline_parser.set_defaults(func=_cmd_predict_baseline)

    challenger_parser = predict_subparsers.add_parser(
        "challenger",
        help="LightGBM challenger corrector (issue #19); constants are fixed",
    )
    challenger_parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    challenger_parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    challenger_parser.set_defaults(func=_cmd_predict_challenger)

    export_parser = subparsers.add_parser(
        "export", help="Write the v1 export contract (manifest, city stats) to --out"
    )
    export_parser.add_argument("--out", required=True, help="Output directory")
    export_parser.set_defaults(func=_cmd_export)

    log_parser = subparsers.add_parser(
        "log", help="Live prediction log (append-only, sealed per run date)"
    )
    log_subparsers = log_parser.add_subparsers(dest="log_command", required=True)
    verify_parser = log_subparsers.add_parser(
        "verify",
        help="Recompute every seal and the hash chain; exit 1 on any mismatch",
    )
    verify_parser.set_defaults(func=_cmd_log_verify)
    issue_parser = log_subparsers.add_parser(
        "issue",
        help="Issue one run date's baseline and challenger predictions into the log",
    )
    issue_parser.add_argument("--run-date", required=True, dest="run_date")
    issue_parser.set_defaults(func=_cmd_log_issue)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
