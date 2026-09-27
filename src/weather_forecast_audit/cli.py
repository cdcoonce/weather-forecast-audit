"""`wfa` command-line interface: `init-db`, `ingest`, and `predict baseline`.

The DuckDB path always comes from `WFA_DUCKDB_PATH`, never a flag: one
database per environment (local dev, CI, the tracer script), set once.
"""

import argparse
import json
import os
import time
from datetime import UTC, date, datetime

import duckdb
import polars as pl

from weather_forecast_audit import warehouse
from weather_forecast_audit.baseline import MIN_PAIRS, WINDOW_DAYS, BaselineModel
from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
