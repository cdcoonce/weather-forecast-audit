"""`wfa` command-line interface: `init-db` and `ingest`.

The DuckDB path always comes from `WFA_DUCKDB_PATH`, never a flag: one
database per environment (local dev, CI, the tracer script), set once.
"""

import argparse
import os
from datetime import date
from pathlib import Path

import duckdb

from weather_forecast_audit import export as export_module
from weather_forecast_audit import warehouse
from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry


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


def _cmd_export(args: argparse.Namespace) -> None:
    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        summary = export_module.export(conn, Path(args.out))
    print(
        f"wfa export: {summary.station_count} stations, "
        f"data_through={summary.data_through}, {len(summary.files)} files "
        f"-> {summary.out_dir}"
    )


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

    export_parser = subparsers.add_parser(
        "export", help="Write the v1 export contract (manifest, city stats) to --out"
    )
    export_parser.add_argument("--out", required=True, help="Output directory")
    export_parser.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
