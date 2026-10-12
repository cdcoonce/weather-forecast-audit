#!/usr/bin/env python3
"""One-time repair of source duplicates already loaded into the raw tables.

The loaders now collapse each key's rows to one before they insert, but a
database loaded earlier still holds the duplicates. This applies the same
per-key merge rule to `raw.asos_hourly`, `raw.nbs_guidance` and
`raw.resolved_windows` in place (a valued row keeps its value next to a
valueless one; a real non-null disagreement stops the run and names the
key).

Order of operations:

1. Take a file copy of the DuckDB database first.
2. Run this script with no flags: a dry run, opened read-only, that writes
   nothing and prints each table's predicted before/after row counts.
3. Run it again with `--apply`. Each table is repaired in one transaction
   that verifies the row count and that no key is still duplicated, and
   rolls back if either check fails. A second `--apply` is a no-op.

Needs the deployed package (`weather_forecast_audit`), so run it inside the
container, not as a bare script. Not part of the installed package.

    scripts/dedupe_raw_once.py [--db PATH] [--apply] [--table raw.asos_hourly ...]

Exit status: 0 done, 1 a conflict or database error (nothing written for
that table or any later one), 2 bad usage or a missing database file.
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb

from weather_forecast_audit.warehouse import RAW_KEY_COLUMNS, dedupe_raw_table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dedupe_raw_once.py",
        description="Collapse already-loaded duplicate keys in the raw tables.",
    )
    parser.add_argument(
        "--db",
        help="DuckDB file (default: $WFA_DUCKDB_PATH)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the repair (default: a read-only dry run)",
    )
    parser.add_argument(
        "--table",
        action="append",
        choices=sorted(RAW_KEY_COLUMNS),
        help="a table to repair; repeatable (default: all three)",
    )
    return parser


def main(argv: list[str]) -> int:
    args = _parser().parse_args(argv[1:])

    db_path = args.db or os.environ.get("WFA_DUCKDB_PATH")
    if not db_path:
        print("dedupe_raw_once.py: pass --db or set WFA_DUCKDB_PATH", file=sys.stderr)
        return 2
    if not Path(db_path).is_file():
        print(
            f"dedupe_raw_once.py: database {db_path} does not exist",
            file=sys.stderr,
        )
        return 2

    tables = list(dict.fromkeys(args.table or RAW_KEY_COLUMNS))
    applied_tables: list[str] = []
    with duckdb.connect(db_path, read_only=not args.apply) as conn:
        for table in tables:
            try:
                report = dedupe_raw_table(conn, table, apply=args.apply)
            except (ValueError, RuntimeError, duckdb.Error) as exc:
                done = (
                    f"; already applied (kept): {', '.join(applied_tables)}"
                    if applied_tables
                    else ""
                )
                print(
                    f"dedupe_raw_once.py: {exc}\n"
                    f"nothing was written for {table} or any later table{done}",
                    file=sys.stderr,
                )
                return 1
            mode = "APPLIED" if report.applied else "DRY RUN"
            print(
                f"{report.table}  {report.rows_before} -> {report.rows_after}  "
                f"(keys_collapsed={report.keys_collapsed})  [{mode}]"
            )
            if report.applied:
                applied_tables.append(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
