#!/usr/bin/env python3
"""Run a .sql file against $WFA_DUCKDB_PATH and print the result as a table.

Used by scripts/tracer_kphx.sh; not part of the installed package.
"""

import os
import sys
from pathlib import Path

import duckdb


def _render(columns: list[str], rows: list[tuple[object, ...]]) -> str:
    widths = [
        max(len(str(value)) for value in [column, *(row[i] for row in rows)])
        for i, column in enumerate(columns)
    ]

    def line(values: list[object]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths, strict=True))

    lines = [line(columns), line(["-" * w for w in widths])]
    lines.extend(line(list(row)) for row in rows)
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: run_sql.py <path-to-sql-file>", file=sys.stderr)
        return 2

    db_path = os.environ.get("WFA_DUCKDB_PATH")
    if not db_path:
        print("run_sql.py: WFA_DUCKDB_PATH must be set", file=sys.stderr)
        return 2

    query = Path(argv[1]).read_text()
    with duckdb.connect(db_path, read_only=True) as conn:
        result = conn.execute(query)
        columns = [description[0] for description in result.description]
        rows = result.fetchall()

    print(_render(columns, rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
