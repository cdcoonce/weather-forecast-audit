#!/usr/bin/env python3
"""Completeness summary (issue #11): gap rate by station x year, and which
stations exceed a gap-rate threshold, from the gap ledger and the resolved
verification windows already on file.

Reads two things directly, read-only, from `$WFA_DUCKDB_PATH`:

- `gap_ledger` (`dbt/models/marts/gap_ledger.sql`, built from
  `raw.ingest_gaps` via `warehouse.load_gaps` -- see `gaps.py` for the
  `Reason` values): `station`, `gap_date`, `reason`.
- `raw.resolved_windows` (`warehouse.py`): its max `target_date` sets
  `data_through`, the last date this backfill has actually reached. This is
  a raw table Python ingestion writes directly, so it doesn't depend on
  whether dbt has been built since the latest ingest.

All the actual computation is in `weather_forecast_audit.completeness`
(pure polars functions, unit-tested on a hand-built fixture); this script is
just the DuckDB read and the markdown/CSV write.

Usage:
    WFA_DUCKDB_PATH=/path/to.duckdb uv run python scripts/completeness_summary.py \\
        --out docs/analysis/completeness
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from weather_forecast_audit.completeness import (  # noqa: E402
    completeness_by_station_year,
    gap_reason_counts,
    stations_over_threshold,
)
from weather_forecast_audit.regimes import load_archive_start  # noqa: E402
from weather_forecast_audit.registry import load_registry  # noqa: E402

DEFAULT_THRESHOLD = 0.10


def _db_path() -> str:
    path = os.environ.get("WFA_DUCKDB_PATH")
    if not path:
        msg = "WFA_DUCKDB_PATH must be set to the DuckDB database file path"
        raise SystemExit(msg)
    return path


def _load_gaps(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return conn.execute("select station, gap_date, reason from gap_ledger").pl()


def _data_through(conn: duckdb.DuckDBPyConnection) -> date:
    (max_date,) = conn.execute(
        "select max(target_date) from raw.resolved_windows"
    ).fetchone()
    if max_date is None:
        msg = "raw.resolved_windows is empty -- nothing has been ingested yet"
        raise SystemExit(msg)
    return max_date


def _with_reason_columns(by_year: pl.DataFrame, gaps: pl.DataFrame) -> pl.DataFrame:
    """`by_year` (station, year, expected_days, gap_days, gap_rate) with one
    extra column per reason seen in `gaps`, counting rows (not distinct
    days -- matching `gap_reason_counts`)."""
    reasons = gap_reason_counts(gaps)
    if reasons.height == 0:
        return by_year
    wide = reasons.pivot(values="count", index=["station", "year"], on="reason")
    reason_columns = [c for c in wide.columns if c not in ("station", "year")]
    return (
        by_year.join(wide, on=["station", "year"], how="left")
        .with_columns([pl.col(c).fill_null(0) for c in reason_columns])
        .sort(["station", "year"])
    )


def _write_markdown(
    path: Path, by_year: pl.DataFrame, over_threshold: pl.DataFrame, threshold: float
) -> None:
    reason_columns = [
        c
        for c in by_year.columns
        if c not in ("station", "year", "expected_days", "gap_days", "gap_rate")
    ]
    header = [
        "station",
        "year",
        "expected_days",
        "gap_days",
        "gap_rate",
        *reason_columns,
    ]
    lines = [
        "# Completeness summary",
        "",
        "## Gap rate by station and year",
        "",
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for row in by_year.to_dicts():
        cells = [
            str(row["station"]),
            str(row["year"]),
            str(row["expected_days"]),
            str(row["gap_days"]),
            f"{row['gap_rate']:.3f}",
            *[str(row.get(c, 0)) for c in reason_columns],
        ]
        lines.append(f"| {' | '.join(cells)} |")

    lines += [
        "",
        f"## Stations over the {threshold:.0%} gap-rate threshold",
        "",
        "| station | gap_rate | dominant_reason |",
        "| --- | --- | --- |",
    ]
    for row in over_threshold.to_dicts():
        lines.append(
            f"| {row['station']} | {row['gap_rate']:.3f} | {row['dominant_reason']} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, help="Output directory for the .md and .csv"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(_db_path(), read_only=True) as conn:
        gaps = _load_gaps(conn)
        data_through = _data_through(conn)

    stations = sorted(load_registry())
    archive_start = load_archive_start()

    by_year = completeness_by_station_year(gaps, stations, archive_start, data_through)
    by_year_with_reasons = _with_reason_columns(by_year, gaps)
    over_threshold = stations_over_threshold(
        gaps, stations, archive_start, data_through, threshold=args.threshold
    )

    by_year_with_reasons.write_csv(out_dir / "completeness_by_station_year.csv")
    _write_markdown(
        out_dir / "completeness_summary.md",
        by_year_with_reasons,
        over_threshold,
        args.threshold,
    )

    print(
        f"completeness_summary: {len(stations)} stations, data_through={data_through}, "
        f"{over_threshold.height} station(s) over {args.threshold:.0%} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
