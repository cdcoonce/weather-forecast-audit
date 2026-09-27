"""Pure completeness-summary computation (issue #11).

`scripts/completeness_summary.py` is the only caller that touches DuckDB;
everything here takes and returns polars frames, so it is unit-testable on a
hand-built fixture with exact expected rates (no database, no fixtures on
disk).

Every station is expected to have data for the *same* archive window --
`station_registry.csv` carries no per-station commissioning date -- so
`expected_days` for a (station, year) only depends on the calendar
(`archive_start`..`data_through`), never on the station itself.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def _expected_days_by_year(archive_start: date, data_through: date) -> pl.DataFrame:
    """One row per calendar year touched by `[archive_start, data_through]`
    (inclusive both ends), with the count of days in that year within the
    range."""
    if data_through < archive_start:
        msg = f"data_through ({data_through}) is before archive_start ({archive_start})"
        raise ValueError(msg)
    dates = pl.date_range(archive_start, data_through, interval="1d", eager=True)
    return (
        pl.DataFrame({"date": dates})
        .with_columns(pl.col("date").dt.year().alias("year"))
        .group_by("year")
        .agg(pl.len().alias("expected_days"))
        .sort("year")
    )


def completeness_by_station_year(
    gaps: pl.DataFrame,
    stations: Sequence[str],
    archive_start: date,
    data_through: date,
) -> pl.DataFrame:
    """Gap rate by `(station, year)`.

    `gaps` must have `station`, `gap_date` (a `pl.Date` column), and
    `reason` columns -- the dbt `gap_ledger` grain, minus `kind` and
    `first_seen` (this function doesn't care which source a gap came from).

    Returns `station`, `year`, `expected_days`, `gap_days` (the count of
    *distinct* calendar days in that station-year carrying at least one gap
    row, from any reason -- a day with two reasons on file still counts
    once, since this is a rate over days, not over gap records), and
    `gap_rate = gap_days / expected_days`. Every station in `stations`
    appears for every year in range, even with zero gaps.
    """
    expected = _expected_days_by_year(archive_start, data_through)
    scaffold = pl.DataFrame({"station": list(stations)}).join(expected, how="cross")

    gap_days = (
        gaps.with_columns(pl.col("gap_date").dt.year().alias("year"))
        .group_by(["station", "year"])
        .agg(pl.col("gap_date").n_unique().alias("gap_days"))
    )

    return (
        scaffold.join(gap_days, on=["station", "year"], how="left")
        .with_columns(pl.col("gap_days").fill_null(0))
        .with_columns((pl.col("gap_days") / pl.col("expected_days")).alias("gap_rate"))
        .select(["station", "year", "expected_days", "gap_days", "gap_rate"])
        .sort(["station", "year"])
    )


def gap_reason_counts(gaps: pl.DataFrame) -> pl.DataFrame:
    """Row counts by `(station, year, reason)`.

    Deliberately *not* deduplicated by day the way
    `completeness_by_station_year`'s `gap_days` is: a day with two reasons
    on file contributes to both reasons' counts here, since the point is
    which reason is most common, not how many distinct days are affected.
    """
    return (
        gaps.with_columns(pl.col("gap_date").dt.year().alias("year"))
        .group_by(["station", "year", "reason"])
        .agg(pl.len().alias("count"))
        .sort(["station", "year", "reason"])
    )


def stations_over_threshold(
    gaps: pl.DataFrame,
    stations: Sequence[str],
    archive_start: date,
    data_through: date,
    threshold: float = 0.10,
) -> pl.DataFrame:
    """Every station whose *overall* gap rate (summed across every year in
    `[archive_start, data_through]`) exceeds `threshold`, with its dominant
    reason -- the reason with the most gap rows overall (row counts, not
    distinct days, matching `gap_reason_counts`), ties broken
    alphabetically for a deterministic result.

    Returns `station`, `gap_rate`, `dominant_reason`, sorted by station.
    """
    by_year = completeness_by_station_year(gaps, stations, archive_start, data_through)
    overall = (
        by_year.group_by("station")
        .agg(pl.col("expected_days").sum(), pl.col("gap_days").sum())
        .with_columns((pl.col("gap_days") / pl.col("expected_days")).alias("gap_rate"))
    )

    reason_totals = gaps.group_by(["station", "reason"]).agg(pl.len().alias("count"))
    dominant = (
        reason_totals.sort(
            ["station", "count", "reason"], descending=[False, True, False]
        )
        .group_by("station", maintain_order=True)
        .agg(pl.col("reason").first().alias("dominant_reason"))
    )

    return (
        overall.join(dominant, on="station", how="left")
        .filter(pl.col("gap_rate") > threshold)
        .select(["station", "gap_rate", "dominant_reason"])
        .sort("station")
    )
