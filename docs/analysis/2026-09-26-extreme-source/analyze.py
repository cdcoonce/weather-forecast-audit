"""Issue #6 pre-registered measurement: choosing the observed-extreme source.

Runs the exact PREREG.md sample, metrics, and matching rules against the
production parsing/tiling code (`resolver.six_hour_extreme`,
`resolver.observed_extreme`) built in this phase. This script reports the
numbers the decision rule needs; it does not apply the rule or publish a
decision (PREREG.md and the build spec both require that).

Station construction: `fetch_hourly`/`fetch_cli` only read `Station.icao`
and `Station.cli_station`. Rather than widen those clients' signatures (a
change with no other caller and no test coverage), this script builds
`Station` instances directly for the 12 PREREG stations, with every other
field a clearly-marked, unused placeholder -- the seed CSV
(`dbt/seeds/station_registry.csv`) is not touched, since only KPHX is
committed there and this analysis is explicitly out of the fact table
(phase 1 does not wire a source into the pipeline).

`cli_station` is assumed equal to `icao` for all 12 stations, following the
only precedent in the repo (the KPHX seed row: `cli_station=icao=KPHX`).
This was not independently probed against the live `cli.py` endpoint for
the other 11 stations; if any of them use a different CLI product id, that
station's CLI comparison will show up as a `missing_report` gap for every
day, not a wrong-station mismatch (`fetch_cli` matches strictly by year and
date, never by name), so a bad guess here fails visibly.

Caching: every raw HTTP response is cached under `.cache/extreme-source/`,
keyed by a SHA-256 hash of the request URL. `--offline` refuses to make any
network call and requires every response to already be cached, so the
default polite live run and a reproducible resource-free rerun share one
code path (`CachingFetcher.get`).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from weather_forecast_audit.iem.http import (
    DEFAULT_USER_AGENT,
    FetchError,
    HttpResponse,
    UrllibFetcher,
)
from weather_forecast_audit.iem.observations import CliDaily, fetch_cli, fetch_hourly
from weather_forecast_audit.registry import REPO_ROOT, Station
from weather_forecast_audit.resolver import (
    SYNOPTIC_REPORT_LOOKBACK,
    Variable,
    observed_extreme,
    resolve_window,
    six_hour_extreme,
)

ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR / "results"
CACHE_DIR = REPO_ROOT / ".cache" / "extreme-source"

# PREREG sample: 12 stations, target dates 2025-01-01..2025-12-31.
STATION_ICAOS = [
    "KPHX", "KSFO", "KSEA", "KDEN", "KSLC", "KBIS",
    "KOKC", "KMSP", "KORD", "KATL", "KMIA", "KBOS",
]  # fmt: skip

TARGET_START = date(2025, 1, 1)
TARGET_END = date(2025, 12, 31)
# One extra day past the year so the last target date's max window
# (Dec 31 12Z .. Jan 1 06Z) and its 6-hour periods (through Jan 1 06Z) have
# their observations on hand.
OBS_START = date(2025, 1, 1)
OBS_END = date(2026, 1, 1)
CLI_YEAR_START = date(2025, 1, 1)
CLI_YEAR_END = date(2025, 12, 31)

OUTLIER_THRESHOLD_F = 10.0
SEASONS = ("DJF", "MAM", "JJA", "SON")
VARIABLES: tuple[Variable, ...] = ("max", "min")


# -- Station construction (placeholders documented in the module docstring) --


def _build_station(icao: str) -> Station:
    return Station(
        cli_station=icao,
        icao=icao,
        nbm_station_id="UNUSED",  # not read by fetch_hourly/fetch_cli
        iana_tz="UTC",  # unused placeholder
        lat=0.0,  # unused placeholder
        lon=0.0,  # unused placeholder
        elevation_m=0,  # unused placeholder
        label=icao,  # unused placeholder
        climate_region="unused",  # unused placeholder
        coastal_flag=False,  # unused placeholder
    )


# -- Caching fetcher -----------------------------------------------------


class CachingFetcher:
    """Caches every response by a hash of its URL under `CACHE_DIR`.

    Only successful responses are cached (a `FetchError` is never written to
    disk), matching the production clients' own retry/gap behavior: a
    rerun after a transient failure should retry, not replay the failure.
    """

    def __init__(
        self, inner: UrllibFetcher | None, cache_dir: Path, offline: bool
    ) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._offline = offline

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{key}.bin"

    def get(self, url: str) -> HttpResponse:
        path = self._cache_path(url)
        if path.exists():
            return HttpResponse(status=200, body=path.read_bytes())
        if self._offline:
            msg = f"--offline: no cached response for {url} (expected {path})"
            raise FetchError(status=None, reason=msg)
        if self._inner is None:
            msg = "CachingFetcher has no inner fetcher configured for a live request"
            raise FetchError(status=None, reason=msg)
        response = self._inner.get(url)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.body)
        return response


# -- Per-station data, sliced for fast per-window lookups --------------------


@dataclass(frozen=True)
class StationData:
    """One station's hourly obs and CLI daily reports, sorted by valid time."""

    valids: list[datetime]
    hourly_pairs: list[tuple[datetime, float | None]]
    six_hour_triples: list[tuple[datetime, float | None, float | None]]
    cli_by_date: dict[date, CliDaily]


def _fetch_station_data(
    station: Station, fetcher: CachingFetcher
) -> StationData | None:
    """Fetch and index one station's data, or None if ASOS returned nothing.

    A station with zero hourly rows over the whole period is reported and
    dropped, not replaced (PREREG sample rule).
    """
    hourly_result = fetch_hourly(station, OBS_START, OBS_END, fetcher)
    if not hourly_result.rows:
        return None

    sorted_rows = sorted(hourly_result.rows, key=lambda row: row.valid_utc)
    valids = [row.valid_utc for row in sorted_rows]
    hourly_pairs = [(row.valid_utc, row.tmpf) for row in sorted_rows]
    six_hour_triples = [
        (row.valid_utc, row.max_6h_f, row.min_6h_f) for row in sorted_rows
    ]

    cli_result = fetch_cli(station, CLI_YEAR_START, CLI_YEAR_END, fetcher)
    cli_by_date = {row.local_date: row for row in cli_result.rows}

    return StationData(
        valids=valids,
        hourly_pairs=hourly_pairs,
        six_hour_triples=six_hour_triples,
        cli_by_date=cli_by_date,
    )


def _slice_by_valid[T](
    valids: list[datetime], rows: list[T], lo_time: datetime, hi_time: datetime
) -> list[T]:
    """The contiguous slice of `rows` with `lo_time <= valid < hi_time`.

    `rows` and `valids` are parallel and sorted by valid time, so a window's
    relevant rows (at most a few dozen, out of a station's ~8760/year) can
    be found by bisection instead of scanning the whole year on every call.
    """
    lo = bisect.bisect_left(valids, lo_time)
    hi = bisect.bisect_left(valids, hi_time)
    return rows[lo:hi]


def _target_dates() -> list[date]:
    days = (TARGET_END - TARGET_START).days
    return [TARGET_START + timedelta(days=offset) for offset in range(days + 1)]


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


# -- One row per (station, target_date, variable) window ---------------------


@dataclass(frozen=True)
class WindowRow:
    station: str
    target_date: date
    variable: Variable
    season: str
    a_f: float | None  # hourly (observed_extreme)
    b_f: float | None  # metar_6h (six_hour_extreme)
    tiled: bool
    scorable: bool
    cli_f: int | None


def _window_rows_for_station(icao: str, data: StationData) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for target in _target_dates():
        for variable in VARIABLES:
            window = resolve_window(target, variable)
            lo_time = window.start_utc - SYNOPTIC_REPORT_LOOKBACK
            hi_time = window.end_utc
            hourly_slice = _slice_by_valid(
                data.valids, data.hourly_pairs, lo_time, hi_time
            )
            six_hour_slice = _slice_by_valid(
                data.valids, data.six_hour_triples, lo_time, hi_time
            )

            a = observed_extreme(window, variable, hourly_slice)
            b = six_hour_extreme(window, variable, six_hour_slice)

            cli_daily = data.cli_by_date.get(target)
            cli_f: int | None = None
            if cli_daily is not None:
                cli_f = cli_daily.high_f if variable == "max" else cli_daily.low_f

            rows.append(
                WindowRow(
                    station=icao,
                    target_date=target,
                    variable=variable,
                    season=_season(target.month),
                    a_f=a.value_f,
                    b_f=b.value_f,
                    tiled=b.tiled,
                    scorable=a.scorable,
                    cli_f=cli_f,
                )
            )
    return rows


# -- Aggregation -----------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _stats(
    values: list[float],
) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return (None, None, None, None)
    array = np.array(values, dtype=float)
    return (
        float(np.mean(array)),
        float(np.median(array)),
        float(np.percentile(array, 5)),
        float(np.percentile(array, 95)),
    )


def _within(diffs: list[float], tolerance: int) -> float | None:
    if not diffs:
        return None
    hits = sum(1 for diff in diffs if abs(round(diff)) <= tolerance)
    return hits / len(diffs)


@dataclass(frozen=True)
class AggregateRow:
    station: str
    variable: str
    season: str
    n_windows: int
    n_tiled: int
    tiling_rate: float | None
    n_hourly_error_pairs: int
    hourly_error_mean: float | None
    hourly_error_median: float | None
    hourly_error_p5: float | None
    hourly_error_p95: float | None
    n_cli_pairs_a: int
    mean_a_minus_cli: float | None
    pct_a_within0: float | None
    pct_a_within1: float | None
    n_cli_pairs_b: int
    mean_b_minus_cli: float | None
    pct_b_within0: float | None
    pct_b_within1: float | None


def _aggregate(
    station_label: str, variable_label: str, season_label: str, rows: list[WindowRow]
) -> AggregateRow:
    n_windows = len(rows)
    tiled_rows = [row for row in rows if row.tiled]
    n_tiled = len(tiled_rows)
    tiling_rate = (n_tiled / n_windows) if n_windows else None

    # Metric 2: hourly sampling error (a - b), on tiled windows with a scorable.
    hourly_errors = [
        row.a_f - row.b_f
        for row in tiled_rows
        if row.a_f is not None and row.b_f is not None
    ]
    error_mean, error_median, error_p5, error_p95 = _stats(hourly_errors)

    # Metric 3: agreement with CLI, on tiled windows, for x in {a, b}.
    a_minus_cli = [
        row.a_f - row.cli_f
        for row in tiled_rows
        if row.a_f is not None and row.cli_f is not None
    ]
    b_minus_cli = [
        row.b_f - row.cli_f
        for row in tiled_rows
        if row.b_f is not None and row.cli_f is not None
    ]

    return AggregateRow(
        station=station_label,
        variable=variable_label,
        season=season_label,
        n_windows=n_windows,
        n_tiled=n_tiled,
        tiling_rate=tiling_rate,
        n_hourly_error_pairs=len(hourly_errors),
        hourly_error_mean=error_mean,
        hourly_error_median=error_median,
        hourly_error_p5=error_p5,
        hourly_error_p95=error_p95,
        n_cli_pairs_a=len(a_minus_cli),
        mean_a_minus_cli=_mean(a_minus_cli),
        pct_a_within0=_within(a_minus_cli, 0),
        pct_a_within1=_within(a_minus_cli, 1),
        n_cli_pairs_b=len(b_minus_cli),
        mean_b_minus_cli=_mean(b_minus_cli),
        pct_b_within0=_within(b_minus_cli, 0),
        pct_b_within1=_within(b_minus_cli, 1),
    )


def _by_station_variable_season(all_rows: list[WindowRow]) -> list[AggregateRow]:
    out: list[AggregateRow] = []
    by_station: dict[str, list[WindowRow]] = {}
    for row in all_rows:
        by_station.setdefault(row.station, []).append(row)
    for station in sorted(by_station):
        station_rows = by_station[station]
        for variable in VARIABLES:
            for season in SEASONS:
                subset = [
                    row
                    for row in station_rows
                    if row.variable == variable and row.season == season
                ]
                out.append(_aggregate(station, variable, season, subset))
    return out


def _pooled(all_rows: list[WindowRow]) -> list[AggregateRow]:
    return [
        _aggregate(
            "ALL",
            variable,
            "ALL",
            [row for row in all_rows if row.variable == variable],
        )
        for variable in VARIABLES
    ]


def _by_station_pooled_season(all_rows: list[WindowRow]) -> list[AggregateRow]:
    """Per-station T pooled across season, used for the "min station T" check."""
    by_station: dict[str, list[WindowRow]] = {}
    for row in all_rows:
        by_station.setdefault(row.station, []).append(row)
    out: list[AggregateRow] = []
    for station in sorted(by_station):
        for variable in VARIABLES:
            subset = [row for row in by_station[station] if row.variable == variable]
            out.append(_aggregate(station, variable, "ALL", subset))
    return out


# -- Leave-one-station-out MAE of the hourly_corrected fallback --------------


def _season_corrections(rows: list[WindowRow]) -> dict[str, float]:
    """mean(b - a) per season, on tiled windows with a scorable."""
    by_season: dict[str, list[float]] = {}
    for row in rows:
        if row.tiled and row.a_f is not None and row.b_f is not None:
            by_season.setdefault(row.season, []).append(row.b_f - row.a_f)
    return {season: statistics.mean(values) for season, values in by_season.items()}


def _loso_mae(
    all_rows: list[WindowRow], variable: Variable
) -> tuple[float | None, int]:
    """Leave-one-station-out MAE of hourly_corrected = a + mean(b-a) vs b.

    For each held-out station, the per-season correction is estimated from
    every *other* station's tiled windows only, then applied to the
    held-out station's own tiled windows; the errors from all 12 folds are
    pooled into one MAE. A held-out window whose season has no correction
    from the training stations (should not occur at this sample size, but
    is possible for a small subset) is excluded rather than guessed.
    """
    by_station: dict[str, list[WindowRow]] = {}
    for row in all_rows:
        if row.variable == variable:
            by_station.setdefault(row.station, []).append(row)

    errors: list[float] = []
    for held_out in by_station:
        training_rows = [
            row
            for station, rows in by_station.items()
            if station != held_out
            for row in rows
        ]
        corrections = _season_corrections(training_rows)
        for row in by_station[held_out]:
            if not (row.tiled and row.a_f is not None and row.b_f is not None):
                continue
            correction = corrections.get(row.season)
            if correction is None:
                continue
            corrected = row.a_f + correction
            errors.append(abs(corrected - row.b_f))

    return (statistics.mean(errors) if errors else None, len(errors))


# -- Outliers ----------------------------------------------------------------


@dataclass(frozen=True)
class OutlierRow:
    station: str
    target_date: date
    variable: str
    season: str
    a_f: float
    b_f: float
    diff_a_minus_b: float


def _outliers(all_rows: list[WindowRow]) -> list[OutlierRow]:
    outliers: list[OutlierRow] = []
    for row in all_rows:
        if row.a_f is None or row.b_f is None:
            continue
        diff = row.a_f - row.b_f
        if abs(diff) > OUTLIER_THRESHOLD_F:
            outliers.append(
                OutlierRow(
                    station=row.station,
                    target_date=row.target_date,
                    variable=row.variable,
                    season=row.season,
                    a_f=row.a_f,
                    b_f=row.b_f,
                    diff_a_minus_b=diff,
                )
            )
    return outliers


# -- CSV / markdown output ---------------------------------------------------


def _write_aggregate_csv(path: Path, rows: list[AggregateRow]) -> None:
    fieldnames = list(AggregateRow.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(vars(row))


def _write_outliers_csv(path: Path, rows: list[OutlierRow]) -> None:
    fieldnames = list(OutlierRow.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(vars(row))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _write_results_md(
    path: Path,
    pooled_rows: list[AggregateRow],
    by_station_pooled_season: list[AggregateRow],
    loso: dict[Variable, tuple[float | None, int]],
    outliers: list[OutlierRow],
    no_data_stations: list[str],
) -> None:
    pooled_by_variable = {row.variable: row for row in pooled_rows}

    lines: list[str] = []
    lines.append("# Results: observed-extreme source measurement (issue #6)")
    lines.append("")
    lines.append(
        "Generated by `analyze.py` against PREREG.md's fixed sample, metrics, "
        "and matching rules. This file reports numbers only; it does not "
        "apply PREREG's decision rule."
    )
    lines.append("")
    lines.append(f"Stations requested: {', '.join(STATION_ICAOS)} (12).")
    if no_data_stations:
        lines.append(
            f"Stations that returned no ASOS data (reported and dropped, not "
            f"replaced): {', '.join(no_data_stations)}."
        )
    else:
        lines.append("Every requested station returned ASOS data.")
    lines.append(
        f"Target dates: {TARGET_START.isoformat()} .. {TARGET_END.isoformat()}."
    )
    lines.append("")

    lines.append("## Decision-rule inputs")
    lines.append("")
    for variable in VARIABLES:
        agg = pooled_by_variable[variable]
        lines.append(f"### {variable}")
        lines.append("")
        lines.append(
            f"- Pooled T: {_fmt(agg.tiling_rate)} ({agg.n_tiled}/{agg.n_windows})"
        )
        error_mean = _fmt(agg.hourly_error_mean)
        lines.append(
            f"- Pooled mean(a - b) on tiled windows: {error_mean} F "
            f"({agg.n_hourly_error_pairs} pairs)"
        )
        lines.append(
            f"- Pooled mean(a - CLI): {_fmt(agg.mean_a_minus_cli)} F "
            f"({agg.n_cli_pairs_a} pairs)"
        )
        lines.append(
            f"- Pooled mean(b - CLI): {_fmt(agg.mean_b_minus_cli)} F "
            f"({agg.n_cli_pairs_b} pairs)"
        )
        a_abs = None if agg.mean_a_minus_cli is None else abs(agg.mean_a_minus_cli)
        b_abs = None if agg.mean_b_minus_cli is None else abs(agg.mean_b_minus_cli)
        lines.append(f"- |mean(a - CLI)|: {_fmt(a_abs)} F")
        lines.append(f"- |mean(b - CLI)|: {_fmt(b_abs)} F")
        mae, n_mae = loso[variable]
        lines.append(
            f"- Leave-one-station-out MAE of hourly_corrected vs b: {_fmt(mae)} F "
            f"({n_mae} pairs)"
        )
        lines.append("")

    lines.append("### T by station (pooled across season)")
    lines.append("")
    lines.append("| station | variable | T | n_tiled/n_windows |")
    lines.append("| --- | --- | --- | --- |")
    min_t: dict[Variable, tuple[str, float]] = {}
    for row in by_station_pooled_season:
        lines.append(
            f"| {row.station} | {row.variable} | {_fmt(row.tiling_rate)} | "
            f"{row.n_tiled}/{row.n_windows} |"
        )
        if row.tiling_rate is not None:
            current = min_t.get(row.variable)
            if current is None or row.tiling_rate < current[1]:
                min_t[row.variable] = (row.station, row.tiling_rate)
    lines.append("")
    for variable in VARIABLES:
        if variable in min_t:
            station, value = min_t[variable]
            lines.append(f"- Min station T for {variable}: {_fmt(value)} at {station}")
    lines.append("")

    lines.append("## Pooled metrics by variable")
    lines.append("")
    lines.append(
        "| variable | n_windows | T | hourly_error_mean | hourly_error_median | "
        "hourly_error_p5 | hourly_error_p95 | mean(a-CLI) | pct_a_within1 | "
        "mean(b-CLI) | pct_b_within1 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in pooled_rows:
        lines.append(
            f"| {row.variable} | {row.n_windows} | {_fmt(row.tiling_rate)} | "
            f"{_fmt(row.hourly_error_mean)} | {_fmt(row.hourly_error_median)} | "
            f"{_fmt(row.hourly_error_p5)} | {_fmt(row.hourly_error_p95)} | "
            f"{_fmt(row.mean_a_minus_cli)} | {_fmt(row.pct_a_within1)} | "
            f"{_fmt(row.mean_b_minus_cli)} | {_fmt(row.pct_b_within1)} |"
        )
    lines.append("")
    lines.append(
        "Full station x variable x season breakdown: "
        "`results/by_station_variable_season.csv`."
    )
    lines.append("")

    lines.append("## Outliers")
    lines.append("")
    lines.append(f"{len(outliers)} windows with |a - b| > {OUTLIER_THRESHOLD_F:.0f} F.")
    lines.append("Full list: `results/outliers.csv`.")
    if outliers:
        lines.append("")
        lines.append("| station | target_date | variable | a_f | b_f | diff (a-b) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in outliers:
            lines.append(
                f"| {row.station} | {row.target_date.isoformat()} | {row.variable} | "
                f"{row.a_f:.1f} | {row.b_f:.1f} | {row.diff_a_minus_b:.1f} |"
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# -- main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refuse any network call; every response must already be cached.",
    )
    args = parser.parse_args()

    inner = None if args.offline else UrllibFetcher(user_agent=DEFAULT_USER_AGENT)
    fetcher = CachingFetcher(inner=inner, cache_dir=CACHE_DIR, offline=args.offline)

    all_rows: list[WindowRow] = []
    no_data_stations: list[str] = []
    for icao in STATION_ICAOS:
        station = _build_station(icao)
        print(f"fetching {icao} ...")
        data = _fetch_station_data(station, fetcher)
        if data is None:
            print(f"  {icao}: no ASOS data returned, dropping")
            no_data_stations.append(icao)
            continue
        rows = _window_rows_for_station(icao, data)
        print(f"  {icao}: {len(rows)} windows")
        all_rows.extend(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    by_svs = _by_station_variable_season(all_rows)
    _write_aggregate_csv(RESULTS_DIR / "by_station_variable_season.csv", by_svs)

    pooled_rows = _pooled(all_rows)
    _write_aggregate_csv(RESULTS_DIR / "pooled.csv", pooled_rows)

    outliers = _outliers(all_rows)
    _write_outliers_csv(RESULTS_DIR / "outliers.csv", outliers)

    by_station_pooled_season = _by_station_pooled_season(all_rows)

    loso: dict[Variable, tuple[float | None, int]] = {
        variable: _loso_mae(all_rows, variable) for variable in VARIABLES
    }

    _write_results_md(
        RESULTS_DIR.parent / "RESULTS.md",
        pooled_rows,
        by_station_pooled_season,
        loso,
        outliers,
        no_data_stations,
    )

    print(f"wrote results to {RESULTS_DIR} and {RESULTS_DIR.parent / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
