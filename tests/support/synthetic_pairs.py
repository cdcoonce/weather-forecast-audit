"""Synthetic `int_raw_verification_pairs`-shaped rows, naive UTC like raw.*.

Shared by the issuance unit tests and the Dagster asset test. Each run is
issued at 13:00 UTC; a max window closes the day after its target at 06:00
and a min window on its target day at 18:00, so every lead-1..3 window of a
run closes after that run's own issuance (the live late rule's premise).
"""

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta

import numpy as np
import polars as pl

LEADS_BY_VARIABLE = {"max": (1, 2), "min": (1, 2, 3)}
ISSUE_HOUR = 13

ActiveFn = Callable[[str, date], bool]
RowFn = Callable[[str, date, str, int], bool]


def runtime_of(run_date: date) -> datetime:
    return datetime.combine(run_date, time(ISSUE_HOUR, 0))


def window_of(variable: str, target: date) -> tuple[datetime, datetime]:
    if variable == "max":
        return (
            datetime.combine(target, time(12, 0)),
            datetime.combine(target + timedelta(days=1), time(6, 0)),
        )
    return (
        datetime.combine(target, time(0, 0)),
        datetime.combine(target, time(18, 0)),
    )


def _always(station: str, run_date: date) -> bool:
    return True


def _never(station: str, run_date: date, variable: str, lead: int) -> bool:
    return False


def make_pairs(
    run_dates: Sequence[date],
    stations: Sequence[str],
    *,
    seed: int = 11,
    active: ActiveFn = _always,
    null_forecast: RowFn = _never,
    unscorable: RowFn = _never,
    closes_at_next_issue: RowFn = _never,
) -> pl.DataFrame:
    """Pairs for `run_dates` x `stations`; the NBS error depends on `tmp_f`.

    * `active(station, run_date)` False: the station has no rows that run
      (a silent group).
    * `null_forecast(...)` True: `forecast_f` is null while the observation
      exists and the pair is scorable (a pair with no guidance value).
    * `unscorable(...)` True: no observation and `scorable` false.
    * `closes_at_next_issue(...)` True: the window closes exactly at the next
      day's issuance, the instant the strict `< T` cut excludes it from that
      run's training and the next delta must therefore include it.
    """
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    observed_cache: dict[tuple[str, date, str], float] = {}
    for run_date in run_dates:
        runtime = runtime_of(run_date)
        for station_index, station in enumerate(stations):
            if not active(station, run_date):
                continue
            for variable, leads in LEADS_BY_VARIABLE.items():
                for lead in leads:
                    target = run_date + timedelta(days=lead)
                    start, end = window_of(variable, target)
                    if closes_at_next_issue(station, run_date, variable, lead):
                        end = runtime_of(run_date + timedelta(days=1))
                    key = (station, target, variable)
                    if key not in observed_cache:
                        observed_cache[key] = float(rng.normal(60.0, 12.0))
                    observed = observed_cache[key]
                    tmp = float(rng.uniform(30.0, 90.0))
                    error = (
                        (4.0 if tmp > 62.0 else 0.0)
                        + 0.3 * station_index
                        + float(rng.normal(0.0, 0.7))
                    )
                    no_obs = unscorable(station, run_date, variable, lead)
                    no_fc = null_forecast(station, run_date, variable, lead)
                    records.append(
                        {
                            "station": station,
                            "run_date": run_date,
                            "runtime_utc": runtime,
                            "cycle_hour": ISSUE_HOUR,
                            "lead_day": lead,
                            "variable": variable,
                            "target_date": target,
                            "forecast_f": None if no_fc else observed + error,
                            "spread_f": 3.0 + float(rng.uniform(0.0, 1.0)),
                            "tmp_f": tmp,
                            "window_start_utc": start,
                            "window_end_utc": end,
                            "observed_f": None if no_obs else observed,
                            "scorable": not no_obs,
                        }
                    )
    return pl.DataFrame(
        records,
        schema={
            "station": pl.String,
            "run_date": pl.Date,
            "runtime_utc": pl.Datetime("us"),
            "cycle_hour": pl.Int32,
            "lead_day": pl.Int32,
            "variable": pl.String,
            "target_date": pl.Date,
            "forecast_f": pl.Float64,
            "spread_f": pl.Float64,
            "tmp_f": pl.Float64,
            "window_start_utc": pl.Datetime("us"),
            "window_end_utc": pl.Datetime("us"),
            "observed_f": pl.Float64,
            "scorable": pl.Boolean,
        },
    )


def station_features(stations: Sequence[str]) -> pl.DataFrame:
    regions = ("Southeast", "Northeast", "West")
    return pl.DataFrame(
        {
            "station": list(stations),
            "latitude": [30.0 + 2.0 * i for i in range(len(stations))],
            "longitude": [-100.0 + 3.0 * i for i in range(len(stations))],
            "elevation_m": [100 * i for i in range(len(stations))],
            "coastal_flag": [i % 2 == 0 for i in range(len(stations))],
            "climate_region": [regions[i % 3] for i in range(len(stations))],
        }
    )
