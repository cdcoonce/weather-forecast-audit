"""Pure resolution of NBM TXN verification windows and observed extremes.

No I/O and no time-zone input: every datetime here is UTC and every window
boundary is a fixed UTC clock hour. That is a deliberate reading of the MDL
NBM text card (build spec fact 1), which defines TXN in UTC:

    TXN = 18-hour maximum and minimum temperatures, degrees F. Min is
    between 00Z-18Z and reported at 12Z ... Max is between 12z(current
    day)-06Z(next day) and reported at 00z(following day).

That is different from the GFS MOS (MAV) 0700-1900 LST convention (MDL TPB
05-03), which does not apply to NBM/NBS guidance and is not used here.

Boundary convention: windows are half-open ``[start, end)``. The MDL text
does not say whether the boundary observations count; the spec's known
answers only distinguish behavior when the window end is not on an observed
minute, so `[start, end)` is a choice, documented in docs/methodology.md,
not a re-derivation of an established fact.

Observed-extreme source (issue #6): `resolve_observed` is the single entry
point callers should use. Per PREREG.md/RESULTS.md, METAR 6-hour max/min
remark groups (`six_hour_extreme`) are the primary source -- a window is
`scorable` exactly when its three synoptic periods tile, with
`extreme_source = 'metar_6h'`. An untiled window is unscorable, with
`extreme_source = 'none'` and `value_f = None`; there is no hourly fallback
and no correction. `observed_extreme`'s hourly max/min is kept only as
`hourly_value_f`, a diagnostic gated by its own `MIN_HOUR_COVERAGE`
threshold, independent of `scorable`. `observed_extreme` and
`six_hour_extreme` remain the pure building blocks `resolve_observed`
composes.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import ceil
from typing import Literal

MIN_HOUR_COVERAGE = 0.75
WINDOW_HOURS = 18

# PREREG matching rule: "US synoptic reports are issued at about H-9 min", so a
# 60-minute lookback before H comfortably covers the issuance report without
# reaching back far enough to catch the *previous* synoptic report.
SYNOPTIC_REPORT_LOOKBACK = timedelta(minutes=60)

Variable = Literal["max", "min"]
ExtremeSource = Literal["metar_6h", "none"]


@dataclass(frozen=True)
class Window:
    """A half-open UTC verification window: ``[start_utc, end_utc)``."""

    start_utc: datetime
    end_utc: datetime


@dataclass(frozen=True)
class Extreme:
    """An observed max/min over a window, with the coverage that backs it."""

    value_f: float | None
    n_obs: int
    hours_covered: int
    hours_expected: int
    scorable: bool


@dataclass(frozen=True)
class SixHourExtreme:
    """The METAR 6-hour-group max/min over a window's three synoptic periods.

    `tiled` is True only when all three periods (see `six_hour_periods`)
    found a qualifying report; `value_f` is None whenever `tiled` is False,
    per PREREG's matching rules (mixing tiled and untiled periods would
    reintroduce exactly the sampling bias the source is meant to avoid).
    """

    value_f: float | None
    periods_found: int
    tiled: bool


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None:
        msg = f"{name} must be a timezone-aware UTC datetime, got a naive one"
        raise ValueError(msg)


def classify_txn(ftime: datetime) -> tuple[Variable, date]:
    """Classify a guidance row's ftime as a max or min TXN and its target date.

    max is reported at ftime hour 00Z for the previous UTC day (the window
    that ended at that 00Z). min is reported at ftime hour 12Z for that same
    UTC day. Any other ftime hour never occurs in canonical NBS runs and is
    treated as a parse error (build spec fact 4).
    """
    _require_aware(ftime, "ftime")
    if ftime.hour == 0:
        return "max", (ftime - timedelta(days=1)).date()
    if ftime.hour == 12:
        return "min", ftime.date()
    msg = f"txn ftime hour must be 0 or 12, got {ftime.hour} ({ftime.isoformat()})"
    raise ValueError(msg)


def lead_day(runtime: datetime, target_date: date) -> int:
    """Days between the run's UTC date and the target UTC date."""
    _require_aware(runtime, "runtime")
    return (target_date - runtime.date()).days


def resolve_window(target_date: date, variable: Variable) -> Window:
    """The fixed 18-hour UTC verification window for a target date/variable."""
    if variable == "max":
        start = datetime.combine(target_date, time(12, 0), tzinfo=UTC)
    elif variable == "min":
        start = datetime.combine(target_date, time(0, 0), tzinfo=UTC)
    else:
        msg = f"variable must be 'max' or 'min', got {variable!r}"
        raise ValueError(msg)
    return Window(start_utc=start, end_utc=start + timedelta(hours=WINDOW_HOURS))


def observed_extreme(
    window: Window,
    variable: Variable,
    observations: Iterable[tuple[datetime, float | None]],
    min_hour_coverage: float = MIN_HOUR_COVERAGE,
) -> Extreme:
    """The observed max/min in `window`, gated by hourly-coverage completeness.

    hours_covered counts distinct UTC clock hours (the floor of each ob's
    valid time) holding at least one non-missing tmpf within the half-open
    window. A window is scorable when hours_covered meets
    ceil(min_hour_coverage * WINDOW_HOURS); see docs/methodology.md for the
    rationale (build spec: tolerates short outages while keeping the daily
    peak/trough hours dense enough).
    """
    values: list[float] = []
    covered_hours: set[datetime] = set()
    n_obs = 0
    for valid, tmpf in observations:
        _require_aware(valid, "observation valid time")
        if not (window.start_utc <= valid < window.end_utc):
            continue
        if tmpf is None:
            continue
        n_obs += 1
        values.append(tmpf)
        covered_hours.add(valid.replace(minute=0, second=0, microsecond=0))

    hours_covered = len(covered_hours)
    threshold = ceil(min_hour_coverage * WINDOW_HOURS)
    scorable = hours_covered >= threshold

    value_f: float | None = None
    if scorable and values:
        value_f = max(values) if variable == "max" else min(values)

    return Extreme(
        value_f=value_f,
        n_obs=n_obs,
        hours_covered=hours_covered,
        hours_expected=WINDOW_HOURS,
        scorable=scorable,
    )


def six_hour_periods(window: Window) -> tuple[datetime, datetime, datetime]:
    """The three synoptic end-hours (H) that tile `window`, per PREREG.

    A max window starts at `D 12Z` and is tiled by the periods ending
    `18Z D`, `00Z D+1`, `06Z D+1`. A min window starts at `D 00Z` and is
    tiled by the periods ending `06Z`, `12Z`, `18Z` of `D`. Both reduce to
    the same offsets from `window.start_utc` (+6h, +12h, +18h); the start
    hour is still validated (0 or 12, matching `resolve_window`'s only two
    outputs) so a caller passing an unexpected window fails loudly rather
    than silently tiling the wrong hours.
    """
    _require_aware(window.start_utc, "window.start_utc")
    if window.start_utc.hour not in (0, 12):
        msg = (
            "window.start_utc hour must be 0 (min window) or 12 (max window), "
            f"got {window.start_utc.hour} ({window.start_utc.isoformat()})"
        )
        raise ValueError(msg)
    start = window.start_utc
    return (
        start + timedelta(hours=6),
        start + timedelta(hours=12),
        start + timedelta(hours=18),
    )


def six_hour_extreme(
    window: Window,
    variable: Variable,
    reports: Iterable[tuple[datetime, float | None, float | None]],
) -> SixHourExtreme:
    """The METAR 6-hour-group max/min over `window`'s three synoptic periods.

    Each report is `(valid_utc, max_6h_f, min_6h_f)`. For every synoptic
    end-hour `H` from `six_hour_periods`, the group value comes from the
    latest report valid in `[H - SYNOPTIC_REPORT_LOOKBACK, H)` that carries
    the group needed for `variable` (a report with that group `None` does
    not qualify for that period). `tiled` is True only when all three
    periods found a qualifying report, per PREREG's matching rules;
    `value_f` is the max/min across the three found values when tiled, else
    `None`.
    """
    periods = six_hour_periods(window)
    report_rows = list(reports)
    found_values: list[float] = []
    for hour in periods:
        lower = hour - SYNOPTIC_REPORT_LOOKBACK
        latest_valid: datetime | None = None
        latest_value: float | None = None
        for valid, max_6h_f, min_6h_f in report_rows:
            _require_aware(valid, "report valid time")
            if not (lower <= valid < hour):
                continue
            candidate = max_6h_f if variable == "max" else min_6h_f
            if candidate is None:
                continue
            if latest_valid is None or valid > latest_valid:
                latest_valid = valid
                latest_value = candidate
        if latest_value is not None:
            found_values.append(latest_value)

    periods_found = len(found_values)
    tiled = periods_found == len(periods)
    value_f: float | None = None
    if tiled:
        value_f = max(found_values) if variable == "max" else min(found_values)

    return SixHourExtreme(value_f=value_f, periods_found=periods_found, tiled=tiled)


@dataclass(frozen=True)
class Observed:
    """The resolved observed extreme for a window, per issue #6's decision.

    `value_f`/`extreme_source`/`scorable` are the authoritative result:
    `metar_6h` and tiled, or `none` and unscorable with `value_f = None`.
    `hourly_value_f` is a diagnostic only -- the legacy hourly max/min,
    gated by its own `MIN_HOUR_COVERAGE` threshold -- and does not affect
    `scorable`. `n_obs`/`hours_covered`/`hours_expected` describe the hourly
    coverage backing that diagnostic.
    """

    value_f: float | None
    extreme_source: ExtremeSource
    scorable: bool
    periods_found: int
    hourly_value_f: float | None
    n_obs: int
    hours_covered: int
    hours_expected: int


def resolve_observed(
    window: Window,
    variable: Variable,
    reports: Iterable[tuple[datetime, float | None, float | None, float | None]],
) -> Observed:
    """Resolve `window`'s observed extreme per issue #6's decision.

    Each report is `(valid_utc, tmpf, max_6h_f, min_6h_f)`. The METAR 6-hour
    groups (`six_hour_extreme`) decide `value_f`/`extreme_source`/`scorable`;
    `observed_extreme`'s hourly max/min is carried through only as the
    diagnostic `hourly_value_f`, per its own coverage threshold. Mixing
    sources is never attempted: an untiled window is `extreme_source =
    'none'` with `value_f = None`, regardless of hourly coverage.
    """
    report_rows = list(reports)
    hourly_observations = [
        (valid, tmpf) for valid, tmpf, _max_6h_f, _min_6h_f in report_rows
    ]
    six_hour_reports = [
        (valid, max_6h_f, min_6h_f) for valid, _tmpf, max_6h_f, min_6h_f in report_rows
    ]

    hourly = observed_extreme(window, variable, hourly_observations)
    tiled = six_hour_extreme(window, variable, six_hour_reports)

    extreme_source: ExtremeSource = "metar_6h" if tiled.tiled else "none"

    return Observed(
        value_f=tiled.value_f,
        extreme_source=extreme_source,
        scorable=tiled.tiled,
        periods_found=tiled.periods_found,
        hourly_value_f=hourly.value_f,
        n_obs=hourly.n_obs,
        hours_covered=hourly.hours_covered,
        hours_expected=hourly.hours_expected,
    )
