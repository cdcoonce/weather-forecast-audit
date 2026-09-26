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
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import ceil
from typing import Literal

MIN_HOUR_COVERAGE = 0.75
WINDOW_HOURS = 18

Variable = Literal["max", "min"]


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
