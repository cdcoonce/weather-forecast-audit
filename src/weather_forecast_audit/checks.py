"""Pure evaluators backing the Dagster asset checks (build spec #10 D7).

No Dagster imports here: `definitions.py` wires these into
`AssetCheckResult`s, but the pass/fail logic and its edge cases (a `None`
latest timestamp, an exact boundary, a zero-station total) are tested here
as plain functions.

#16 tunes `GUIDANCE_MAX_AGE_HOURS`/`OBS_MAX_AGE_HOURS` to the real ingest
schedule; the values here are placeholders wide enough not to fire on a
manual or backfill run.
"""

from datetime import datetime, timedelta

GUIDANCE_MAX_AGE_HOURS = 36
OBS_MAX_AGE_HOURS = 30
GAP_RATE_WARN = 0.05


def evaluate_freshness(
    latest: datetime | None, now: datetime, max_age: timedelta
) -> tuple[bool, dict]:
    """Whether `latest` is no older than `max_age` as of `now`.

    `None` (an empty table -- nothing has ever landed) always fails. The
    boundary is inclusive: an age exactly equal to `max_age` passes.
    """
    max_age_hours = max_age.total_seconds() / 3600
    if latest is None:
        return False, {"age_hours": None, "max_age_hours": max_age_hours}

    age = now - latest
    passed = age <= max_age
    metadata = {"age_hours": age.total_seconds() / 3600, "max_age_hours": max_age_hours}
    return passed, metadata


def evaluate_gap_rate(
    stations_with_gap: int, stations_total: int, threshold: float
) -> tuple[bool, float]:
    """The share of processed stations with at least one gap, vs `threshold`.

    Passes when `rate <= threshold`. Raises if `stations_total` is 0: a rate
    is undefined with nothing processed, not vacuously passing or failing.
    """
    if stations_total == 0:
        msg = "stations_total must be > 0 to compute a gap rate"
        raise ValueError(msg)

    rate = stations_with_gap / stations_total
    return rate <= threshold, rate
