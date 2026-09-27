"""National backfill sensor (issue #11).

Walks `backfill.plan_units` through `ingest_job` then `transform_job`, one
unit (or the transform) per evaluation that actually submits a run,
respecting rammingspeed's single shared run slot and the blackout window
around the other tenants' (oura, waga) 06:00-07:00 America/Phoenix
schedules. `decide` (in `backfill.py`) makes every real decision; this
module only translates its output into Dagster calls -- looking up the last
submitted run's status by tag, checking the shared slot, and building the
`RunRequest`s.

The sensor starts `DefaultSensorStatus.STOPPED`: a human turns it on from
the Dagster UI once the plan looks right (see
`docs/methodology.md#national-backfill-issue-11`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import dagster as dg

# Not part of dagster's public API (nothing under `dagster/__init__.py`
# exports the partition-range tag names), but this is the same mechanism
# `dagster/_cli/job.py`'s `--partition-range` and the asset-backfill daemon
# use to make a run cover more than one partition: the execution context
# reads these two tags straight off the run (see
# `dagster/_core/execution/context/system.py`), regardless of how the run
# was launched.
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
)

from weather_forecast_audit.assets import ARCHIVE_START
from weather_forecast_audit.backfill import (
    BackfillUnit,
    CursorState,
    Done,
    Halt,
    Submit,
    SubmitTransform,
    Wait,
    decide,
    in_blackout,
    plan_units,
)
from weather_forecast_audit.jobs import MAX_RUNTIME_SECONDS, ingest_job, transform_job
from weather_forecast_audit.resources import StationsResource

# D2's month-batched ingest already assumes a run receives at most one
# month; the national backfill also chunks the (sorted) station list, so
# each run's fetch volume (~12s/station-month, rate-limited to 1 req/s)
# stays inside MAX_RUNTIME_SECONDS. 573 CONUS stations x 12s would be
# ~6880s in one run -- far past the 600s cap -- so a month is split into
# chunks of 30 stations (~360s), one ingest_job run per chunk.
BACKFILL_CHUNK_SIZE = 30

MAX_BACKFILL_ATTEMPTS = 3

# The blackout must cover the whole time a run could still be occupying the
# shared slot after this evaluation launches it: the run's own hard cap
# (MAX_RUNTIME_SECONDS, enforced by run monitoring) plus a margin for the
# time between "the daemon launches the run" and "the run actually starts
# using the slot" (container/process start). This is what makes the 600s
# cap load-bearing for safety, not just a runtime budget: without it, a run
# started just before a blackout could still be occupying the slot when
# oura/waga need it at 06:00-07:00.
BLACKOUT_HORIZON_S = MAX_RUNTIME_SECONDS + 300

UNIT_TAG = "wfa/backfill_unit"
ATTEMPT_TAG = "wfa/backfill_attempt"
TRANSFORM_UNIT_ID = "transform"

# Any run in one of these states may still be using -- or about to use --
# the single shared run slot.
IN_FLIGHT_STATUSES = [
    dg.DagsterRunStatus.QUEUED,
    dg.DagsterRunStatus.NOT_STARTED,
    dg.DagsterRunStatus.STARTING,
    dg.DagsterRunStatus.STARTED,
    dg.DagsterRunStatus.CANCELING,
]


def _now() -> datetime:
    """The current time, as a naive UTC datetime (repo-wide convention).

    A module-level seam so tests can monkeypatch `sensors._now`, matching
    `backfill.in_blackout`'s own injectable-clock convention -- no sleeping
    in tests, no wall-clock flakiness.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _all_station_icaos() -> list[str]:
    """Every station the backfill should touch, in the seed registry's terms.

    `dbt/seeds/station_registry.csv` already excludes
    `station_exclusions.csv`'s stations at seed-build time
    (`scripts/registry/build_registry.py`), so `StationsResource().stations()`
    with no `only` filter already *is* "every station this backfill should
    touch" -- there is nothing further to subtract here.
    """
    return [station.icao for station in StationsResource().stations()]


def _load_cursor(raw: str | None) -> tuple[date | None, CursorState]:
    """The frozen plan end (`None` before the first evaluation) and the
    backfill state machine's own cursor, from one JSON envelope."""
    if not raw:
        return None, CursorState(next_index=0, attempt=1, last_run_id=None)
    envelope = json.loads(raw)
    plan_end_raw = envelope.get("plan_end")
    plan_end = date.fromisoformat(plan_end_raw) if plan_end_raw else None
    return plan_end, CursorState(**envelope["state"])


def _dump_cursor(plan_end: date, state: CursorState) -> str:
    envelope = {"plan_end": plan_end.isoformat(), "state": json.loads(state.to_json())}
    return json.dumps(envelope)


def _slot_busy(instance: dg.DagsterInstance) -> bool:
    """Any run on the instance queued or in progress, in any code location --
    rammingspeed's run slot is shared across every tenant, not just wfa."""
    return instance.get_runs_count(dg.RunsFilter(statuses=IN_FLIGHT_STATUSES)) > 0


def _run_status(instance: dg.DagsterInstance, tags: dict[str, str]) -> str | None:
    """The most recent run's status carrying every one of `tags`, or `None`
    if no such run has been submitted (or is not yet visible in storage)."""
    runs = instance.get_runs(filters=dg.RunsFilter(tags=tags), limit=1)
    if not runs:
        return None
    return runs[0].status.value


def _unit_run_request(unit: BackfillUnit, attempt: int) -> dg.RunRequest:
    run_key = f"{unit.unit_id}-a{attempt}"
    return dg.RunRequest(
        run_key=run_key,
        job_name=ingest_job.name,
        run_config={
            "resources": {"stations": {"config": {"only": list(unit.stations)}}}
        },
        tags={
            UNIT_TAG: unit.unit_id,
            ATTEMPT_TAG: str(attempt),
            "dagster/max_runtime": str(MAX_RUNTIME_SECONDS),
            ASSET_PARTITION_RANGE_START_TAG: unit.start.isoformat(),
            ASSET_PARTITION_RANGE_END_TAG: unit.end.isoformat(),
        },
    )


def _transform_run_request() -> dg.RunRequest:
    return dg.RunRequest(
        run_key=TRANSFORM_UNIT_ID,
        job_name=transform_job.name,
        tags={
            UNIT_TAG: TRANSFORM_UNIT_ID,
            "dagster/max_runtime": str(MAX_RUNTIME_SECONDS),
        },
    )


@dg.sensor(
    name="national_backfill_sensor",
    jobs=[ingest_job, transform_job],
    minimum_interval_seconds=60,
    default_status=dg.DefaultSensorStatus.STOPPED,
    description=(
        "Walks the national backfill (issue #11) through ingest_job then "
        "transform_job, one unit per tick, honoring the shared run slot and "
        "the oura/waga blackout window. Stopped by default -- start by hand."
    ),
)
def national_backfill_sensor(context: dg.SensorEvaluationContext) -> dg.SensorResult:
    plan_end, state = _load_cursor(context.cursor)
    if plan_end is None:
        # Frozen here, on first evaluation, so the plan can't grow while the
        # backfill is in flight: "yesterday" per the injectable clock, read
        # once and stored in the cursor from then on.
        plan_end = _now().date() - timedelta(days=1)

    units = plan_units(
        ARCHIVE_START, plan_end, _all_station_icaos(), BACKFILL_CHUNK_SIZE
    )

    if state.last_run_id is None:
        last_run_status = None
    elif state.transform_requested:
        last_run_status = _run_status(context.instance, {UNIT_TAG: TRANSFORM_UNIT_ID})
    else:
        unit = units[state.next_index]
        last_run_status = _run_status(
            context.instance,
            {UNIT_TAG: unit.unit_id, ATTEMPT_TAG: str(state.attempt)},
        )

    decision = decide(
        state,
        n_units=len(units),
        last_run_status=last_run_status,
        slot_busy=_slot_busy(context.instance),
        blackout=in_blackout(_now(), BLACKOUT_HORIZON_S),
        max_attempts=MAX_BACKFILL_ATTEMPTS,
    )

    if isinstance(decision, Submit):
        unit = units[decision.unit_index]
        new_state = replace(
            decision.new_state, last_run_id=f"{unit.unit_id}-a{decision.attempt}"
        )
        context.update_cursor(_dump_cursor(plan_end, new_state))
        return dg.SensorResult(run_requests=[_unit_run_request(unit, decision.attempt)])

    if isinstance(decision, SubmitTransform):
        new_state = replace(decision.new_state, last_run_id=TRANSFORM_UNIT_ID)
        context.update_cursor(_dump_cursor(plan_end, new_state))
        return dg.SensorResult(run_requests=[_transform_run_request()])

    if isinstance(decision, Wait):
        context.update_cursor(_dump_cursor(plan_end, state))
        return dg.SensorResult(skip_reason=dg.SkipReason(decision.reason))

    if isinstance(decision, Done):
        context.update_cursor(_dump_cursor(plan_end, state))
        return dg.SensorResult(skip_reason=dg.SkipReason("national backfill complete"))

    if isinstance(decision, Halt):
        # No cursor change: replaying the same (halted) cursor against the
        # same (still-terminal) run status halts again on every subsequent
        # tick, forever, until a human resets the cursor by hand.
        context.update_cursor(_dump_cursor(plan_end, state))
        return dg.SensorResult(skip_reason=dg.SkipReason(f"HALTED: {decision.reason}"))

    msg = f"unhandled backfill decision: {decision!r}"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover
