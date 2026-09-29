"""Pure backfill planning and state machine for the national backfill (issue #11).

No Dagster imports: `sensors.national_backfill_sensor` is the only
Dagster-aware caller. Everything here is exercised directly by
`tests/unit/test_backfill.py` with no running instance.

`plan_units` breaks the archive into calendar-month x station-chunk units
(D2's existing month-batched ingest already assumes a run receives at most
one month); `decide` is a pure state machine that turns one evaluation's
inputs (the persisted cursor, the outcome of the last submitted run, and the
current slot/blackout state) into exactly one `Decision`, so the sensor
itself only has to translate a `Decision` into Dagster calls.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from collections.abc import Sequence

PHOENIX = ZoneInfo("America/Phoenix")

# rammingspeed's other tenants (oura, waga) fire 06:00-07:00 America/Phoenix;
# the window is padded 15 minutes either side for a run that starts just
# before or lands just after one of those schedules.
BLACKOUT_WINDOWS: list[tuple[time, time]] = [(time(5, 45), time(7, 15))]

TERMINAL_RUN_STATUSES = frozenset({"SUCCESS", "FAILURE", "CANCELED"})
FAILED_RUN_STATUSES = frozenset({"FAILURE", "CANCELED"})


@dataclass(frozen=True)
class BackfillUnit:
    """One ingest_job run: a calendar month (clipped to the archive/end
    bounds), for one chunk of the (sorted) station list."""

    index: int
    start: date
    end: date
    stations: tuple[str, ...]
    chunk: int

    @property
    def unit_id(self) -> str:
        return f"{self.start:%Y-%m}-c{self.chunk:02d}"


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _month_windows(archive_start: date, end: date) -> list[tuple[date, date]]:
    """Calendar months from `archive_start` to `end`, both inclusive and clipped."""
    windows: list[tuple[date, date]] = []
    year, month = archive_start.year, archive_start.month
    while True:
        month_last = _month_end(year, month)
        window_start = max(date(year, month, 1), archive_start)
        window_end = min(month_last, end)
        windows.append((window_start, window_end))
        if month_last >= end:
            return windows
        month, year = (month + 1, year) if month < 12 else (1, year + 1)


def plan_units(
    archive_start: date,
    end: date,
    stations: Sequence[str],
    chunk_size: int,
) -> list[BackfillUnit]:
    """Every backfill unit from `archive_start` to `end`, month-major then
    chunk-minor, indexed 0..N-1 in that order.

    `stations` is sorted before chunking so unit membership is deterministic
    regardless of the caller's ordering. No stations means no units: there
    is nothing to ingest.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1, got {chunk_size}"
        raise ValueError(msg)
    if end < archive_start:
        msg = f"end ({end}) is before archive_start ({archive_start})"
        raise ValueError(msg)

    sorted_stations = sorted(stations)
    chunks = [
        tuple(sorted_stations[i : i + chunk_size])
        for i in range(0, len(sorted_stations), chunk_size)
    ]

    units: list[BackfillUnit] = []
    index = 0
    for window_start, window_end in _month_windows(archive_start, end):
        for chunk_index, chunk in enumerate(chunks):
            units.append(
                BackfillUnit(
                    index=index,
                    start=window_start,
                    end=window_end,
                    stations=chunk,
                    chunk=chunk_index,
                )
            )
            index += 1
    return units


def remap_plan_index(
    old_index: int,
    *,
    old_chunk_size: int,
    new_chunk_size: int,
    n_stations: int,
) -> int:
    """Convert a `next_index` from one chunking of the plan to another.

    The plan is month-major, chunk-minor, with `ceil(n_stations / chunk_size)`
    chunks per month. `old_index` is split into its month and its chunk within
    that month; the chunk implies `min(c * old_chunk_size, n_stations)`
    stations already done that month, and the new chunk is that count
    floor-divided by `new_chunk_size`.

    The division is a FLOOR on purpose: when the old boundary falls inside a
    new chunk, the new chunk starts at or before the first undone station, so
    up to `new_chunk_size - 1` already-done stations are redone. That is safe
    because the loaders are idempotent replace-by-station-and-date. Rounding
    up would instead skip undone stations, which is never acceptable.

    Use this together with a `generation` bump on the cursor: unit ids embed
    the chunk index (`2020-10-c15` names different stations after the chunk
    size changes), and Dagster's sensor daemon silently drops a reused
    run_key, so an unchanged generation could collide with a run submitted
    under the old chunking.
    """
    if old_chunk_size < 1:
        msg = f"old_chunk_size must be >= 1, got {old_chunk_size}"
        raise ValueError(msg)
    if new_chunk_size < 1:
        msg = f"new_chunk_size must be >= 1, got {new_chunk_size}"
        raise ValueError(msg)
    if n_stations < 1:
        msg = f"n_stations must be >= 1, got {n_stations}"
        raise ValueError(msg)
    if old_index < 0:
        msg = f"old_index must be >= 0, got {old_index}"
        raise ValueError(msg)

    chunks_per_month_old = -(-n_stations // old_chunk_size)
    chunks_per_month_new = -(-n_stations // new_chunk_size)
    month, old_chunk = divmod(old_index, chunks_per_month_old)
    stations_done = min(old_chunk * old_chunk_size, n_stations)
    return month * chunks_per_month_new + stations_done // new_chunk_size


def in_blackout(now_utc: datetime, horizon_s: int) -> bool:
    """True if `[now_utc, now_utc + horizon_s]` intersects a blackout window
    on any local (America/Phoenix) date it touches.

    `now_utc` is a naive UTC datetime (repo-wide convention; see
    `ClockResource.now`). The interval is converted to local time once and
    then checked against every blackout window on every local date the
    interval spans, so an interval that crosses local midnight is handled
    without special-casing.
    """
    start_local = now_utc.replace(tzinfo=UTC).astimezone(PHOENIX)
    end_local = (
        (now_utc + timedelta(seconds=horizon_s)).replace(tzinfo=UTC).astimezone(PHOENIX)
    )

    day = start_local.date()
    while day <= end_local.date():
        for window_start, window_end in BLACKOUT_WINDOWS:
            local_window_start = datetime.combine(day, window_start, tzinfo=PHOENIX)
            local_window_end = datetime.combine(day, window_end, tzinfo=PHOENIX)
            if start_local <= local_window_end and local_window_start <= end_local:
                return True
        day += timedelta(days=1)
    return False


@dataclass(frozen=True)
class CursorState:
    """The backfill sensor's persisted progress, serialized into its cursor.

    `last_run_id` is repurposed to hold the run_key of the most recently
    submitted run, not a Dagster-assigned run id: the sensor never has a
    real run id to hand `decide` at submission time (the run does not exist
    yet), and a deterministic run_key (`f"{unit_id}-a{attempt}-g{generation}"`,
    or `"transform-g{generation}"`) is exactly what the sensor needs anyway,
    to look the run back up next tick via `RunsFilter(tags={RUN_KEY_TAG:
    run_key})` or the equivalent `wfa/backfill_unit`/`wfa/backfill_attempt`/
    `wfa/backfill_generation` tags. This keeps `decide` fully pure: it never
    has to be told about Dagster's run-id generation, and the field's name
    matches the shape the build spec asked for.

    `submitted_at` (a naive-UTC ISO-8601 string, repo convention) is stamped
    by `decide` itself on every `Submit`/`SubmitTransform`, so a later
    evaluation can tell how long a run has been "not found" (see `decide`'s
    not-found-timeout rule). `generation` exists purely for a human reset:
    Dagster's sensor daemon dedupes `RunRequest`s by `run_key`, scoped to
    the sensor (`dagster/_daemon/sensor.py`'s `fetch_existing_runs` and
    `_get_or_create_sensor_run`), so replaying the *same* run_key after
    clearing `last_run_id` to retry a stuck unit would just find (or
    silently skip past) that same never-materialized run again. Bumping
    `generation` changes the run_key, so the next submission is guaranteed
    to be a fresh one. `decide` never changes `generation` itself -- only a
    human, editing the cursor by hand, does.
    """

    next_index: int
    attempt: int
    last_run_id: str | None
    transform_requested: bool = False
    submitted_at: str | None = None
    generation: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | None) -> CursorState:
        if not raw:
            return cls(next_index=0, attempt=1, last_run_id=None)
        # Old cursors predate `submitted_at`/`generation`; both have
        # defaults, so a dict missing either key still constructs cleanly.
        return cls(**json.loads(raw))


@dataclass(frozen=True)
class Submit:
    unit_index: int
    attempt: int
    new_state: CursorState


@dataclass(frozen=True)
class Wait:
    reason: str


@dataclass(frozen=True)
class SubmitTransform:
    new_state: CursorState


@dataclass(frozen=True)
class Done:
    pass


@dataclass(frozen=True)
class Halt:
    reason: str


Decision = Submit | Wait | SubmitTransform | Done | Halt


def decide(
    state: CursorState,
    *,
    n_units: int,
    last_run_status: str | None,
    slot_busy: bool,
    blackout: bool,
    now_utc: datetime,
    max_attempts: int = 3,
    not_found_timeout_s: int = 900,
) -> Decision:
    """One sensor evaluation's worth of decision-making, given the current
    cursor and this tick's read of the world (the last run's status, the
    shared run slot, and the blackout calendar).

    Rules, in order (matching the build spec):

    1. An outstanding run (`last_run_id` set, status not yet terminal --
       including "not found yet", read as `None`) waits, regardless of
       `slot_busy`/`blackout` -- UNLESS the status has been `None` for more
       than `not_found_timeout_s` since `submitted_at`, in which case it
       halts (see "Never-appeared runs" below).
    2. A resolved run's outcome is applied first, before anything else is
       decided: SUCCESS advances to the next unit (attempt reset to 1);
       FAILURE/CANCELED retries the same unit with `attempt + 1`, or halts
       if that exceeds `max_attempts`. A transform run's outcome resolves
       the whole backfill (Done on SUCCESS, Halt on FAILURE/CANCELED)
       instead. This bookkeeping happens even when a blackout or a busy
       slot will block the next actual submission (rules 4/5) -- the
       *decision* just isn't a Submit that evaluation.
    3. Every unit done: request the transform job (gated by the same
       slot/blackout checks as a unit submission), then wait for it, then
       Done.
    4. Otherwise, a busy shared run slot blocks submission.
    5. Otherwise, a blackout window blocks submission.
    6. Otherwise, submit the next unit. `Submit`/`SubmitTransform`'s
       `new_state` always carries `submitted_at=now_utc` (naive UTC,
       repo convention), stamped here rather than by the sensor.

    A `Halt` carries no new state: the sensor persists the cursor
    unchanged, so replaying the same (halted) cursor against the same
    (still-terminal, or still-missing) run status halts again, forever,
    until a human resets the cursor by hand.

    Never-appeared runs: `last_run_status is None` means "no run tagged
    with this run_key exists yet" -- either it hasn't been picked up by the
    daemon, or it never will be (a daemon crash between building the
    `RunRequest` and creating the run, or the run_key was already used).
    Dagster's sensor daemon dedupes `RunRequest`s by `run_key`, scoped to
    the sensor (`dagster/_daemon/sensor.py`'s `fetch_existing_runs`, which
    looks up every run tagged `RUN_KEY_TAG == run_key` for this sensor, and
    `_get_or_create_sensor_run`, which returns that existing run -- or
    silently skips creating a new one -- whenever one is found). So a
    cursor that keeps resubmitting the identical run_key for a run that
    never actually got created will keep finding nothing, forever: there is
    no automatic recovery, because the daemon will never mint a second run
    under the same key. Past `not_found_timeout_s`, this halts loudly
    instead of waiting silently forever, and says so.
    """
    if state.last_run_id is not None and last_run_status not in TERMINAL_RUN_STATUSES:
        if last_run_status is None and state.submitted_at is not None:
            submitted_at = datetime.fromisoformat(state.submitted_at)
            elapsed_s = (now_utc - submitted_at).total_seconds()
            if elapsed_s > not_found_timeout_s:
                return Halt(
                    f"submitted run {state.last_run_id} never appeared after "
                    f"{not_found_timeout_s}s: run_key dedupe (bump the cursor's "
                    "generation and clear last_run_id to force a fresh run_key) "
                    "or a daemon launch failure"
                )
        return Wait("run in flight")

    working = state
    if state.last_run_id is not None:
        # last_run_status is terminal here (the branch above returned
        # otherwise), so `working` always reflects a resolved run below.
        if state.transform_requested:
            if last_run_status == "SUCCESS":
                return Done()
            return Halt(f"transform run {state.last_run_id} ended {last_run_status}")
        if last_run_status == "SUCCESS":
            working = replace(
                state,
                next_index=state.next_index + 1,
                attempt=1,
                last_run_id=None,
                submitted_at=None,
            )
        else:
            next_attempt = state.attempt + 1
            if next_attempt > max_attempts:
                return Halt(
                    f"unit {state.next_index} failed {max_attempts} times "
                    f"(last run {state.last_run_id}, status {last_run_status})"
                )
            working = replace(
                state, attempt=next_attempt, last_run_id=None, submitted_at=None
            )

    if working.next_index >= n_units:
        if working.transform_requested:
            return Wait("transform pending")
        if slot_busy:
            return Wait("slot busy")
        if blackout:
            return Wait("blackout")
        return SubmitTransform(
            replace(working, transform_requested=True, submitted_at=now_utc.isoformat())
        )

    if slot_busy:
        return Wait("slot busy")
    if blackout:
        return Wait("blackout")
    return Submit(
        working.next_index,
        working.attempt,
        replace(working, submitted_at=now_utc.isoformat()),
    )
