#!/usr/bin/env python3
"""Backfill watcher (issue #11): print one line and exit only when the national
backfill needs a human (or an agent) to look, and say nothing otherwise.

The national backfill (`national_backfill_sensor`, see "National backfill" in
`docs/methodology.md`) submits ~2,117 units of ~3.5 minutes each. It halts and
never recovers on its own after 3 failed attempts of a unit, so someone has to
notice. Polling it through an agent session costs a full-context turn per
check; this script is the harness-independent detector instead. It polls the
Dagster GraphQL API and exits **only when there is news**, so it can run as an
exit-on-event background command (and later be wrapped by an out-of-band
notifier). Silence means healthy.

It is stdlib-only on purpose (no dagster, polars, duckdb, no `sys.path` edits):
it runs with plain `python3` on a laptop that has not synced the repo's heavy
dependencies.

Usage:
    export WFA_DAGSTER_URL=http://<dagster-host>:<port>/graphql
    python3 scripts/watch_backfill.py status
    python3 scripts/watch_backfill.py wait [--interval 60] [--slow-s 450] \\
        [--stall-s 10800] [--unreachable-polls 10] [--ignore-failed-attempts] \\
        [--max-s 0]

The URL comes from `--url`, else `$WFA_DAGSTER_URL`; with neither the script
prints one line to stderr and exits 2. The sensor, code location, and
repository names default to the production ones and can be overridden with
`--sensor`, `--location`, `--repository`.

`status` polls once and prints one line, exit 0 (`UNREACHABLE <error>`, exit 3):

    <UTC time> sensor=RUNNING gen=5 next_index=1055 attempt=1 \\
        tick=SKIPPED:run in flight halted=no median_s=240 last=2023-09-c11:STARTED:60s

`wait` polls every `--interval` seconds and, on the first poll that yields one
or more events, prints each as `EVENT <KIND> <detail>`, then one status line,
and exits 0. With `--max-s N` (N > 0) it instead prints `QUIET <seconds>` and
exits 0 after N seconds without an event, so a caller can tell the two apart.

Events:
    HALT            newest tick is `HALTED: ...` AND its cursor equals the
                    sensor's current cursor on generation, next_index, attempt.
                    The guard matters: right after a human resets the cursor
                    (bumped generation), the newest tick can still be the OLD
                    halt for up to a minute; that stale halt does not fire.
                    A deleted (null) cursor cannot be matched, so it does not
                    fire either.
    SENSOR_ERROR    newest tick status is FAILURE (the sensor itself crashed).
    STOPPED         the sensor's status is not RUNNING.
    COMPLETE        newest tick skip reason is `national backfill complete`.
                    Independent of the other events: a stopped sensor that
                    has also finished fires both STOPPED and COMPLETE.
    SLOW            a sensor run lasted >= --slow-s (default 450; the server's
                    hard cap is 600). Once per run, whether it is still running
                    or already finished.
    FAILED_ATTEMPT  a sensor run ended FAILURE or CANCELED that was not already
                    seen. Off with --ignore-failed-attempts.
    STALLED         the highest next_index seen has not increased for --stall-s
                    (default 10800 s, outlasting the ~105 minute morning
                    blackout). Fires once until the index advances again.
                    Timed on the monotonic clock, so a laptop sleep or a clock
                    step cannot fake a stall (nor a `--max-s` QUIET exit).
                    SLOW is the one wall-clock comparison, because it measures
                    against the server's epoch `startTime`.
    UNREACHABLE     --unreachable-polls consecutive failed polls (network or
                    HTTP error, bad JSON, or a GraphQL error / non-Sensor
                    result). Any successful poll resets the count.

One event per line: every server-derived value is collapsed to a single line
with control characters stripped and truncated to 200 characters before it is
printed, so a hostile or garbled field cannot forge an `EVENT ` line.

Numeric flags are validated (exit 2): `--interval`, `--slow-s`, `--stall-s`
must be > 0, `--unreachable-polls` >= 1, `--max-s` >= 0.

Baseline rule: on the first successful poll every already-finished sensor run
is marked seen, so history produces no SLOW or FAILED_ATTEMPT. A run still
running on that poll is evaluated for SLOW. HALT, STOPPED, COMPLETE, and
SENSOR_ERROR are state-based and fire on the first poll if true. Routine skip
reasons (`run in flight`, `slot busy`, `blackout`, `transform pending`) never
fire.

Do not re-arm `wait` while the sensor is still halted: it would fire again
immediately. Reset the cursor first (see "Starting, stopping, and resetting
the sensor" in `docs/methodology.md`).
"""

import argparse
import http.client
import json
import math
import os
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TextIO

JSON = dict[str, Any]

SENSOR_NAME = "national_backfill_sensor"
LOCATION_NAME = "weather-forecast-audit"
REPOSITORY_NAME = "__repository__"

REQUEST_TIMEOUT_S = 15
DETAIL_LIMIT = 200
MEDIAN_WINDOW = 10
FINISHED_FAILURES = frozenset({"FAILURE", "CANCELED"})
COMPLETE_REASON = "national backfill complete"
HALT_PREFIX = "HALTED:"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNREACHABLE = 3

Post = Callable[[str, JSON, float], JSON]
Clock = Callable[[], float]


class SnapshotError(Exception):
    """A poll could not produce a usable snapshot (transport, JSON, GraphQL)."""


# -- data ---------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    status: str
    timestamp: float
    skip_reason: str | None
    cursor: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class Run:
    run_id: str
    status: str
    start_time: float | None
    end_time: float | None
    unit: str
    attempt: str

    @property
    def finished(self) -> bool:
        return self.end_time is not None

    def duration(self, now: float) -> float | None:
        """Seconds the run took (finished) or has been going (running)."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time is not None else now
        return end - self.start_time


@dataclass(frozen=True)
class Snapshot:
    sensor_status: str
    cursor_state: dict[str, Any] | None  # the sensor's current cursor `state`
    ticks: list[Tick]
    runs: list[Run]  # this sensor's runs only, newest first

    @property
    def newest_tick(self) -> Tick | None:
        return max(self.ticks, key=lambda t: t.timestamp, default=None)


@dataclass(frozen=True)
class Event:
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"EVENT {self.kind} {self.detail}"


@dataclass(frozen=True)
class Config:
    slow_s: float = 450
    stall_s: float = 10800
    unreachable_polls: int = 10
    ignore_failed_attempts: bool = False


@dataclass
class Memory:
    first_poll: bool = True
    seen_finished: set[str] = field(default_factory=set)
    slow_reported: set[str] = field(default_factory=set)
    max_next_index: int | None = None
    last_advance_at: float | None = None
    stall_reported: bool = False
    consecutive_failures: int = 0


# -- parsing ------------------------------------------------------------


def _one_line(text: str) -> str:
    """Collapse all whitespace (newlines, NEL, U+2028/9 included) to single
    spaces and drop the other control / format characters (NUL, ESC, DEL...)."""
    kept = (
        " " if char.isspace() else char
        for char in text
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    return " ".join("".join(kept).split())


def _field(value: object, limit: int = DETAIL_LIMIT) -> str:
    """A server-derived value made safe for one output line, capped at `limit`."""
    return _one_line(str(value))[:limit]


def _epoch(value: object, name: str) -> float | None:
    """A server epoch (number or numeric string) as a float; None stays None."""
    if value is None:
        return None
    if isinstance(value, bool):
        msg = f"{name} is not a number"
        raise ValueError(msg)
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        msg = f"{name} is not finite"
        raise ValueError(msg)
    return number


def _parse_cursor(raw: object) -> dict[str, Any] | None:
    """The cursor's `state` dict, or None for a null / unparseable cursor."""
    if not isinstance(raw, str):
        return None
    try:
        envelope = json.loads(raw)
    except ValueError:
        return None
    state = envelope.get("state") if isinstance(envelope, dict) else None
    return state if isinstance(state, dict) else None


def _graphql_errors(response: JSON, what: str) -> None:
    if not isinstance(response, dict):
        msg = f"{what}: response is not a JSON object"
        raise SnapshotError(msg)
    errors = response.get("errors")
    if errors:
        first = errors[0] if isinstance(errors, list) else errors
        text = first.get("message") if isinstance(first, dict) else first
        msg = f"{what}: GraphQL error: {_field(text)}"
        raise SnapshotError(msg)


def _parse_tick(raw: JSON) -> Tick:
    error = raw.get("error")
    skip_reason = raw.get("skipReason")
    return Tick(
        status=str(raw.get("status")),
        timestamp=_epoch(raw.get("timestamp"), "tick timestamp") or 0.0,
        skip_reason=None if skip_reason is None else str(skip_reason),
        cursor=_parse_cursor(raw.get("cursor")),
        error=(
            None
            if not isinstance(error, dict) or error.get("message") is None
            else str(error["message"])
        ),
    )


def _parse_run(raw: JSON) -> Run:
    tags = {t.get("key"): t.get("value") for t in raw.get("tags") or []}
    return Run(
        run_id=str(raw.get("runId")),
        status=str(raw.get("status")),
        start_time=_epoch(raw.get("startTime"), "startTime"),
        end_time=_epoch(raw.get("endTime"), "endTime"),
        unit=tags.get("wfa/backfill_unit") or "?",
        attempt=tags.get("wfa/backfill_attempt") or "?",
    )


def parse_snapshot(
    sensor_response: JSON, runs_response: JSON, sensor_name: str = SENSOR_NAME
) -> Snapshot:
    """Parse the two raw GraphQL responses; raise `SnapshotError` on any
    error or unexpected shape."""
    _graphql_errors(sensor_response, "sensor query")
    _graphql_errors(runs_response, "runs query")
    try:
        sensor = sensor_response["data"]["sensorOrError"]
        runs = runs_response["data"]["runsOrError"]
    except (KeyError, TypeError) as exc:
        msg = f"unexpected response shape (missing {_field(exc)})"
        raise SnapshotError(msg) from exc
    if not isinstance(sensor, dict) or "sensorState" not in sensor:
        text = sensor.get("message") if isinstance(sensor, dict) else sensor
        msg = f"sensor query did not return a Sensor: {_field(text)}"
        raise SnapshotError(msg)
    if not isinstance(runs, dict) or "results" not in runs:
        text = runs.get("message") if isinstance(runs, dict) else runs
        msg = f"runs query did not return Runs: {_field(text)}"
        raise SnapshotError(msg)
    try:
        state = sensor["sensorState"]
        specific = state.get("typeSpecificData") or {}
        ticks = [_parse_tick(t) for t in state.get("ticks") or []]
        parsed_runs = [_parse_run(r) for r in runs["results"]]
        sensor_status = str(state["status"])
        cursor_state = _parse_cursor(specific.get("lastCursor"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        msg = f"unexpected response shape ({_field(repr(exc))})"
        raise SnapshotError(msg) from exc
    mine = []
    for raw, run in zip(runs["results"], parsed_runs, strict=True):
        tags = {t.get("key"): t.get("value") for t in raw.get("tags") or []}
        if tags.get("dagster/sensor_name") == sensor_name:
            mine.append(run)
    return Snapshot(
        sensor_status=sensor_status,
        cursor_state=cursor_state,
        ticks=ticks,
        runs=mine,
    )


# -- fetching -----------------------------------------------------------


def sensor_query(location: str, repository: str, sensor_name: str) -> str:
    selector = (
        f"repositoryLocationName:{json.dumps(location)},"
        f"repositoryName:{json.dumps(repository)},"
        f"sensorName:{json.dumps(sensor_name)}"
    )
    return (
        "{ sensorOrError(sensorSelector:{" + selector + "})"
        "{ ... on Sensor { sensorState { status typeSpecificData "
        "{ ... on SensorData { lastCursor lastTickTimestamp } } "
        "ticks(limit:5){ status timestamp skipReason cursor error{message} } } } } }"
    )


RUNS_QUERY = (
    "{ runsOrError(limit:12){ ... on Runs { results "
    "{ runId status startTime endTime tags{key value} } } } }"
)


def urllib_post(url: str, payload: JSON, timeout: float) -> JSON:
    """POST `payload` as JSON and return the decoded JSON response."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_snapshot(
    url: str,
    post: Post = urllib_post,
    *,
    location: str = LOCATION_NAME,
    repository: str = REPOSITORY_NAME,
    sensor_name: str = SENSOR_NAME,
) -> Snapshot:
    """Issue the sensor and runs queries and parse them; every failure mode
    (network, HTTP, bad JSON, GraphQL error) surfaces as `SnapshotError`."""
    query = sensor_query(location, repository, sensor_name)
    try:
        sensor_response = post(url, {"query": query}, REQUEST_TIMEOUT_S)
        runs_response = post(url, {"query": RUNS_QUERY}, REQUEST_TIMEOUT_S)
    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ) as exc:
        msg = f"{type(exc).__name__}: {_field(exc)}"
        raise SnapshotError(msg) from exc
    return parse_snapshot(sensor_response, runs_response, sensor_name)


# -- evaluation ---------------------------------------------------------


def _triple(state: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
    if state is None:
        return None
    return (state.get("generation"), state.get("next_index"), state.get("attempt"))


def _halt_is_current(snapshot: Snapshot) -> bool:
    tick = snapshot.newest_tick
    if tick is None or not (tick.skip_reason or "").startswith(HALT_PREFIX):
        return False
    current = _triple(snapshot.cursor_state)
    return current is not None and _triple(tick.cursor) == current


def _fmt_seconds(seconds: float) -> str:
    return f"{round(seconds)}s"


def evaluate(
    snapshot: Snapshot, memory: Memory, config: Config, now: float, mono: float
) -> list[Event]:
    """Events this poll warrants. Mutates `memory`; otherwise pure.

    `now` is the wall clock (comparable with the server's epoch `startTime`,
    used for SLOW durations); `mono` is a monotonic reading, used only for the
    STALLED idle timer so a clock step or laptop sleep cannot fake a stall.
    """
    events: list[Event] = []
    tick = snapshot.newest_tick
    state = snapshot.cursor_state or {}

    if snapshot.sensor_status != "RUNNING":
        events.append(
            Event("STOPPED", f"sensor status={_field(snapshot.sensor_status)}")
        )
    if _halt_is_current(snapshot):
        events.append(
            Event(
                "HALT",
                f"{_field(tick.skip_reason)} "
                f"gen={_field(state.get('generation'))} "
                f"next_index={_field(state.get('next_index'))}",
            )
        )
    if tick is not None and tick.status == "FAILURE":
        message = _field(tick.error or "no error message")
        events.append(Event("SENSOR_ERROR", message))
    if tick is not None and tick.skip_reason == COMPLETE_REASON:
        events.append(Event("COMPLETE", COMPLETE_REASON))

    ordered = sorted(snapshot.runs, key=lambda r: r.start_time or 0.0)
    for run in ordered:
        if memory.first_poll and run.finished:
            memory.seen_finished.add(run.run_id)
            memory.slow_reported.add(run.run_id)
            continue
        duration = run.duration(now)
        if (
            run.run_id not in memory.slow_reported
            and duration is not None
            and duration >= config.slow_s
        ):
            memory.slow_reported.add(run.run_id)
            events.append(
                Event(
                    "SLOW",
                    f"unit={_field(run.unit)} attempt={_field(run.attempt)} "
                    f"duration={_fmt_seconds(duration)} status={_field(run.status)}",
                )
            )
        if run.finished and run.run_id not in memory.seen_finished:
            memory.seen_finished.add(run.run_id)
            if run.status in FINISHED_FAILURES and not config.ignore_failed_attempts:
                events.append(
                    Event(
                        "FAILED_ATTEMPT",
                        f"unit={_field(run.unit)} attempt={_field(run.attempt)} "
                        f"status={_field(run.status)} "
                        f"duration={_fmt_seconds(duration or 0)}",
                    )
                )

    index = state.get("next_index")
    if isinstance(index, int):
        if memory.max_next_index is None or index > memory.max_next_index:
            memory.max_next_index = index
            memory.last_advance_at = mono
            memory.stall_reported = False
        elif (
            not memory.stall_reported
            and memory.last_advance_at is not None
            and mono - memory.last_advance_at >= config.stall_s
        ):
            memory.stall_reported = True
            idle = _fmt_seconds(mono - memory.last_advance_at)
            events.append(Event("STALLED", f"next_index={index} unchanged for {idle}"))

    memory.first_poll = False
    memory.consecutive_failures = 0
    return events


def note_poll_failure(memory: Memory, config: Config, error: str) -> list[Event]:
    """Record a failed poll; return UNREACHABLE when the streak reaches N."""
    memory.consecutive_failures += 1
    if memory.consecutive_failures == config.unreachable_polls:
        return [Event("UNREACHABLE", _field(error))]
    return []


# -- output -------------------------------------------------------------


def _iso(now: float) -> str:
    return datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_status(snapshot: Snapshot, now: float) -> str:
    state = snapshot.cursor_state or {}

    def field_of(key: str) -> str:
        value = state.get(key)
        return "n/a" if value is None else _field(value)

    tick = snapshot.newest_tick
    if tick is None:
        tick_text = "none"
    elif tick.skip_reason:
        tick_text = f"{_field(tick.status)}:{_field(tick.skip_reason)}"
    else:
        tick_text = _field(tick.status)

    by_start = sorted(snapshot.runs, key=lambda r: r.start_time or 0.0, reverse=True)
    durations = [
        r.duration(now) for r in by_start if r.finished and r.start_time is not None
    ][:MEDIAN_WINDOW]
    median = f"{round(statistics.median(durations))}" if durations else "n/a"
    last = "none"
    if by_start:
        run = by_start[0]
        duration = run.duration(now)
        shown = "?" if duration is None else _fmt_seconds(duration)
        last = f"{_field(run.unit)}:{_field(run.status)}:{shown}"

    return (
        f"{_iso(now)} sensor={_field(snapshot.sensor_status)} "
        f"gen={field_of('generation')} next_index={field_of('next_index')} "
        f"attempt={field_of('attempt')} tick={tick_text} "
        f"halted={'yes' if _halt_is_current(snapshot) else 'no'} "
        f"median_s={median} last={last}"
    )


# -- loop ---------------------------------------------------------------


def watch(
    *,
    fetch: Callable[[], Snapshot],
    config: Config,
    interval: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Clock = time.time,
    mono: Clock = time.monotonic,
    out: TextIO | None = None,
    max_s: float = 0,
) -> int:
    """Poll until an event (print it plus a status line, return 0) or, when
    `max_s` > 0, until that many seconds pass quietly (print QUIET, return 0).

    `clock` is the wall clock (SLOW and the status timestamp); `mono` is the
    monotonic clock (the STALLED idle timer and the `max_s` timer)."""
    out = out if out is not None else sys.stdout
    memory = Memory()
    started = mono()
    while True:
        now = clock()
        mono_now = mono()
        snapshot: Snapshot | None = None
        try:
            snapshot = fetch()
        except SnapshotError as exc:
            events = note_poll_failure(memory, config, str(exc))
        else:
            events = evaluate(snapshot, memory, config, now, mono_now)
        if events:
            for event in events:
                print(event, file=out)
            if snapshot is not None:
                print(format_status(snapshot, now), file=out)
            out.flush()
            return EXIT_OK
        elapsed = mono() - started
        if max_s > 0:
            if elapsed >= max_s:
                print(f"QUIET {int(elapsed)}", file=out)
                out.flush()
                return EXIT_OK
            sleep(min(interval, max_s - elapsed))
        else:
            sleep(interval)


# -- CLI ----------------------------------------------------------------


def _number(text: str, *, minimum: float, inclusive: bool) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        msg = f"not a number: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if (
        not math.isfinite(value)
        or value < minimum
        or (not inclusive and value == minimum)
    ):
        bound = ">=" if inclusive else ">"
        msg = f"must be a finite number {bound} {minimum:g}, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _positive_float(text: str) -> float:
    return _number(text, minimum=0, inclusive=False)


def _non_negative_float(text: str) -> float:
    return _number(text, minimum=0, inclusive=True)


def _at_least_one(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        msg = f"not an integer: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if value < 1:
        msg = f"must be an integer >= 1, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one line and exit only when the national backfill "
        "needs attention (see the module docstring)."
    )
    parser.add_argument("mode", choices=["status", "wait"])
    parser.add_argument("--url", help="Dagster GraphQL URL (default: $WFA_DAGSTER_URL)")
    parser.add_argument("--interval", type=_positive_float, default=60)
    parser.add_argument("--slow-s", type=_positive_float, default=Config.slow_s)
    parser.add_argument("--stall-s", type=_positive_float, default=Config.stall_s)
    parser.add_argument(
        "--unreachable-polls", type=_at_least_one, default=Config.unreachable_polls
    )
    parser.add_argument("--ignore-failed-attempts", action="store_true")
    parser.add_argument(
        "--max-s",
        type=_non_negative_float,
        default=0,
        help="print QUIET and exit 0 after this many event-free seconds (0 = never)",
    )
    parser.add_argument("--sensor", default=SENSOR_NAME)
    parser.add_argument("--location", default=LOCATION_NAME)
    parser.add_argument("--repository", default=REPOSITORY_NAME)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    post: Post = urllib_post,
    clock: Clock = time.time,
    mono: Clock = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = _build_parser().parse_args(argv)
    url = args.url or os.environ.get("WFA_DAGSTER_URL")
    if not url:
        print("error: set WFA_DAGSTER_URL or pass --url", file=err)
        return EXIT_USAGE

    def fetch() -> Snapshot:
        return fetch_snapshot(
            url,
            post=post,
            location=args.location,
            repository=args.repository,
            sensor_name=args.sensor,
        )

    if args.mode == "status":
        try:
            snapshot = fetch()
        except SnapshotError as exc:
            print(f"UNREACHABLE {_field(exc)}", file=out)
            return EXIT_UNREACHABLE
        print(format_status(snapshot, clock()), file=out)
        return EXIT_OK

    config = Config(
        slow_s=args.slow_s,
        stall_s=args.stall_s,
        unreachable_polls=args.unreachable_polls,
        ignore_failed_attempts=args.ignore_failed_attempts,
    )
    return watch(
        fetch=fetch,
        config=config,
        interval=args.interval,
        sleep=sleep,
        clock=clock,
        mono=mono,
        out=out,
        max_s=args.max_s,
    )


if __name__ == "__main__":
    sys.exit(main())
