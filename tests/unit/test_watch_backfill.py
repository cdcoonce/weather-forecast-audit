"""Tests for `scripts/watch_backfill.py`, the exit-on-event backfill watcher.

The script is stdlib-only and lives in `scripts/` (not a package), so it is
loaded by path. Every response here is built by loading the two real captures
in `tests/fixtures/backfill_watcher/` (raw GraphQL from the live server) and
mutating copies; no response shape is invented. Nothing touches the network:
`post`, `fetch`, `sleep`, `clock`, and `out` are all injected.
"""

import ast
import copy
import importlib.util
import io
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "watch_backfill.py"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "backfill_watcher"
SENSOR_NAME = "national_backfill_sensor"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("watch_backfill", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["watch_backfill"] = module  # dataclasses resolve cls.__module__
    spec.loader.exec_module(module)
    return module


wb = _load_script()

# The live capture's one STARTED run, and the fixed "now" 60 s into it.
LIVE_START = 1791159656.310995
HEALTHY_NOW = LIVE_START + 60


def _sensor() -> dict:
    return json.loads((FIXTURES / "sensor_live_2026-10-04.json").read_text())


def _runs() -> dict:
    return json.loads((FIXTURES / "runs_live_2026-10-04.json").read_text())


def _state(sensor: dict) -> dict:
    return sensor["data"]["sensorOrError"]["sensorState"]


def _results(runs: dict) -> list[dict]:
    return runs["data"]["runsOrError"]["results"]


def _newest_tick(sensor: dict) -> dict:
    return max(_state(sensor)["ticks"], key=lambda t: t["timestamp"])


def _cursor(sensor: dict) -> dict:
    return json.loads(_state(sensor)["typeSpecificData"]["lastCursor"])


def _set_cursor(sensor: dict, **state_updates: int) -> str:
    """Rewrite the sensor's lastCursor `state` fields; return the new raw JSON."""
    cursor = _cursor(sensor)
    cursor["state"].update(state_updates)
    raw = json.dumps(cursor)
    _state(sensor)["typeSpecificData"]["lastCursor"] = raw
    return raw


def _halted_sensor(reason: str = "HALTED: unit 2023-09-c11 failed 3 attempts") -> dict:
    """Live sensor whose newest tick is a halt matching the current cursor."""
    sensor = _sensor()
    tick = _newest_tick(sensor)
    tick["status"] = "SKIPPED"
    tick["skipReason"] = reason
    tick["cursor"] = _state(sensor)["typeSpecificData"]["lastCursor"]
    return sensor


def _run(
    runs: dict,
    *,
    run_id: str,
    status: str,
    start: float,
    end: float | None,
    unit: str = "2023-09-c99",
    attempt: str = "1",
    sensor_name: str = SENSOR_NAME,
) -> dict:
    """A new run built from a copy of a real run in the capture."""
    run = copy.deepcopy(_results(runs)[1])
    run.update(runId=run_id, status=status, startTime=start, endTime=end)
    for tag in run["tags"]:
        if tag["key"] == "dagster/sensor_name":
            tag["value"] = sensor_name
        elif tag["key"] == "wfa/backfill_unit":
            tag["value"] = unit
        elif tag["key"] == "wfa/backfill_attempt":
            tag["value"] = attempt
    return run


def _config(**overrides: float | bool) -> object:
    return wb.Config(**overrides)


def _poll(sensor: dict, runs: dict, memory: object, config: object, now: float) -> list:
    return wb.evaluate(wb.parse_snapshot(sensor, runs), memory, config, now)


def _kinds(events: list) -> list[str]:
    return [e.kind for e in events]


# -- structure ----------------------------------------------------------


def test_script_is_executable_with_shebang() -> None:
    assert os.access(SCRIPT, os.X_OK)
    assert SCRIPT.read_text().startswith("#!/usr/bin/env python3\n")


def test_script_imports_only_the_standard_library() -> None:
    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module.split(".")[0])
    assert imported
    assert imported <= set(sys.stdlib_module_names)


# -- parse_snapshot -----------------------------------------------------


def test_parse_live_capture() -> None:
    snap = wb.parse_snapshot(_sensor(), _runs())
    assert snap.sensor_status == "RUNNING"
    assert snap.cursor_state == {
        "next_index": 1055,
        "attempt": 1,
        "last_run_id": "2023-09-c11-a1-g5",
        "transform_requested": False,
        "submitted_at": "2026-10-05T00:20:41.240255",
        "generation": 5,
    }
    assert len(snap.ticks) == 5
    assert snap.newest_tick.skip_reason is None
    assert [r.status for r in snap.runs][:2] == ["STARTED", "SUCCESS"]
    assert snap.runs[0].unit == "2023-09-c11"
    assert snap.runs[0].attempt == "1"
    assert snap.runs[0].end_time is None


def test_parse_keeps_only_runs_of_this_sensor() -> None:
    runs = _runs()
    runs["data"]["runsOrError"]["results"].append(
        _run(
            runs,
            run_id="other",
            status="FAILURE",
            start=1.0,
            end=2.0,
            sensor_name="some_other_sensor",
        )
    )
    runs["data"]["runsOrError"]["results"].append(
        {
            "runId": "manual",
            "status": "SUCCESS",
            "startTime": 1.0,
            "endTime": 2.0,
            "tags": [],
        }
    )
    snap = wb.parse_snapshot(_sensor(), runs)
    assert "other" not in [r.run_id for r in snap.runs]
    assert "manual" not in [r.run_id for r in snap.runs]
    assert len(snap.runs) == 6


def test_parse_null_or_garbled_cursor_gives_no_cursor_state() -> None:
    sensor = _sensor()
    _state(sensor)["typeSpecificData"]["lastCursor"] = None
    assert wb.parse_snapshot(sensor, _runs()).cursor_state is None
    _state(sensor)["typeSpecificData"]["lastCursor"] = "{not json"
    assert wb.parse_snapshot(sensor, _runs()).cursor_state is None


def test_parse_empty_ticks_has_no_newest_tick() -> None:
    sensor = _sensor()
    _state(sensor)["ticks"] = []
    assert wb.parse_snapshot(sensor, _runs()).newest_tick is None


def test_newest_tick_is_largest_timestamp_not_first_listed() -> None:
    sensor = _sensor()
    _state(sensor)["ticks"].reverse()
    assert wb.parse_snapshot(sensor, _runs()).newest_tick.timestamp == 1791159640.818564


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, r: s.update(errors=[{"message": "boom"}]),
        lambda s, r: s["data"].update(
            sensorOrError={"__typename": "SensorNotFoundError", "message": "gone"}
        ),
        lambda s, r: s["data"].update(
            sensorOrError={"__typename": "PythonError", "message": "py"}
        ),
        lambda s, r: s.pop("data"),
        lambda s, r: r.update(errors=[{"message": "boom"}]),
        lambda s, r: r["data"].update(runsOrError={"message": "no results"}),
        lambda s, r: r.pop("data"),
    ],
)
def test_parse_raises_snapshot_error_on_error_shapes(
    mutate: Callable[[dict, dict], object],
) -> None:
    sensor, runs = _sensor(), _runs()
    mutate(sensor, runs)
    with pytest.raises(wb.SnapshotError):
        wb.parse_snapshot(sensor, runs)


# -- healthy / routine --------------------------------------------------


def test_healthy_live_capture_first_poll_has_no_events() -> None:
    events = _poll(_sensor(), _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert events == []


@pytest.mark.parametrize(
    "reason", ["blackout", "slot busy", "run in flight", "transform pending"]
)
def test_routine_skip_reasons_never_fire(reason: str) -> None:
    sensor = _sensor()
    tick = _newest_tick(sensor)
    tick["status"] = "SKIPPED"
    tick["skipReason"] = reason
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


# -- HALT ---------------------------------------------------------------


def test_halt_fires_on_first_poll_when_tick_matches_cursor() -> None:
    events = _poll(_halted_sensor(), _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert _kinds(events) == ["HALT"]
    assert "HALTED: unit 2023-09-c11 failed 3 attempts" in events[0].detail
    assert "gen=5" in events[0].detail
    assert "next_index=1055" in events[0].detail


def test_stale_halt_after_generation_bump_does_not_fire_but_new_halt_does() -> None:
    sensor = _halted_sensor()
    # A human resets the cursor: new generation, attempt back to 1. The newest
    # tick is still the OLD halt (it carries the old cursor).
    _set_cursor(sensor, generation=6, attempt=1)
    memory = wb.Memory()
    assert _poll(sensor, _runs(), memory, _config(), HEALTHY_NOW) == []
    # Later the new generation itself halts: the tick now carries the new cursor.
    _newest_tick(sensor)["cursor"] = _state(sensor)["typeSpecificData"]["lastCursor"]
    events = _poll(sensor, _runs(), memory, _config(), HEALTHY_NOW + 60)
    assert _kinds(events) == ["HALT"]
    assert "gen=6" in events[0].detail


@pytest.mark.parametrize(
    "change", [{"generation": 6}, {"next_index": 1056}, {"attempt": 2}]
)
def test_halt_requires_generation_next_index_and_attempt_to_match(
    change: dict[str, int],
) -> None:
    sensor = _halted_sensor()
    _set_cursor(sensor, **change)
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


def test_halt_does_not_fire_when_cursor_was_deleted() -> None:
    sensor = _halted_sensor()
    _state(sensor)["typeSpecificData"]["lastCursor"] = None
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


def test_halt_only_considers_the_newest_tick() -> None:
    sensor = _halted_sensor()
    older = copy.deepcopy(_newest_tick(sensor))
    older["timestamp"] -= 1000
    # The newest tick becomes healthy; the (matching) halt is now history.
    _newest_tick(sensor).update(status="SKIPPED", skipReason="run in flight")
    _state(sensor)["ticks"].append(older)
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


# -- SENSOR_ERROR / STOPPED / COMPLETE ----------------------------------


def test_sensor_error_fires_with_truncated_message() -> None:
    sensor = _sensor()
    tick = _newest_tick(sensor)
    tick["status"] = "FAILURE"
    tick["error"] = {"message": "x" * 500}
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert _kinds(events) == ["SENSOR_ERROR"]
    assert "x" * 200 in events[0].detail
    assert "x" * 201 not in events[0].detail


def test_sensor_error_with_null_error_still_fires() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(status="FAILURE", error=None)
    assert _kinds(_poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)) == [
        "SENSOR_ERROR"
    ]


def test_older_failed_tick_does_not_fire() -> None:
    sensor = _sensor()
    _state(sensor)["ticks"][-1]["status"] = "FAILURE"
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


def test_stopped_fires() -> None:
    sensor = _sensor()
    _state(sensor)["status"] = "STOPPED"
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert _kinds(events) == ["STOPPED"]
    assert "STOPPED" in events[0].detail


def test_complete_fires_on_exact_skip_reason() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(
        status="SKIPPED", skipReason="national backfill complete"
    )
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert _kinds(events) == ["COMPLETE"]


def test_complete_is_exact_match_only() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(
        status="SKIPPED", skipReason="national backfill complete soon"
    )
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


# -- SLOW ---------------------------------------------------------------


def test_slow_running_run_fires_at_threshold_not_before() -> None:
    sensor, runs = _sensor(), _runs()
    assert _poll(sensor, runs, wb.Memory(), _config(), LIVE_START + 449) == []
    events = _poll(sensor, runs, wb.Memory(), _config(), LIVE_START + 450)
    assert _kinds(events) == ["SLOW"]
    detail = events[0].detail
    assert "2023-09-c11" in detail
    assert "attempt=1" in detail
    assert "450" in detail
    assert "STARTED" in detail


def test_slow_threshold_follows_the_config() -> None:
    sensor, runs = _sensor(), _runs()
    events = _poll(sensor, runs, wb.Memory(), _config(slow_s=100), LIVE_START + 100)
    assert _kinds(events) == ["SLOW"]


def test_slow_reports_once_per_run_across_polls_and_on_finish() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    assert _kinds(_poll(sensor, runs, memory, _config(), LIVE_START + 460)) == ["SLOW"]
    assert _poll(sensor, runs, memory, _config(), LIVE_START + 520) == []
    # The same run now finishes (SUCCESS, 530 s): still no second SLOW.
    _results(runs)[0].update(status="SUCCESS", endTime=LIVE_START + 530)
    assert _poll(sensor, runs, memory, _config(), LIVE_START + 540) == []


def test_baseline_finished_slow_runs_are_silent_but_a_later_slow_run_fires() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    _results(runs).append(
        _run(runs, run_id="old-slow", status="SUCCESS", start=1000.0, end=1700.0)
    )
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs)[0].update(status="SUCCESS", endTime=LIVE_START + 240)
    _results(runs).insert(
        0,
        _run(
            runs,
            run_id="new-slow",
            status="SUCCESS",
            start=LIVE_START + 70,
            end=LIVE_START + 600,
        ),
    )
    events = _poll(sensor, runs, memory, _config(), LIVE_START + 610)
    assert _kinds(events) == ["SLOW"]
    assert "530" in events[0].detail


def test_slow_uses_end_minus_start_for_finished_runs() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    # Finished 449 s after starting: not slow, however late we notice it.
    _results(runs)[0].update(status="SUCCESS", endTime=LIVE_START + 449)
    assert _poll(sensor, runs, memory, _config(), LIVE_START + 5000) == []


# -- FAILED_ATTEMPT -----------------------------------------------------


@pytest.mark.parametrize("status", ["FAILURE", "CANCELED"])
def test_failed_attempt_fires_for_new_failed_run_after_silent_baseline(
    status: str,
) -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    _results(runs).append(
        _run(runs, run_id="old-fail", status="FAILURE", start=1000.0, end=1100.0)
    )
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs).insert(
        1,
        _run(
            runs,
            run_id="new-fail",
            status=status,
            start=LIVE_START - 100,
            end=LIVE_START - 50,
            unit="2023-09-c10",
            attempt="2",
        ),
    )
    events = _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 60)
    assert _kinds(events) == ["FAILED_ATTEMPT"]
    detail = events[0].detail
    assert "2023-09-c10" in detail
    assert "attempt=2" in detail
    assert status in detail
    assert "50" in detail
    # Reported once.
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 120) == []


def test_failed_attempt_for_run_seen_started_fires_when_it_fails() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs)[0].update(status="FAILURE", endTime=LIVE_START + 120)
    events = _poll(sensor, runs, memory, _config(), LIVE_START + 130)
    assert _kinds(events) == ["FAILED_ATTEMPT"]
    assert "2023-09-c11" in events[0].detail


def test_successful_new_run_is_not_news() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs)[0].update(status="SUCCESS", endTime=LIVE_START + 240)
    assert _poll(sensor, runs, memory, _config(), LIVE_START + 250) == []


def test_ignore_failed_attempts_suppresses_the_event() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(ignore_failed_attempts=True)
    assert _poll(sensor, runs, memory, config, HEALTHY_NOW) == []
    _results(runs)[0].update(status="FAILURE", endTime=LIVE_START + 120)
    assert _poll(sensor, runs, memory, config, LIVE_START + 130) == []


def test_other_sensors_failed_runs_do_not_fire() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs).insert(
        0,
        _run(
            runs,
            run_id="x",
            status="FAILURE",
            start=LIVE_START,
            end=LIVE_START + 5,
            sensor_name="other",
        ),
    )
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 60) == []


# -- STALLED ------------------------------------------------------------


def test_stalled_fires_once_after_stall_s_without_advance() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=10800, slow_s=10**9)
    t0 = HEALTHY_NOW
    assert _poll(sensor, runs, memory, config, t0) == []
    assert _poll(sensor, runs, memory, config, t0 + 10799) == []
    events = _poll(sensor, runs, memory, config, t0 + 10800)
    assert _kinds(events) == ["STALLED"]
    assert "1055" in events[0].detail
    assert _poll(sensor, runs, memory, config, t0 + 12000) == []


def test_stall_clock_resets_when_next_index_advances() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=1000, slow_s=10**9)
    t0 = HEALTHY_NOW
    assert _poll(sensor, runs, memory, config, t0) == []
    _set_cursor(sensor, next_index=1056)
    assert _poll(sensor, runs, memory, config, t0 + 900) == []
    assert _poll(sensor, runs, memory, config, t0 + 1800) == []
    assert _kinds(_poll(sensor, runs, memory, config, t0 + 1900)) == ["STALLED"]


def test_stall_ignores_a_lower_next_index_after_reset() -> None:
    # "Highest next_index seen": a reset to a lower index is not an advance.
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=1000, slow_s=10**9)
    assert _poll(sensor, runs, memory, config, HEALTHY_NOW) == []
    _set_cursor(sensor, next_index=3)
    assert _kinds(_poll(sensor, runs, memory, config, HEALTHY_NOW + 1000)) == [
        "STALLED"
    ]


def test_default_stall_window_outlasts_the_morning_blackout() -> None:
    assert wb.Config().stall_s > 105 * 60


# -- UNREACHABLE --------------------------------------------------------


def test_unreachable_needs_n_consecutive_failures() -> None:
    memory, config = wb.Memory(), _config(unreachable_polls=3)
    assert wb.note_poll_failure(memory, config, "timed out") == []
    assert wb.note_poll_failure(memory, config, "timed out") == []
    events = wb.note_poll_failure(memory, config, "connection refused")
    assert _kinds(events) == ["UNREACHABLE"]
    assert "connection refused" in events[0].detail


def test_unreachable_counter_resets_on_a_successful_poll() -> None:
    memory, config = wb.Memory(), _config(unreachable_polls=3)
    wb.note_poll_failure(memory, config, "e")
    wb.note_poll_failure(memory, config, "e")
    assert _poll(_sensor(), _runs(), memory, config, HEALTHY_NOW) == []
    assert wb.note_poll_failure(memory, config, "e") == []
    assert wb.note_poll_failure(memory, config, "e") == []
    assert _kinds(wb.note_poll_failure(memory, config, "e")) == ["UNREACHABLE"]


def test_failed_first_poll_does_not_consume_the_baseline() -> None:
    memory, config = wb.Memory(), _config(unreachable_polls=5)
    wb.note_poll_failure(memory, config, "e")
    runs = _runs()
    _results(runs)[0].update(status="FAILURE", endTime=LIVE_START + 120)
    # First *successful* poll is still the baseline: history is not news.
    assert _poll(_sensor(), runs, memory, config, LIVE_START + 130) == []


# -- status line --------------------------------------------------------

STATUS_RE = re.compile(
    r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ sensor=RUNNING gen=5 next_index=1055 "
    r"attempt=1 tick=SUCCESS halted=no median_s=\d+ last=2023-09-c11:STARTED:\d+s$"
)


def test_status_line_format_on_live_capture() -> None:
    line = wb.format_status(wb.parse_snapshot(_sensor(), _runs()), HEALTHY_NOW)
    assert STATUS_RE.match(line), line
    assert "\n" not in line


def test_status_line_values() -> None:
    line = wb.format_status(wb.parse_snapshot(_sensor(), _runs()), HEALTHY_NOW)
    assert line.startswith("2026-10-05T00:21:56Z ")
    # Median of the five finished live runs (275, 240, 205, 259, 228 s).
    assert "median_s=240 " in line
    assert line.endswith("last=2023-09-c11:STARTED:60s")


def test_status_line_shows_skip_reason_and_halt() -> None:
    line = wb.format_status(wb.parse_snapshot(_halted_sensor(), _runs()), HEALTHY_NOW)
    assert "tick=SKIPPED:HALTED: unit 2023-09-c11 failed 3 attempts" in line
    assert "halted=yes" in line
    sensor = _sensor()
    _newest_tick(sensor).update(status="SKIPPED", skipReason="run in flight")
    line = wb.format_status(wb.parse_snapshot(sensor, _runs()), HEALTHY_NOW)
    assert "tick=SKIPPED:run in flight" in line
    assert "halted=no" in line


def test_status_line_is_one_line_even_for_multiline_reason() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(status="SKIPPED", skipReason="HALTED: a\nb\nc")
    line = wb.format_status(wb.parse_snapshot(sensor, _runs()), HEALTHY_NOW)
    assert "\n" not in line


def test_status_line_with_no_runs_or_cursor() -> None:
    sensor, runs = _sensor(), _runs()
    _state(sensor)["typeSpecificData"]["lastCursor"] = None
    _results(runs).clear()
    line = wb.format_status(wb.parse_snapshot(sensor, runs), HEALTHY_NOW)
    assert "gen=n/a next_index=n/a attempt=n/a" in line
    assert "median_s=n/a last=none" in line


# -- fetch_snapshot -----------------------------------------------------


class FakePost:
    """Stands in for the HTTP POST: answers sensor and runs queries from fixtures."""

    def __init__(
        self,
        sensor: dict | None = None,
        runs: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.sensor = sensor if sensor is not None else _sensor()
        self.runs = runs if runs is not None else _runs()
        self.error = error
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls.append((url, payload, timeout))
        if self.error is not None:
            raise self.error
        return self.sensor if "sensorOrError" in payload["query"] else self.runs


def test_fetch_snapshot_issues_both_queries_with_a_15s_timeout() -> None:
    post = FakePost()
    snap = wb.fetch_snapshot("http://dagster.invalid:3000/graphql", post=post)
    assert snap.sensor_status == "RUNNING"
    assert len(post.calls) == 2
    assert {c[0] for c in post.calls} == {"http://dagster.invalid:3000/graphql"}
    assert {c[2] for c in post.calls} == {15}
    queries = [c[1]["query"] for c in post.calls]
    sensor_q = next(q for q in queries if "sensorOrError" in q)
    runs_q = next(q for q in queries if "runsOrError" in q)
    assert 'repositoryLocationName:"weather-forecast-audit"' in sensor_q
    assert 'repositoryName:"__repository__"' in sensor_q
    assert 'sensorName:"national_backfill_sensor"' in sensor_q
    assert "ticks(limit:5)" in sensor_q
    assert "runsOrError(limit:12)" in runs_q


def test_fetch_snapshot_uses_overridden_names() -> None:
    post = FakePost(sensor=_sensor(), runs=_runs())
    wb.fetch_snapshot(
        "http://x", post=post, location="loc", repository="repo", sensor_name="s"
    )
    sensor_q = next(
        c[1]["query"] for c in post.calls if "sensorOrError" in c[1]["query"]
    )
    assert 'repositoryLocationName:"loc"' in sensor_q
    assert 'repositoryName:"repo"' in sensor_q
    assert 'sensorName:"s"' in sensor_q


@pytest.mark.parametrize(
    "error",
    [
        OSError("connection refused"),
        TimeoutError("timed out"),
        ValueError("bad json"),
        json.JSONDecodeError("bad", "doc", 0),
    ],
)
def test_fetch_snapshot_wraps_transport_errors(error: Exception) -> None:
    with pytest.raises(wb.SnapshotError):
        wb.fetch_snapshot("http://x", post=FakePost(error=error))


def test_fetch_snapshot_graphql_errors_are_snapshot_errors() -> None:
    sensor = _sensor()
    sensor["errors"] = [{"message": "nope"}]
    with pytest.raises(wb.SnapshotError):
        wb.fetch_snapshot("http://x", post=FakePost(sensor=sensor))


# -- watch --------------------------------------------------------------


class Clock:
    def __init__(self, start: float = HEALTHY_NOW) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _scripted_fetch(snapshots: list) -> Callable[[], object]:
    """A fetch that returns / raises the next scripted item, then repeats the last."""
    items = list(snapshots)
    state = {"i": 0}

    def fetch() -> object:
        item = items[min(state["i"], len(items) - 1)]
        state["i"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    fetch.state = state
    return fetch


def _healthy_snapshot() -> object:
    return wb.parse_snapshot(_sensor(), _runs())


def _halted_snapshot() -> object:
    return wb.parse_snapshot(_halted_sensor(), _runs())


def test_watch_prints_nothing_while_healthy_then_events_and_status() -> None:
    clock, out = Clock(), io.StringIO()
    fetch = _scripted_fetch(
        [
            _healthy_snapshot(),
            _healthy_snapshot(),
            _healthy_snapshot(),
            _halted_snapshot(),
        ]
    )
    code = wb.watch(
        fetch=fetch,
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
    )
    assert code == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("EVENT HALT ")
    assert "HALTED: unit 2023-09-c11 failed 3 attempts" in lines[0]
    assert "sensor=RUNNING" in lines[1] and "halted=yes" in lines[1]
    assert clock.sleeps == [60, 60, 60]


def test_watch_is_silent_while_healthy(capsys: pytest.CaptureFixture[str]) -> None:
    clock, out = Clock(), io.StringIO()
    fetch = _scripted_fetch([_healthy_snapshot()])
    code = wb.watch(
        fetch=fetch,
        config=_config(slow_s=10**9),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
        max_s=600,
    )
    assert code == 0
    # Only the QUIET marker; no EVENT, no status line.
    assert out.getvalue().splitlines() == ["QUIET 600"]
    assert capsys.readouterr().out == ""


def test_watch_first_poll_event_exits_without_sleeping() -> None:
    clock, out = Clock(), io.StringIO()
    fetch = _scripted_fetch([_halted_snapshot()])
    assert (
        wb.watch(
            fetch=fetch,
            config=_config(),
            interval=60,
            sleep=clock.sleep,
            clock=clock,
            out=out,
        )
        == 0
    )
    assert clock.sleeps == []
    assert out.getvalue().startswith("EVENT HALT ")


def test_watch_prints_every_event_then_one_status_line() -> None:
    clock, out = Clock(), io.StringIO()
    sensor = _halted_sensor()
    _state(sensor)["status"] = "STOPPED"
    snap = wb.parse_snapshot(sensor, _runs())
    wb.watch(
        fetch=_scripted_fetch([snap]),
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
    )
    lines = out.getvalue().splitlines()
    assert [ln.split()[1] for ln in lines[:-1]] == ["STOPPED", "HALT"]
    assert all(ln.startswith("EVENT ") for ln in lines[:-1])
    assert lines[-1].startswith("20") and "sensor=" in lines[-1]


def test_watch_unreachable_after_n_consecutive_failures() -> None:
    clock, out = Clock(), io.StringIO()
    err = wb.SnapshotError("connection refused")
    fetch = _scripted_fetch([err])
    code = wb.watch(
        fetch=fetch,
        config=_config(unreachable_polls=3),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
    )
    assert code == 0
    assert fetch.state["i"] == 3
    assert clock.sleeps == [60, 60]
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("EVENT UNREACHABLE ")
    assert "connection refused" in lines[0]


def test_watch_unreachable_streak_resets_on_success() -> None:
    clock, out = Clock(), io.StringIO()
    err = wb.SnapshotError("down")
    healthy = _healthy_snapshot()
    fetch = _scripted_fetch([err, err, healthy, err, err, healthy, _halted_snapshot()])
    wb.watch(
        fetch=fetch,
        config=_config(unreachable_polls=3),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
    )
    assert out.getvalue().startswith("EVENT HALT ")
    assert "UNREACHABLE" not in out.getvalue()


def test_watch_max_s_zero_never_quits_on_time() -> None:
    clock, out = Clock(), io.StringIO()
    snaps = [_healthy_snapshot()] * 500 + [_halted_snapshot()]
    wb.watch(
        fetch=_scripted_fetch(snaps),
        config=_config(stall_s=10**9, slow_s=10**9),
        interval=3600,
        sleep=clock.sleep,
        clock=clock,
        out=out,
        max_s=0,
    )
    assert "QUIET" not in out.getvalue()
    assert out.getvalue().startswith("EVENT HALT ")


def test_watch_max_s_does_not_pre_empt_an_event() -> None:
    clock, out = Clock(), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([_halted_snapshot()]),
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        out=out,
        max_s=1,
    )
    assert "QUIET" not in out.getvalue()
    assert out.getvalue().startswith("EVENT HALT ")


# -- main (end to end through a fake post) ------------------------------


def _main(
    argv: list[str], post: Callable, clock: Clock | None = None
) -> tuple[int, str, str]:
    clock = clock or Clock()
    out, err = io.StringIO(), io.StringIO()
    code = wb.main(argv, post=post, clock=clock, sleep=clock.sleep, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_main_without_url_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    post = FakePost()
    code, out, err = _main(["status"], post)
    assert code == 2
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert "WFA_DAGSTER_URL" in err and "--url" in err
    assert post.calls == []


def test_main_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFA_DAGSTER_URL", "http://env.invalid/graphql")
    post = FakePost()
    code, out, _ = _main(["status"], post)
    assert code == 0
    assert {c[0] for c in post.calls} == {"http://env.invalid/graphql"}
    assert out.count("\n") == 1


def test_main_url_flag_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFA_DAGSTER_URL", "http://env.invalid/graphql")
    post = FakePost()
    _main(["status", "--url", "http://flag.invalid/graphql"], post)
    assert {c[0] for c in post.calls} == {"http://flag.invalid/graphql"}


def test_main_status_prints_one_status_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock()
    code, out, err = _main(["status", "--url", "http://x"], FakePost(), clock)
    assert code == 0
    assert err == ""
    assert STATUS_RE.match(out.strip()), out
    assert out.count("\n") == 1


def test_main_status_unreachable_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    code, out, _ = _main(
        ["status", "--url", "http://x"], FakePost(error=OSError("refused"))
    )
    assert code == 3
    assert out.startswith("UNREACHABLE ")
    assert "refused" in out
    assert out.count("\n") == 1


def test_main_wait_exits_0_with_event_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    post = FakePost(sensor=_halted_sensor())
    code, out, _ = _main(["wait", "--url", "http://x", "--interval", "5"], post)
    lines = out.splitlines()
    assert code == 0
    assert lines[0].startswith("EVENT HALT ")
    assert lines[1].startswith("20") and "halted=yes" in lines[1]


def test_main_wait_polls_silently_until_the_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock()
    post = FakePost()
    polls = {"n": 0}
    inner = post.__call__

    def flipping(url: str, payload: dict, timeout: float) -> dict:
        if "sensorOrError" in payload["query"]:
            polls["n"] += 1
            if polls["n"] == 4:
                post.sensor = _halted_sensor()
        return inner(url, payload, timeout)

    argv = ["wait", "--url", "http://x", "--interval", "30"]
    code, out, _ = _main(argv, flipping, clock)
    assert code == 0
    assert polls["n"] == 4
    assert clock.sleeps == [30, 30, 30]
    assert out.splitlines()[0].startswith("EVENT HALT ")


def test_main_wait_ignore_failed_attempts_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)

    def run(flag: list[str]) -> str:
        clock = Clock()
        post = FakePost()
        polls = {"n": 0}
        inner = post.__call__

        def flipping(url: str, payload: dict, timeout: float) -> dict:
            if "runsOrError" in payload["query"]:
                polls["n"] += 1
                if polls["n"] == 2:
                    _results(post.runs)[0].update(
                        status="FAILURE", endTime=LIVE_START + 100
                    )
            return inner(url, payload, timeout)

        argv = ["wait", "--url", "http://x", "--interval", "30", "--max-s", "300"]
        argv += flag
        _, out, _ = _main(argv, flipping, clock)
        return out

    assert "EVENT FAILED_ATTEMPT" in run([])
    quiet = run(["--ignore-failed-attempts"])
    assert "FAILED_ATTEMPT" not in quiet
    assert quiet.splitlines() == ["QUIET 300"]


def test_main_wait_slow_and_stall_flags_are_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(HEALTHY_NOW)
    code, out, _ = _main(
        [
            "wait",
            "--url",
            "http://x",
            "--interval",
            "10",
            "--slow-s",
            "100",
            "--stall-s",
            "100000",
            "--max-s",
            "1000",
        ],
        FakePost(),
        clock,
    )
    assert code == 0
    assert out.splitlines()[0].startswith("EVENT SLOW ")


def test_main_wait_unreachable_polls_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock()
    code, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--unreachable-polls", "2"],
        FakePost(error=OSError("refused")),
        clock,
    )
    assert code == 0
    assert out.startswith("EVENT UNREACHABLE ")
    assert clock.sleeps == [10]


def test_main_wait_quiet_marker_on_max_s(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock()
    code, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "60", "--max-s", "180"],
        FakePost(),
        clock,
    )
    assert code == 0
    assert out == "QUIET 180\n"


def test_main_name_overrides_reach_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    post = FakePost()
    _main(
        [
            "status",
            "--url",
            "http://x",
            "--sensor",
            "s2",
            "--location",
            "l2",
            "--repository",
            "r2",
        ],
        post,
    )
    q = next(c[1]["query"] for c in post.calls if "sensorOrError" in c[1]["query"])
    assert 'sensorName:"s2"' in q and 'repositoryLocationName:"l2"' in q
    assert 'repositoryName:"r2"' in q
