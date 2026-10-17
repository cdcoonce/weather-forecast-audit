"""Tests for `scripts/watch_backfill.py`, the exit-on-event backfill watcher.

The script is stdlib-only and lives in `scripts/` (not a package), so it is
loaded by path. Every response here is built by loading the two real captures
in `tests/fixtures/backfill_watcher/` (raw GraphQL from the live server) and
mutating copies; no response shape is invented. Nothing touches the network:
`post`, `fetch`, `sleep`, `clock`, and `out` are all injected.
"""

import ast
import copy
import http.client
import importlib.util
import io
import json
import os
import re
import sys
import unicodedata
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


def _poll(
    sensor: dict,
    runs: dict,
    memory: object,
    config: object,
    now: float,
    mono: float | None = None,
) -> list:
    """Evaluate one poll; the monotonic clock follows `now` unless given."""
    snapshot = wb.parse_snapshot(sensor, runs)
    return wb.evaluate(
        snapshot, memory, config, now, mono=now if mono is None else mono
    )


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
    """Fake wall clock (`__call__`) and monotonic clock (`mono`).

    `sleep` advances both. It raises after `max_sleeps` calls so a regressed
    exit guard turns a hang into a red test instead of an endless loop.
    """

    def __init__(self, start: float = HEALTHY_NOW, max_sleeps: int = 1000) -> None:
        self.now = start
        self.mono_now = 5000.0
        self.max_sleeps = max_sleeps
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def mono(self) -> float:
        return self.mono_now

    def step_wall(self, seconds: float) -> None:
        """A wall-clock step (NTP, suspend/resume) the monotonic clock ignores."""
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        if len(self.sleeps) >= self.max_sleeps:
            msg = "watch looped past max_s"
            raise AssertionError(msg)
        self.sleeps.append(seconds)
        self.now += seconds
        self.mono_now += seconds


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
        mono=clock.mono,
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
        mono=clock.mono,
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
            mono=clock.mono,
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
        mono=clock.mono,
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
        mono=clock.mono,
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
        mono=clock.mono,
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
        mono=clock.mono,
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
        mono=clock.mono,
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
    code = wb.main(
        argv,
        post=post,
        clock=clock,
        mono=clock.mono,
        sleep=clock.sleep,
        out=out,
        err=err,
    )
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


# -- main: threshold flags are honoured (set BELOW their defaults) ------


def test_main_slow_s_flag_sets_the_slow_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(HEALTHY_NOW, max_sleeps=100)
    code, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--slow-s", "100"]
        + ["--stall-s", "100000", "--max-s", "1000"],
        FakePost(),
        clock,
    )
    assert code == 0
    assert out.splitlines()[0].startswith("EVENT SLOW ")
    assert "duration=100s" in out.splitlines()[0]
    # The run was 60 s old at the first poll; 100 s is 4 intervals later.
    assert clock.sleeps == [10, 10, 10, 10]


def test_main_stall_s_flag_sets_the_stall_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(HEALTHY_NOW, max_sleeps=100)
    code, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--stall-s", "100"]
        + ["--slow-s", "1000000000", "--max-s", "5000"],
        FakePost(),
        clock,
    )
    assert code == 0
    assert out.splitlines()[0].startswith("EVENT STALLED ")
    assert "unchanged for 100s" in out.splitlines()[0]
    assert clock.sleeps == [10] * 10


def test_main_defaults_are_450_slow_and_3h_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(HEALTHY_NOW, max_sleeps=100)
    _, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--max-s", "1000"],
        FakePost(),
        clock,
    )
    assert out.splitlines()[0].startswith("EVENT SLOW ")
    assert "duration=450s" in out.splitlines()[0]


# -- --sensor decides which runs are kept ------------------------------


def _runs_for_other_sensor() -> dict:
    runs = _runs()
    runs["data"]["runsOrError"]["results"].insert(
        0,
        _run(
            runs,
            run_id="s2-run",
            status="SUCCESS",
            start=LIVE_START + 10,
            end=LIVE_START + 20,
            unit="2024-01-s2",
            sensor_name="s2",
        ),
    )
    return runs


def test_parse_snapshot_sensor_name_parameter_chooses_the_runs() -> None:
    runs = _runs_for_other_sensor()
    mine = wb.parse_snapshot(_sensor(), runs, sensor_name="s2")
    assert [r.run_id for r in mine.runs] == ["s2-run"]
    default = wb.parse_snapshot(_sensor(), runs)
    assert "s2-run" not in [r.run_id for r in default.runs]


def test_fetch_snapshot_sensor_name_chooses_the_runs() -> None:
    post = FakePost(runs=_runs_for_other_sensor())
    snap = wb.fetch_snapshot("http://x", post=post, sensor_name="s2")
    assert [r.run_id for r in snap.runs] == ["s2-run"]


def test_main_sensor_flag_changes_which_runs_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    _, out, _ = _main(
        ["status", "--url", "http://x", "--sensor", "s2"],
        FakePost(runs=_runs_for_other_sensor()),
    )
    assert "last=2024-01-s2:SUCCESS:10s" in out
    assert "median_s=10 " in out
    _, out, _ = _main(
        ["status", "--url", "http://x"], FakePost(runs=_runs_for_other_sensor())
    )
    assert "last=2023-09-c11:STARTED" in out


# -- STALLED: re-arm, idle seconds -------------------------------------


def test_stalled_rearms_after_the_index_advances_again() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=1000, slow_s=10**9)
    t0 = HEALTHY_NOW
    assert _poll(sensor, runs, memory, config, t0) == []
    assert _kinds(_poll(sensor, runs, memory, config, t0 + 1000)) == ["STALLED"]
    assert _poll(sensor, runs, memory, config, t0 + 1500) == []
    _set_cursor(sensor, next_index=1056)
    assert _poll(sensor, runs, memory, config, t0 + 1600) == []
    assert _poll(sensor, runs, memory, config, t0 + 2599) == []
    events = _poll(sensor, runs, memory, config, t0 + 2600)
    assert _kinds(events) == ["STALLED"]
    assert "next_index=1056" in events[0].detail


def test_stalled_detail_reports_the_actual_idle_seconds() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=1000, slow_s=10**9)
    assert _poll(sensor, runs, memory, config, HEALTHY_NOW) == []
    events = _poll(sensor, runs, memory, config, HEALTHY_NOW + 1234)
    assert _kinds(events) == ["STALLED"]
    assert "unchanged for 1234s" in events[0].detail


# -- HALT: staleness, null cursors, prefix ------------------------------


def test_status_line_stale_halt_prints_halted_no() -> None:
    sensor = _halted_sensor()
    _set_cursor(sensor, generation=6, attempt=1)
    line = wb.format_status(wb.parse_snapshot(sensor, _runs()), HEALTHY_NOW)
    assert "tick=SKIPPED:HALTED: unit 2023-09-c11 failed 3 attempts" in line
    assert "halted=no" in line
    assert "gen=6" in line


@pytest.mark.parametrize("raw", [None, "{not json", '{"state": 3}', "[]"])
def test_halt_does_not_fire_when_tick_and_current_cursor_are_both_unusable(
    raw: str | None,
) -> None:
    sensor = _halted_sensor()
    _state(sensor)["typeSpecificData"]["lastCursor"] = raw
    _newest_tick(sensor)["cursor"] = raw
    snap = wb.parse_snapshot(sensor, _runs())
    assert wb.evaluate(snap, wb.Memory(), _config(), HEALTHY_NOW, mono=0.0) == []
    assert "halted=no" in wb.format_status(snap, HEALTHY_NOW)


def test_halt_does_not_fire_when_only_the_tick_cursor_is_null() -> None:
    sensor = _halted_sensor()
    _newest_tick(sensor)["cursor"] = None
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


@pytest.mark.parametrize("reason", ["HALTING", "HALTED", "halted: x", "Halted: x"])
def test_halt_needs_the_exact_halted_colon_prefix(reason: str) -> None:
    sensor = _halted_sensor(reason)
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []
    line = wb.format_status(wb.parse_snapshot(sensor, _runs()), HEALTHY_NOW)
    assert "halted=no" in line


# -- SENSOR_ERROR / UNREACHABLE detail text -----------------------------


def test_sensor_error_multiline_message_collapses_to_one_line() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(
        status="FAILURE", error={"message": "Traceback:\n  line 1\r\n  line 2\t\n"}
    )
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert events[0].detail == "Traceback: line 1 line 2"


def test_sensor_error_null_error_gives_the_fallback_text() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(status="FAILURE", error=None)
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert events[0].detail == "no error message"


def test_unreachable_default_threshold_is_ten_polls() -> None:
    assert wb.Config().unreachable_polls == 10
    memory, config = wb.Memory(), wb.Config()
    for _ in range(9):
        assert wb.note_poll_failure(memory, config, "down") == []
    assert _kinds(wb.note_poll_failure(memory, config, "down")) == ["UNREACHABLE"]


def test_main_wait_default_unreachable_threshold_is_ten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(max_sleeps=50)
    _, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10"],
        FakePost(error=OSError("refused")),
        clock,
    )
    assert out.startswith("EVENT UNREACHABLE ")
    assert clock.sleeps == [10] * 9


def test_unreachable_detail_is_truncated_and_one_line() -> None:
    config = _config(unreachable_polls=1)
    long = wb.note_poll_failure(wb.Memory(), config, "x" * 500)
    assert long[0].detail == "x" * 200
    multi = wb.note_poll_failure(wb.Memory(), config, "a\nb\n\nEVENT COMPLETE c")
    assert multi[0].detail == "a b EVENT COMPLETE c"


# -- status line: median window and rounding ----------------------------


def _finished_runs(durations_newest_first: list[float]) -> dict:
    """Runs of this sensor, one finished run per duration, newest first."""
    runs = _runs()
    built = [
        _run(
            runs,
            run_id=f"r{i}",
            status="SUCCESS",
            start=100_000.0 - i * 1000,
            end=100_000.0 - i * 1000 + duration,
        )
        for i, duration in enumerate(durations_newest_first)
    ]
    _results(runs)[:] = built
    return runs


def test_status_median_covers_the_ten_most_recent_finished_runs() -> None:
    # Newest ten: median is the mean of 241 and 242, rounded to 242 (not 241).
    # A window of 3 would say 110, of 11 say 241, of 12 say 186 (the two 10 s runs).
    newest = [100, 110, 120, 130, 241, 242, 300, 310, 320, 330]
    runs = _finished_runs(newest + [10, 10])
    line = wb.format_status(wb.parse_snapshot(_sensor(), runs), 200_000.0)
    assert "median_s=242 " in line


def test_status_median_rounds_to_nearest_not_down() -> None:
    runs = _finished_runs([240, 241])
    line = wb.format_status(wb.parse_snapshot(_sensor(), runs), 200_000.0)
    assert "median_s=240 " in line or "median_s=241 " in line
    runs = _finished_runs([241, 242])
    line = wb.format_status(wb.parse_snapshot(_sensor(), runs), 200_000.0)
    assert "median_s=242 " in line  # 241.5 -> 242 (round half to even)


def test_status_last_duration_rounds_to_nearest_second() -> None:
    runs = _finished_runs([0.6])
    line = wb.format_status(wb.parse_snapshot(_sensor(), runs), 200_000.0)
    assert line.endswith("last=2023-09-c99:SUCCESS:1s")


# -- COMPLETE together with STOPPED -------------------------------------


def test_complete_fires_even_while_the_sensor_is_stopped() -> None:
    sensor = _sensor()
    _state(sensor)["status"] = "STOPPED"
    _newest_tick(sensor).update(
        status="SKIPPED", skipReason="national backfill complete"
    )
    events = _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW)
    assert sorted(_kinds(events)) == ["COMPLETE", "STOPPED"]


# -- watch: boundary and bounded loops ----------------------------------


def test_watch_quiet_exit_at_exactly_max_s_does_not_loop() -> None:
    clock, out = Clock(max_sleeps=3), io.StringIO()
    code = wb.watch(
        fetch=_scripted_fetch([_healthy_snapshot()]),
        config=_config(slow_s=10**9, stall_s=10**9),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
        max_s=180,
    )
    assert code == 0
    assert out.getvalue() == "QUIET 180\n"
    assert clock.sleeps == [60, 60, 60]


def test_watch_final_sleep_is_clipped_to_the_remaining_max_s() -> None:
    clock, out = Clock(max_sleeps=3), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([_healthy_snapshot()]),
        config=_config(slow_s=10**9, stall_s=10**9),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
        max_s=150,
    )
    assert clock.sleeps == [60, 60, 30]
    assert out.getvalue() == "QUIET 150\n"


def test_a_wall_clock_step_cannot_fake_a_stall_or_a_quiet_exit() -> None:
    clock, out = Clock(max_sleeps=5), io.StringIO()
    snapshot = _healthy_snapshot()
    polls = {"n": 0}

    def fetch() -> object:
        polls["n"] += 1
        if polls["n"] == 2:
            clock.step_wall(10 * 3600)  # laptop sleep / clock step
        if polls["n"] == 4:
            return _halted_snapshot()
        return snapshot

    wb.watch(
        fetch=fetch,
        config=_config(stall_s=1000, slow_s=10**12),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
        max_s=100_000,
    )
    assert out.getvalue().startswith("EVENT HALT ")
    assert "STALLED" not in out.getvalue()
    assert "QUIET" not in out.getvalue()


def test_a_real_monotonic_idle_period_still_stalls_in_watch() -> None:
    clock, out = Clock(max_sleeps=100), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([_healthy_snapshot()]),
        config=_config(stall_s=300, slow_s=10**12),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
    )
    assert out.getvalue().startswith("EVENT STALLED ")
    assert "unchanged for 300s" in out.getvalue()
    assert clock.sleeps == [60] * 5


def test_slow_uses_the_wall_clock_against_the_server_start_time() -> None:
    # The monotonic clock is an unrelated epoch: SLOW must not read it.
    events = _poll(
        _sensor(), _runs(), wb.Memory(), _config(), LIVE_START + 450, mono=12.0
    )
    assert _kinds(events) == ["SLOW"]


def test_evaluate_stall_uses_the_monotonic_clock_not_the_wall_clock() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    config = _config(stall_s=1000, slow_s=10**12)
    assert _poll(sensor, runs, memory, config, HEALTHY_NOW, mono=50.0) == []
    # Wall clock leaps a day; monotonic moved 10 s: not a stall.
    assert _poll(sensor, runs, memory, config, HEALTHY_NOW + 86400, mono=60.0) == []
    events = _poll(sensor, runs, memory, config, HEALTHY_NOW + 86401, mono=1050.0)
    assert _kinds(events) == ["STALLED"]


# -- transport and shape hardening --------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        http.client.IncompleteRead(b"ab"),
        http.client.BadStatusLine("x"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_fetch_snapshot_wraps_http_client_exceptions(error: Exception) -> None:
    with pytest.raises(wb.SnapshotError):
        wb.fetch_snapshot("http://x", post=FakePost(error=error))


def test_default_post_body_read_errors_are_snapshot_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def read(self) -> bytes:
            raise http.client.IncompleteRead(b"ab")

    seen: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> Response:
        seen.append(request)
        return Response()

    monkeypatch.setattr(wb.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(wb.SnapshotError):
        wb.fetch_snapshot("http://dagster.invalid/graphql", post=wb.urllib_post)
    assert seen


def test_main_wait_counts_http_client_errors_as_failed_polls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(max_sleeps=10)
    code, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--unreachable-polls", "2"],
        FakePost(error=http.client.IncompleteRead(b"ab")),
        clock,
    )
    assert code == 0
    assert out.startswith("EVENT UNREACHABLE ")
    assert clock.sleeps == [10]


def test_main_status_http_client_error_is_unreachable_exit_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    code, out, _ = _main(
        ["status", "--url", "http://x"],
        FakePost(error=http.client.BadStatusLine("x")),
    )
    assert code == 3
    assert out.startswith("UNREACHABLE ")
    assert out.count("\n") == 1


@pytest.mark.parametrize(
    "errors",
    [{"message": "x"}, {}, "boom", [{"message": "x"}], ["plain"], [None]],
)
def test_graphql_errors_in_any_shape_are_a_failed_poll(errors: object) -> None:
    sensor = _sensor()
    sensor["errors"] = errors
    if not errors:
        assert wb.parse_snapshot(sensor, _runs()).sensor_status == "RUNNING"
        return
    with pytest.raises(wb.SnapshotError):
        wb.parse_snapshot(sensor, _runs())


def test_graphql_errors_dict_message_is_reported() -> None:
    sensor = _sensor()
    sensor["errors"] = {"message": "bad\nthing"}
    with pytest.raises(wb.SnapshotError, match="bad thing"):
        wb.parse_snapshot(sensor, _runs())


def test_string_epoch_times_are_coerced_to_floats() -> None:
    runs = _runs()
    _results(runs)[1].update(startTime="1791159656.3", endTime="1791159900")
    snap = wb.parse_snapshot(_sensor(), runs)
    assert snap.runs[1].start_time == 1791159656.3
    assert snap.runs[1].end_time == 1791159900.0
    assert snap.runs[1].duration(0) == pytest.approx(243.7)


@pytest.mark.parametrize("bad", ["soon", "", [], {}, "nan", "inf", True])
@pytest.mark.parametrize("field_name", ["startTime", "endTime"])
def test_unusable_epoch_times_are_snapshot_errors(bad: object, field_name: str) -> None:
    runs = _runs()
    _results(runs)[1][field_name] = bad
    with pytest.raises(wb.SnapshotError):
        wb.parse_snapshot(_sensor(), runs)


def test_non_string_skip_reason_does_not_crash() -> None:
    sensor = _sensor()
    _newest_tick(sensor).update(status="SKIPPED", skipReason=42)
    assert _poll(sensor, _runs(), wb.Memory(), _config(), HEALTHY_NOW) == []


@pytest.mark.parametrize("specific", ["x", 7, [1]])
def test_non_dict_type_specific_data_is_a_failed_poll_not_a_traceback(
    specific: object,
) -> None:
    sensor = _sensor()
    _state(sensor)["typeSpecificData"] = specific
    with pytest.raises(wb.SnapshotError):
        wb.parse_snapshot(sensor, _runs())


# -- a run that has not started (null startTime) ------------------------


def _queued_run(
    runs: dict, *, end: float | None = None, status: str = "QUEUED"
) -> dict:
    run = _run(runs, run_id="queued", status=status, start=0.0, end=end)
    run["startTime"] = None
    return run


def test_run_without_start_time_is_not_slow_and_does_not_crash() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    _results(runs).insert(0, _queued_run(runs))
    config = _config(slow_s=1)
    events = _poll(sensor, runs, memory, config, HEALTHY_NOW)
    assert [e.detail for e in events if "queued" in e.detail] == []
    # Only the already-running live run (60 s old, slow_s=1) can be SLOW.
    assert all("2023-09-c99" not in e.detail for e in events)
    line = wb.format_status(wb.parse_snapshot(sensor, runs), HEALTHY_NOW)
    assert "\n" not in line


def test_null_start_run_that_later_fails_still_reports_once() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    _results(runs).insert(0, _queued_run(runs))
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    _results(runs)[0].update(status="CANCELED", endTime=HEALTHY_NOW + 5)
    events = _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 10)
    assert _kinds(events) == ["FAILED_ATTEMPT"]
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 20) == []


def test_baseline_finished_null_start_run_is_history() -> None:
    sensor, runs, memory = _sensor(), _runs(), wb.Memory()
    _results(runs).insert(0, _queued_run(runs, end=LIVE_START, status="CANCELED"))
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW) == []
    assert _poll(sensor, runs, memory, _config(), HEALTHY_NOW + 60) == []


# -- one event per line: server-derived text is sanitised ---------------

HOSTILE = "u1\nEVENT COMPLETE fake"


def _assert_event_lines(text: str, expected_kinds: list[str], total: int) -> None:
    """Exactly `total` lines; the `EVENT ` lines are exactly the real ones;
    no control character survives anywhere."""
    assert text.endswith("\n")
    lines = text.splitlines()  # also splits on NEL, U+2028 and U+2029
    for char in "".join(lines):
        assert unicodedata.category(char) not in {"Cc", "Cf", "Zl", "Zp"}, repr(char)
    assert len(lines) == total, lines
    assert [ln.split()[1] for ln in lines if ln.startswith("EVENT ")] == expected_kinds


def _hostile_runs() -> dict:
    runs = _runs()
    run = _run(
        runs,
        run_id="evil",
        status="FAILURE",
        start=LIVE_START - 2000,
        end=LIVE_START - 1000,
        unit=HOSTILE,
        attempt="2\x1b[31m\nEVENT STOPPED y",
    )
    _results(runs).insert(1, run)
    return runs


def test_hostile_run_tags_cannot_forge_event_lines_via_watch() -> None:
    clock, out = Clock(), io.StringIO()
    baseline = wb.parse_snapshot(_sensor(), _runs())
    hostile = wb.parse_snapshot(_sensor(), _hostile_runs())
    wb.watch(
        fetch=_scripted_fetch([baseline, hostile]),
        config=_config(slow_s=900),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
    )
    # SLOW (1000 s) and FAILED_ATTEMPT for the hostile run, plus the status line.
    _assert_event_lines(out.getvalue(), ["SLOW", "FAILED_ATTEMPT"], total=3)
    assert "u1 EVENT COMPLETE fake" in out.getvalue()


def test_hostile_sensor_status_cannot_forge_event_lines() -> None:
    sensor = _sensor()
    _state(sensor)["status"] = "STOPPED\nEVENT COMPLETE x"
    clock, out = Clock(), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([wb.parse_snapshot(sensor, _runs())]),
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
    )
    _assert_event_lines(out.getvalue(), ["STOPPED"], total=2)
    assert "sensor=STOPPED EVENT COMPLETE x" in out.getvalue()


def test_hostile_cursor_fields_cannot_forge_event_lines() -> None:
    sensor = _halted_sensor()
    _set_cursor(sensor, generation="5\nEVENT COMPLETE g", next_index="9\u2028EVENT X")
    _newest_tick(sensor)["cursor"] = _state(sensor)["typeSpecificData"]["lastCursor"]
    clock, out = Clock(), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([wb.parse_snapshot(sensor, _runs())]),
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
    )
    _assert_event_lines(out.getvalue(), ["HALT"], total=2)


def test_control_characters_in_a_skip_reason_are_stripped() -> None:
    reason = "HALTED: bad\x1b[2J\x00\x7f\x85thing\u2028EVENT COMPLETE z"
    sensor = _halted_sensor(reason)
    clock, out = Clock(), io.StringIO()
    wb.watch(
        fetch=_scripted_fetch([wb.parse_snapshot(sensor, _runs())]),
        config=_config(),
        interval=60,
        sleep=clock.sleep,
        clock=clock,
        mono=clock.mono,
        out=out,
    )
    _assert_event_lines(out.getvalue(), ["HALT"], total=2)
    assert "HALTED: bad[2J thing EVENT COMPLETE z" in out.getvalue()


def test_a_huge_halted_reason_is_truncated_in_the_event_and_status_line() -> None:
    sensor = _halted_sensor("HALTED: " + "y" * 5000)
    snap = wb.parse_snapshot(sensor, _runs())
    events = wb.evaluate(snap, wb.Memory(), _config(), HEALTHY_NOW, mono=0.0)
    assert _kinds(events) == ["HALT"]
    assert "y" * 150 in events[0].detail
    assert len(events[0].detail) < 260
    line = wb.format_status(snap, HEALTHY_NOW)
    assert "y" * 150 in line
    assert len(line) < 600


def test_hostile_values_in_the_status_command_stay_on_one_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    sensor = _halted_sensor("HALTED: a\nEVENT COMPLETE b")
    _state(sensor)["status"] = "RUNNING\nEVENT STOPPED q"
    _, out, _ = _main(
        ["status", "--url", "http://x"], FakePost(sensor=sensor, runs=_hostile_runs())
    )
    _assert_event_lines(out, [], total=1)


def test_unreachable_status_text_is_one_line_and_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    code, out, _ = _main(
        ["status", "--url", "http://x"],
        FakePost(error=OSError("a\nEVENT COMPLETE b" + "z" * 5000)),
    )
    assert code == 3
    _assert_event_lines(out, [], total=1)
    assert len(out) < 300


def test_unreachable_event_cannot_forge_event_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    clock = Clock(max_sleeps=5)
    _, out, _ = _main(
        ["wait", "--url", "http://x", "--interval", "10", "--unreachable-polls", "1"],
        FakePost(error=OSError("down\nEVENT COMPLETE fake")),
        clock,
    )
    _assert_event_lines(out, ["UNREACHABLE"], total=1)


# -- config validation --------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ["--unreachable-polls", "0"],
        ["--unreachable-polls", "-3"],
        ["--unreachable-polls", "1.5"],
        ["--interval", "0"],
        ["--interval", "-1"],
        ["--interval", "nan"],
        ["--interval", "soon"],
        ["--slow-s", "0"],
        ["--slow-s", "-5"],
        ["--stall-s", "0"],
        ["--stall-s", "-5"],
        ["--max-s", "-1"],
        ["--max-s", "nan"],
    ],
)
@pytest.mark.parametrize("mode", ["wait", "status"])
def test_invalid_numeric_flags_exit_2_before_any_poll(
    flags: list[str],
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    post = FakePost()
    with pytest.raises(SystemExit) as excinfo:
        _main([mode, "--url", "http://x", *flags], post)
    assert excinfo.value.code == 2
    assert post.calls == []
    assert flags[0] in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [
        ["--unreachable-polls", "1"],
        ["--interval", "0.5"],
        ["--slow-s", "0.5"],
        ["--stall-s", "0.5"],
        ["--max-s", "0"],
    ],
)
def test_boundary_valid_numeric_flags_are_accepted(
    flags: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    args = wb._build_parser().parse_args(["wait", "--url", "http://x", *flags])
    assert args.url == "http://x"


def test_hostile_newest_run_stays_on_one_line_in_the_status_last_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)
    runs = _runs()
    run = _run(
        runs,
        run_id="evil",
        status="STARTED\nEVENT STOPPED q",
        start=LIVE_START + 30,
        end=None,
        unit="u\nEVENT COMPLETE z",
    )
    _results(runs).insert(0, run)
    _, out, _ = _main(["status", "--url", "http://x"], FakePost(runs=runs))
    _assert_event_lines(out, [], total=1)
    assert "last=u EVENT COMPLETE z:STARTED EVENT STOPPED q:" in out


def test_status_command_sanitises_a_snapshot_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DAGSTER_URL", raising=False)

    def failing(*_args: object, **_kwargs: object) -> object:
        raise wb.SnapshotError("a\nEVENT COMPLETE b" + "z" * 5000)

    monkeypatch.setattr(wb, "fetch_snapshot", failing)
    code, out, _ = _main(["status", "--url", "http://x"], FakePost())
    assert code == 3
    _assert_event_lines(out, [], total=1)
    assert len(out) < 300
