"""national_backfill_sensor (issue #11): translating decide()'s output into
Dagster calls -- every substantive backfill rule is already covered, with no
running instance, by tests/unit/test_backfill.py. These tests only check
that the sensor reads the world (an ephemeral instance's runs, the injected
clock) into decide()'s inputs correctly, and turns a Submit/SubmitTransform
into the right RunRequest.
"""

import json
from datetime import datetime

import pytest
from dagster import DagsterInstance, DagsterRunStatus, build_sensor_context, job, op
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
)

from weather_forecast_audit import sensors as sensors_module
from weather_forecast_audit.jobs import ingest_job, transform_job
from weather_forecast_audit.sensors import (
    ATTEMPT_TAG,
    GENERATION_TAG,
    TRANSFORM_UNIT_ID,
    UNIT_TAG,
    national_backfill_sensor,
)

pytestmark = [pytest.mark.dagster, pytest.mark.io]

# Sorted: KORD, KPHX. Both fit in one BACKFILL_CHUNK_SIZE=30 chunk, so every
# calendar month in the plan is exactly one unit.
STATIONS = ["KPHX", "KORD"]


@op
def _noop() -> None:
    pass


@job
def _fake_run_job() -> None:
    """A resource-free stand-in for `ingest_job`/`transform_job` when
    fabricating run records: `_slot_busy`/`_run_status` only read a run's
    status and tags, never its job, and the real jobs' resources need
    `WFA_DUCKDB_PATH` to even validate a run config at creation time."""
    _noop()


def _record_run(
    instance: DagsterInstance, tags: dict, status: DagsterRunStatus
) -> None:
    instance.create_run_for_job(_fake_run_job, tags=tags, status=status)


@pytest.fixture(autouse=True)
def _fixed_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sensors_module, "_all_station_icaos", lambda: STATIONS)


def _set_now(monkeypatch: pytest.MonkeyPatch, iso: str) -> None:
    fixed = datetime.fromisoformat(iso)
    monkeypatch.setattr(sensors_module, "_now", lambda: fixed)


def _cursor_state(cursor: str | None) -> dict:
    assert cursor is not None
    return json.loads(cursor)


def test_first_evaluation_submits_unit_zero_with_partition_range_and_stations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ARCHIVE_START is 2020-09-29 (regimes.load_archive_start()); "yesterday"
    # here is the same day, so the whole plan is that single day.
    _set_now(monkeypatch, "2020-09-30T12:00:00")  # 05:00 local -- outside blackout

    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(instance=instance, cursor=None)
        result = national_backfill_sensor(context)

        assert result.skip_reason is None
        [run_request] = result.run_requests
        assert run_request.job_name == ingest_job.name
        assert run_request.run_key == "2020-09-c00-a1-g0"
        assert run_request.tags[UNIT_TAG] == "2020-09-c00"
        assert run_request.tags[ATTEMPT_TAG] == "1"
        assert run_request.tags[GENERATION_TAG] == "0"
        assert run_request.tags[ASSET_PARTITION_RANGE_START_TAG] == "2020-09-29"
        assert run_request.tags[ASSET_PARTITION_RANGE_END_TAG] == "2020-09-29"
        assert run_request.tags["dagster/max_runtime"] == "600"
        assert run_request.run_config == {
            "resources": {"stations": {"config": {"only": ["KORD", "KPHX"]}}}
        }

        envelope = _cursor_state(context.cursor)
        assert envelope["plan_end"] == "2020-09-29"
        state = envelope["state"]
        assert state["next_index"] == 0
        assert state["attempt"] == 1
        assert state["last_run_id"] == "2020-09-c00-a1-g0"
        assert state["transform_requested"] is False
        # The not-found-timeout clock: stamped by `decide` itself, not the
        # sensor, so a later tick can tell how long this run has been
        # missing (see backfill.decide's not-found-timeout rule).
        assert state["submitted_at"] == "2020-09-30T12:00:00"


def test_a_run_elsewhere_on_the_shared_slot_blocks_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_now(monkeypatch, "2020-09-30T12:00:00")

    with DagsterInstance.ephemeral() as instance:
        # Some other tenant's run, unrelated tags -- slot_busy is global.
        _record_run(instance, tags={"tenant": "oura"}, status=DagsterRunStatus.STARTED)

        context = build_sensor_context(instance=instance, cursor=None)
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message == "slot busy"
        envelope = _cursor_state(context.cursor)
        assert envelope["state"]["next_index"] == 0
        assert envelope["state"]["last_run_id"] is None


def test_blackout_window_blocks_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    # 2026-09-27 13:00 UTC == 06:00 America/Phoenix -- inside 05:45-07:15.
    _set_now(monkeypatch, "2026-09-27T13:00:00")

    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(instance=instance, cursor=None)
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message == "blackout"


def _cursor_after_unit_zero(
    attempt: int, generation: int = 0, submitted_at: str | None = "2020-09-30T00:00:00"
) -> str:
    """A cursor as if unit 0's attempt `attempt` (generation `generation`)
    just resolved, with the plan already frozen through 2020-10-31 (two
    months -> two units at this station count)."""
    return json.dumps(
        {
            "plan_end": "2020-10-31",
            "state": {
                "next_index": 0,
                "attempt": attempt,
                "last_run_id": f"2020-09-c00-a{attempt}-g{generation}",
                "transform_requested": False,
                "submitted_at": submitted_at,
                "generation": generation,
            },
        }
    )


def test_success_advances_to_the_next_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_now(monkeypatch, "2020-11-15T12:00:00")  # after plan_end: freeze must hold

    with DagsterInstance.ephemeral() as instance:
        _record_run(
            instance,
            tags={UNIT_TAG: "2020-09-c00", ATTEMPT_TAG: "1", GENERATION_TAG: "0"},
            status=DagsterRunStatus.SUCCESS,
        )

        context = build_sensor_context(
            instance=instance, cursor=_cursor_after_unit_zero(1)
        )
        result = national_backfill_sensor(context)

        [run_request] = result.run_requests
        assert run_request.run_key == "2020-10-c00-a1-g0"
        assert run_request.tags[UNIT_TAG] == "2020-10-c00"
        assert run_request.tags[GENERATION_TAG] == "0"
        assert run_request.tags[ASSET_PARTITION_RANGE_START_TAG] == "2020-10-01"
        assert run_request.tags[ASSET_PARTITION_RANGE_END_TAG] == "2020-10-31"

        envelope = _cursor_state(context.cursor)
        # The plan end is unchanged from what was already in the cursor,
        # even though `_now` has moved well past it -- it is never
        # recomputed after the first evaluation.
        assert envelope["plan_end"] == "2020-10-31"
        assert envelope["state"]["next_index"] == 1
        assert envelope["state"]["attempt"] == 1


def test_failure_retries_the_same_unit_with_incremented_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_now(monkeypatch, "2020-09-30T12:00:00")

    with DagsterInstance.ephemeral() as instance:
        _record_run(
            instance,
            tags={UNIT_TAG: "2020-09-c00", ATTEMPT_TAG: "1", GENERATION_TAG: "0"},
            status=DagsterRunStatus.FAILURE,
        )

        context = build_sensor_context(
            instance=instance, cursor=_cursor_after_unit_zero(1)
        )
        result = national_backfill_sensor(context)

        [run_request] = result.run_requests
        assert run_request.run_key == "2020-09-c00-a2-g0"
        assert run_request.tags[ATTEMPT_TAG] == "2"

        envelope = _cursor_state(context.cursor)
        assert envelope["state"]["next_index"] == 0
        assert envelope["state"]["attempt"] == 2


def test_halted_cursor_keeps_skipping_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_now(monkeypatch, "2020-09-30T12:00:00")
    halted_cursor = _cursor_after_unit_zero(3)

    with DagsterInstance.ephemeral() as instance:
        _record_run(
            instance,
            tags={UNIT_TAG: "2020-09-c00", ATTEMPT_TAG: "3", GENERATION_TAG: "0"},
            status=DagsterRunStatus.FAILURE,
        )

        context = build_sensor_context(instance=instance, cursor=halted_cursor)
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message.startswith("HALTED:")
        # Unchanged: replaying this cursor must halt again next tick too.
        assert context.cursor == halted_cursor


def test_not_found_run_within_timeout_waits_without_resubmitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No run tagged with this unit/attempt/generation exists yet (nothing
    was `_record_run`'d), but it was only submitted a moment ago: wait,
    don't resubmit (which would be a no-op anyway -- same run_key) or halt."""
    _set_now(monkeypatch, "2020-09-30T00:05:00")  # 5 minutes after submitted_at

    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(
            instance=instance,
            cursor=_cursor_after_unit_zero(1, submitted_at="2020-09-30T00:00:00"),
        )
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message == "run in flight"


def test_not_found_run_past_timeout_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same never-appeared run, but 20 minutes on (past the 15-minute
    `not_found_timeout_s` default): halt loudly instead of waiting forever
    on a run_key the daemon will never mint a second run under."""
    _set_now(monkeypatch, "2020-09-30T00:20:00")

    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(
            instance=instance,
            cursor=_cursor_after_unit_zero(1, submitted_at="2020-09-30T00:00:00"),
        )
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message.startswith("HALTED:")
        assert "generation" in result.skip_reason.skip_message


def test_bumped_generation_produces_a_new_run_key_and_submits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented reset procedure: a human clears `last_run_id` and
    bumps `generation` on a halted cursor. The next tick must submit a
    genuinely new run_key (never used by any prior generation), not the
    same one the daemon already refuses to re-launch."""
    _set_now(monkeypatch, "2020-09-30T12:00:00")
    reset_cursor = json.dumps(
        {
            "plan_end": "2020-10-31",
            "state": {
                "next_index": 0,
                "attempt": 3,
                "last_run_id": None,
                "transform_requested": False,
                "submitted_at": None,
                "generation": 1,
            },
        }
    )

    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(instance=instance, cursor=reset_cursor)
        result = national_backfill_sensor(context)

        [run_request] = result.run_requests
        assert run_request.run_key == "2020-09-c00-a3-g1"
        assert run_request.tags[GENERATION_TAG] == "1"

        envelope = _cursor_state(context.cursor)
        assert envelope["state"]["last_run_id"] == "2020-09-c00-a3-g1"
        assert envelope["state"]["generation"] == 1


def test_transform_requested_once_every_unit_is_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_now(monkeypatch, "2020-09-30T12:00:00")
    cursor = json.dumps(
        {
            "plan_end": "2020-09-29",  # single-day plan -- one unit, index 0
            "state": {
                "next_index": 0,
                "attempt": 1,
                "last_run_id": "2020-09-c00-a1-g0",
                "transform_requested": False,
                "submitted_at": "2020-09-30T00:00:00",
                "generation": 0,
            },
        }
    )

    with DagsterInstance.ephemeral() as instance:
        _record_run(
            instance,
            tags={UNIT_TAG: "2020-09-c00", ATTEMPT_TAG: "1", GENERATION_TAG: "0"},
            status=DagsterRunStatus.SUCCESS,
        )

        context = build_sensor_context(instance=instance, cursor=cursor)
        result = national_backfill_sensor(context)

        [run_request] = result.run_requests
        assert run_request.job_name == transform_job.name
        assert run_request.run_key == "transform-g0"
        assert run_request.tags[UNIT_TAG] == TRANSFORM_UNIT_ID
        assert run_request.tags[GENERATION_TAG] == "0"

        envelope = _cursor_state(context.cursor)
        assert envelope["state"]["transform_requested"] is True
        assert envelope["state"]["last_run_id"] == "transform-g0"


def test_done_once_the_transform_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_now(monkeypatch, "2020-09-30T12:00:00")
    cursor = json.dumps(
        {
            "plan_end": "2020-09-29",
            "state": {
                "next_index": 1,
                "attempt": 1,
                "last_run_id": "transform-g0",
                "transform_requested": True,
                "submitted_at": "2020-09-30T00:00:00",
                "generation": 0,
            },
        }
    )

    with DagsterInstance.ephemeral() as instance:
        _record_run(
            instance,
            tags={UNIT_TAG: TRANSFORM_UNIT_ID, GENERATION_TAG: "0"},
            status=DagsterRunStatus.SUCCESS,
        )

        context = build_sensor_context(instance=instance, cursor=cursor)
        result = national_backfill_sensor(context)

        assert result.run_requests is None or result.run_requests == []
        assert result.skip_reason.skip_message == "national backfill complete"
