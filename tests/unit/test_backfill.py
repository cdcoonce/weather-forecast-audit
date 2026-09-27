"""Pure backfill planning and state machine (issue #11 D-backfill).

No Dagster imports anywhere in this file or in `weather_forecast_audit.backfill`:
`sensors.national_backfill_sensor` is the only Dagster-aware caller, and every
branch of `decide` is exercised here without a running instance.
"""

from datetime import UTC, date, datetime

import pytest

from weather_forecast_audit.backfill import (
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

pytestmark = pytest.mark.unit


# -- plan_units ----------------------------------------------------------


def test_plan_units_single_short_month_one_chunk() -> None:
    units = plan_units(date(2020, 9, 29), date(2020, 9, 30), ["KPHX", "KORD"], 30)

    assert len(units) == 1
    unit = units[0]
    assert unit.index == 0
    assert unit.start == date(2020, 9, 29)
    assert unit.end == date(2020, 9, 30)
    assert unit.stations == ("KORD", "KPHX")
    assert unit.chunk == 0
    assert unit.unit_id == "2020-09-c00"


def test_plan_units_spans_multiple_months_clipped_at_both_ends() -> None:
    units = plan_units(date(2020, 9, 29), date(2020, 11, 5), ["KPHX"], 30)

    months = [(u.start, u.end) for u in units]
    assert months == [
        (date(2020, 9, 29), date(2020, 9, 30)),
        (date(2020, 10, 1), date(2020, 10, 31)),
        (date(2020, 11, 1), date(2020, 11, 5)),
    ]
    assert [u.index for u in units] == [0, 1, 2]


def test_plan_units_chunks_stations_and_orders_month_major_chunk_minor() -> None:
    stations = ["KORD", "KPHX", "KDEN"]  # deliberately unsorted
    units = plan_units(date(2020, 9, 29), date(2020, 10, 31), stations, chunk_size=2)

    assert [u.stations for u in units] == [
        ("KDEN", "KORD"),
        ("KPHX",),
        ("KDEN", "KORD"),
        ("KPHX",),
    ]
    assert [u.index for u in units] == [0, 1, 2, 3]
    assert [u.chunk for u in units] == [0, 1, 0, 1]
    assert [u.unit_id for u in units] == [
        "2020-09-c00",
        "2020-09-c01",
        "2020-10-c00",
        "2020-10-c01",
    ]


def test_plan_units_no_stations_yields_no_units() -> None:
    units = plan_units(date(2020, 9, 29), date(2020, 9, 30), [], 30)

    assert units == []


def test_plan_units_rejects_chunk_size_below_one() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        plan_units(date(2020, 9, 29), date(2020, 9, 30), ["KPHX"], 0)


def test_plan_units_rejects_end_before_archive_start() -> None:
    with pytest.raises(ValueError, match="archive_start"):
        plan_units(date(2020, 9, 29), date(2020, 9, 28), ["KPHX"], 30)


# -- in_blackout -----------------------------------------------------------

# Phoenix (America/Phoenix) has no DST, so its UTC offset is a fixed -07:00.
# 2026-09-27 06:00 America/Phoenix == 2026-09-27 13:00 UTC.


def _naive_utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """A naive UTC datetime (repo-wide convention; see ClockResource.now())."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC).replace(tzinfo=None)


def test_in_blackout_true_when_now_is_inside_the_window() -> None:
    now_utc = _naive_utc(2026, 9, 27, 13, 0)  # 06:00 local, inside 05:45-07:15

    assert in_blackout(now_utc, horizon_s=60) is True


def test_in_blackout_false_when_interval_is_well_outside_the_window() -> None:
    now_utc = _naive_utc(2026, 9, 27, 20, 0)  # 13:00 local

    assert in_blackout(now_utc, horizon_s=600) is False


def test_in_blackout_true_when_horizon_reaches_forward_into_the_window() -> None:
    # 05:00 local, window opens at 05:45 local -- a 60-minute horizon reaches it.
    now_utc = _naive_utc(2026, 9, 27, 12, 0)

    assert in_blackout(now_utc, horizon_s=3600) is True
    assert in_blackout(now_utc, horizon_s=600) is False


def test_in_blackout_true_when_interval_crosses_midnight_into_next_window() -> None:
    # 23:50 local on 09-27; a long horizon reaches 05:45 local on 09-28.
    now_utc = _naive_utc(2026, 9, 28, 6, 50)

    assert in_blackout(now_utc, horizon_s=6 * 3600) is True


# -- CursorState JSON round-trip -------------------------------------------


def test_cursor_state_from_json_none_or_empty_is_initial_state() -> None:
    initial = CursorState(next_index=0, attempt=1, last_run_id=None)

    assert CursorState.from_json(None) == initial
    assert CursorState.from_json("") == initial


def test_cursor_state_round_trips_through_json() -> None:
    state = CursorState(
        next_index=3, attempt=2, last_run_id="2020-09-c00-a2", transform_requested=True
    )

    assert CursorState.from_json(state.to_json()) == state


# -- decide: the state machine ----------------------------------------------


def _state(**kwargs: object) -> CursorState:
    defaults = {
        "next_index": 0,
        "attempt": 1,
        "last_run_id": None,
        "transform_requested": False,
    }
    defaults.update(kwargs)
    return CursorState(**defaults)  # type: ignore[arg-type]


def test_decide_first_evaluation_submits_unit_zero() -> None:
    decision = decide(
        _state(), n_units=5, last_run_status=None, slot_busy=False, blackout=False
    )

    assert decision == Submit(0, 1, _state())


def test_decide_waits_when_a_run_is_in_flight_even_if_slot_is_not_busy() -> None:
    state = _state(last_run_id="2020-09-c00-a1")

    for status in ["QUEUED", "NOT_STARTED", "STARTING", "STARTED", "CANCELING", None]:
        decision = decide(
            state, n_units=5, last_run_status=status, slot_busy=False, blackout=False
        )
        assert decision == Wait("run in flight"), status


def test_decide_advances_to_next_unit_after_success() -> None:
    state = _state(next_index=0, attempt=2, last_run_id="2020-09-c00-a2")

    decision = decide(
        state, n_units=5, last_run_status="SUCCESS", slot_busy=False, blackout=False
    )

    assert decision == Submit(1, 1, _state(next_index=1, attempt=1, last_run_id=None))


def test_decide_retries_same_unit_after_failure() -> None:
    state = _state(next_index=2, attempt=1, last_run_id="2020-11-c00-a1")

    decision = decide(
        state, n_units=5, last_run_status="FAILURE", slot_busy=False, blackout=False
    )

    assert decision == Submit(2, 2, _state(next_index=2, attempt=2, last_run_id=None))


def test_decide_retries_same_unit_after_cancellation() -> None:
    state = _state(next_index=2, attempt=1, last_run_id="2020-11-c00-a1")

    decision = decide(
        state, n_units=5, last_run_status="CANCELED", slot_busy=False, blackout=False
    )

    assert decision == Submit(2, 2, _state(next_index=2, attempt=2, last_run_id=None))


def test_decide_halts_after_max_attempts_exceeded() -> None:
    state = _state(next_index=2, attempt=3, last_run_id="2020-11-c00-a3")

    decision = decide(
        state,
        n_units=5,
        last_run_status="FAILURE",
        slot_busy=False,
        blackout=False,
        max_attempts=3,
    )

    assert isinstance(decision, Halt)
    assert "2" in decision.reason
    assert "2020-11-c00-a3" in decision.reason


def test_decide_stays_halted_on_repeated_evaluation_with_the_same_cursor() -> None:
    """A Halt must not advance the cursor: replaying the identical inputs
    (as the sensor does on every subsequent tick with an unmodified cursor)
    must halt again, forever, until a human resets the cursor by hand."""
    state = _state(next_index=2, attempt=3, last_run_id="2020-11-c00-a3")
    kwargs = {
        "n_units": 5,
        "last_run_status": "FAILURE",
        "slot_busy": False,
        "blackout": False,
        "max_attempts": 3,
    }

    first = decide(state, **kwargs)
    second = decide(state, **kwargs)

    assert isinstance(first, Halt)
    assert first == second


def test_decide_waits_for_slot_when_units_remain() -> None:
    decision = decide(
        _state(), n_units=5, last_run_status=None, slot_busy=True, blackout=False
    )

    assert decision == Wait("slot busy")


def test_decide_waits_for_blackout_when_units_remain() -> None:
    decision = decide(
        _state(), n_units=5, last_run_status=None, slot_busy=False, blackout=True
    )

    assert decision == Wait("blackout")


def test_decide_blackout_blocks_submit_but_still_books_a_success() -> None:
    """A SUCCESS is recognized (the decision reflects the advanced unit, not
    'run in flight') even though blackout blocks the resulting Submit."""
    state = _state(next_index=0, attempt=1, last_run_id="2020-09-c00-a1")

    decision = decide(
        state, n_units=5, last_run_status="SUCCESS", slot_busy=False, blackout=True
    )

    assert decision == Wait("blackout")


def test_decide_requests_transform_only_after_every_unit_is_done() -> None:
    state = _state(next_index=2, attempt=1, last_run_id="2020-11-c00-a1")

    decision = decide(
        state, n_units=3, last_run_status="SUCCESS", slot_busy=False, blackout=False
    )

    assert decision == SubmitTransform(
        _state(next_index=3, attempt=1, last_run_id=None, transform_requested=True)
    )


def test_decide_transform_request_respects_slot_busy_and_blackout() -> None:
    state = _state(next_index=3, attempt=1, last_run_id=None)

    assert decide(
        state, n_units=3, last_run_status=None, slot_busy=True, blackout=False
    ) == Wait("slot busy")
    assert decide(
        state, n_units=3, last_run_status=None, slot_busy=False, blackout=True
    ) == Wait("blackout")


def test_decide_done_after_transform_succeeds() -> None:
    state = _state(
        next_index=3, attempt=1, last_run_id="transform", transform_requested=True
    )

    decision = decide(
        state, n_units=3, last_run_status="SUCCESS", slot_busy=False, blackout=False
    )

    assert decision == Done()


def test_decide_halts_when_transform_fails() -> None:
    state = _state(
        next_index=3, attempt=1, last_run_id="transform", transform_requested=True
    )

    decision = decide(
        state, n_units=3, last_run_status="FAILURE", slot_busy=False, blackout=False
    )

    assert isinstance(decision, Halt)
    assert "transform" in decision.reason


def test_decide_is_idempotent_given_identical_inputs() -> None:
    state = _state(next_index=1, attempt=1, last_run_id=None)
    kwargs = {
        "n_units": 5,
        "last_run_status": None,
        "slot_busy": False,
        "blackout": False,
    }

    assert decide(state, **kwargs) == decide(state, **kwargs)
