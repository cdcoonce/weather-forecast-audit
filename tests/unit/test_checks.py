"""Pure evaluators backing the freshness and gap-rate asset checks (D7)."""

from datetime import UTC, datetime, timedelta

import pytest

from weather_forecast_audit.checks import evaluate_freshness, evaluate_gap_rate

pytestmark = pytest.mark.unit


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def test_evaluate_freshness_passes_when_recent() -> None:
    now = _utc("2024-01-02 00:00:00")
    latest = now - timedelta(hours=1)

    passed, metadata = evaluate_freshness(latest, now, timedelta(hours=36))

    assert passed is True
    assert metadata["age_hours"] == pytest.approx(1.0)


def test_evaluate_freshness_fails_when_stale() -> None:
    now = _utc("2024-01-02 00:00:00")
    latest = now - timedelta(hours=48)

    passed, metadata = evaluate_freshness(latest, now, timedelta(hours=36))

    assert passed is False
    assert metadata["age_hours"] == pytest.approx(48.0)


def test_evaluate_freshness_fails_on_none() -> None:
    now = _utc("2024-01-02 00:00:00")

    passed, metadata = evaluate_freshness(None, now, timedelta(hours=36))

    assert passed is False
    assert metadata["age_hours"] is None


def test_evaluate_freshness_boundary_is_inclusive() -> None:
    now = _utc("2024-01-02 00:00:00")
    max_age = timedelta(hours=36)
    latest = now - max_age

    passed, metadata = evaluate_freshness(latest, now, max_age)

    assert passed is True
    assert metadata["age_hours"] == pytest.approx(36.0)


def test_evaluate_gap_rate_below_threshold_passes() -> None:
    passed, rate = evaluate_gap_rate(1, 100, 0.05)

    assert passed is True
    assert rate == pytest.approx(0.01)


def test_evaluate_gap_rate_exactly_at_threshold_passes() -> None:
    passed, rate = evaluate_gap_rate(5, 100, 0.05)

    assert passed is True
    assert rate == pytest.approx(0.05)


def test_evaluate_gap_rate_above_threshold_fails() -> None:
    passed, rate = evaluate_gap_rate(6, 100, 0.05)

    assert passed is False
    assert rate == pytest.approx(0.06)


def test_evaluate_gap_rate_zero_total_raises() -> None:
    with pytest.raises(ValueError, match="stations_total"):
        evaluate_gap_rate(0, 0, 0.05)
