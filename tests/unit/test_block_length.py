"""Tests for block_length.py's block-length selection and coverage-replay
helpers backing docs/analysis/2026-09-27-block-length (issue #31).

Covers each function against PREREG.md: the pooled/per-station series
filters, the upper-median tie-break, the ACF and deseasonalizing helpers,
Yule-Walker AR fitting and AIC order selection, AR simulation, the
coverage replay (the acceptance check), and the BLOCK_DAYS decision rule
(including its cap and fallback).
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from weather_forecast_audit import block_length, scoring
from weather_forecast_audit.block_length import (
    ArFit,
    BlockChoice,
    ShapeResult,
    acf,
    choose_block_days,
    choose_min_sample_blocks,
    contiguous_dates,
    deseasonalize,
    first_passing_block_days,
    fit_ar_yule_walker,
    n_blocks,
    needs_rerun,
    pooled_series,
    replay_coverage,
    replay_coverage_segments,
    season_segments,
    simulate_ar,
    station_series,
    upper_median_index,
)

pytestmark = pytest.mark.unit


def _row(
    *,
    station: str = "KPHX",
    run_date: date,
    lead_day: int = 1,
    variable: str = "max",
    source: str = "raw_nbm",
    scorable: bool = True,
    error_f: float | None,
) -> dict[str, object]:
    return {
        "station": station,
        "run_date": run_date,
        "lead_day": lead_day,
        "variable": variable,
        "source": source,
        "target_date": run_date + timedelta(days=1),
        "scorable": scorable,
        "error_f": error_f,
    }


# -- pooled_series -------------------------------------------------------


def test_pooled_series_filters_and_aggregates() -> None:
    rows = []
    for i in range(6):
        rows.append(
            _row(
                station=f"S{i:02d}",
                run_date=date(2025, 1, 1),
                error_f=float(i),
            )
        )
    # Duplicate row for S00 on the same date: n_stations must not double count.
    rows.append(_row(station="S00", run_date=date(2025, 1, 1), error_f=10.0))
    # Non-raw source: dropped.
    rows.append(
        _row(
            station="S06",
            run_date=date(2025, 1, 1),
            source="baseline",
            error_f=99.0,
        )
    )
    # Unscorable: dropped.
    rows.append(
        _row(station="S07", run_date=date(2025, 1, 1), scorable=False, error_f=99.0)
    )
    # Null error_f: dropped.
    rows.append(_row(station="S08", run_date=date(2025, 1, 1), error_f=None))
    # A second date with only 5 distinct stations: below min_stations=6, dropped.
    for i in range(5):
        rows.append(
            _row(station=f"T{i:02d}", run_date=date(2025, 1, 2), error_f=float(i))
        )

    frame = pl.DataFrame(rows)
    result = pooled_series(frame, min_stations=6)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["variable"] == "max"
    assert row["lead_day"] == 1
    assert row["run_date"] == date(2025, 1, 1)
    assert row["n_stations"] == 6
    expected_mean = (0 + 1 + 2 + 3 + 4 + 5 + 10) / 7
    assert row["mean_error"] == pytest.approx(expected_mean)


# -- station_series --------------------------------------------------------


def test_station_series_filters_and_columns() -> None:
    rows = [
        _row(station="S00", run_date=date(2025, 1, 1), error_f=1.0),
        _row(station="S00", run_date=date(2025, 1, 2), error_f=2.0),
        _row(
            station="S01",
            run_date=date(2025, 1, 1),
            source="baseline",
            error_f=99.0,
        ),
        _row(
            station="S01",
            run_date=date(2025, 1, 1),
            variable="min",
            scorable=False,
            error_f=99.0,
        ),
        _row(station="S01", run_date=date(2025, 1, 1), lead_day=2, error_f=None),
    ]
    frame = pl.DataFrame(rows)
    result = station_series(frame)

    assert result.columns == ["station", "variable", "lead_day", "run_date", "error_f"]
    assert result.height == 2
    assert result.sort(["station", "run_date"])["error_f"].to_list() == [1.0, 2.0]


# -- upper_median_index ------------------------------------------------------


def test_upper_median_index_twelve_values() -> None:
    values = [5.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0, 11.0, 10.0]
    idx = upper_median_index(values)
    seventh_smallest = sorted(values)[6]
    assert values[idx] == seventh_smallest


def test_upper_median_index_ties_are_stable() -> None:
    values = [1.0, 1.0, 1.0, 1.0]
    # len // 2 == 2; a stable sort keeps ties in original-index order, so
    # sorted position 2 lands on the element originally at index 2.
    assert upper_median_index(values) == 2


def test_upper_median_index_empty_raises() -> None:
    with pytest.raises(ValueError):
        upper_median_index([])


# -- acf ---------------------------------------------------------------------


def test_acf_known_alternating_series() -> None:
    x = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    result = acf(x, max_lag=3)
    # mean=1.5, demeaned = [-.5, .5, -.5, .5, -.5, .5, -.5, .5]; gamma0 = 2.0
    assert result[0] == pytest.approx(-0.875)
    assert result[1] == pytest.approx(0.75)
    assert result[2] == pytest.approx(-0.625)


# -- deseasonalize -------------------------------------------------------


def test_deseasonalize_removes_pure_sinusoid() -> None:
    t = np.arange(365)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 365)
    result = deseasonalize(seasonal, window=31)
    assert result.std() < seasonal.std() * 0.3


def test_deseasonalize_leaves_white_noise_variance_roughly_unchanged() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, 500)
    result = deseasonalize(noise, window=31)
    assert result.std() == pytest.approx(noise.std(), rel=0.2)


def test_deseasonalize_window_is_centered_not_trailing() -> None:
    # A single spike at index 3, window=3 (half=1): the centered window at
    # index 2 is [1, 3] -- it sees the spike one step *ahead* of it. A
    # trailing window ending at the current index would not.
    x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    result = deseasonalize(x, window=3)
    assert result[2] == pytest.approx(-10.0 / 3.0)


# -- fit_ar_yule_walker --------------------------------------------------


def test_fit_ar_yule_walker_recovers_ar1() -> None:
    rng = np.random.default_rng(20260927)
    n = 20_000
    rho = 0.8
    x = np.empty(n)
    x[0] = rng.normal(0.0, 1.0 / np.sqrt(1 - rho**2))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal(0.0, 1.0)

    fit = fit_ar_yule_walker(x, max_p=7)

    assert fit.p == 1
    assert fit.phi[0] == pytest.approx(rho, abs=0.02)


def test_fit_ar_yule_walker_selects_white_noise() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 5000)

    fit = fit_ar_yule_walker(x, max_p=7)

    assert fit.p == 0
    assert fit.phi.size == 0
    assert fit.sigma2 == pytest.approx(float(x.var()), rel=0.05)


def test_fit_ar_yule_walker_recovers_ar2() -> None:
    rng = np.random.default_rng(7)
    n = 20_000
    phi1, phi2 = 0.5, -0.3
    x = np.empty(n)
    x[0] = rng.normal()
    x[1] = rng.normal()
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + rng.normal()

    fit = fit_ar_yule_walker(x, max_p=7)

    assert fit.p == 2
    assert fit.phi[0] == pytest.approx(phi1, abs=0.05)
    assert fit.phi[1] == pytest.approx(phi2, abs=0.05)


def test_fit_ar_yule_walker_uses_biased_autocovariance() -> None:
    # A small, deterministic series where an unbiased (/ (n - k)) covariance
    # would give a visibly different reflection coefficient than the biased
    # (/ n) one the docstring specifies.
    x = np.array([1.0, 0.5, 0.6, 0.2, 0.1, -0.1, -0.3, -0.2, 0.0, 0.4])
    n = x.size
    demeaned = x - x.mean()
    gamma0 = np.sum(demeaned**2) / n
    gamma1 = np.sum(demeaned[:-1] * demeaned[1:]) / n
    expected_phi1 = gamma1 / gamma0

    fit = fit_ar_yule_walker(x, max_p=1)

    assert fit.p == 1
    assert fit.phi[0] == pytest.approx(expected_phi1, abs=1e-9)


# -- simulate_ar -----------------------------------------------------------


def test_simulate_ar_shape_and_reproducibility() -> None:
    fit = ArFit(p=1, phi=np.array([0.5]), sigma2=1.0, mean=2.0)

    series_a = simulate_ar(fit, n=50, rng=np.random.default_rng(42), burn_in=20)
    series_b = simulate_ar(fit, n=50, rng=np.random.default_rng(42), burn_in=20)

    assert series_a.shape == (50,)
    np.testing.assert_array_equal(series_a, series_b)


def test_simulate_ar_discards_burn_in() -> None:
    # p=0 keeps the recursion out of it: `series` is exactly the innovations
    # array, so the returned slice's offset is the only thing under test.
    fit = ArFit(p=0, phi=np.array([]), sigma2=2.0, mean=1.0)
    n, burn_in = 5, 20

    result = simulate_ar(fit, n=n, rng=np.random.default_rng(3), burn_in=burn_in)

    expected_full = np.random.default_rng(3).normal(
        0.0, np.sqrt(fit.sigma2), size=n + burn_in
    )
    expected = expected_full[burn_in:] + fit.mean
    np.testing.assert_allclose(result, expected)


# -- choose_block_days -----------------------------------------------------


def test_choose_block_days_fractional_ceil() -> None:
    b = {("max", 1): (2.3, 1.0), ("max", 2): (1.0, 1.0)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=3, raw_ceiling=3, capped=False)


def test_choose_block_days_floors_at_one() -> None:
    b = {("max", 1): (0.2, 0.1)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=1, raw_ceiling=1, capped=False)


def test_choose_block_days_floor_applies_at_exactly_zero() -> None:
    # ceil(0.2) is already 1 -- the floor above never actually exercises
    # `max(1, ...)`. Only an estimate of exactly 0 does.
    b = {("max", 1): (0.0, 0.0)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=1, raw_ceiling=1, capped=False)


def test_choose_block_days_station_can_exceed_pooled() -> None:
    b = {("max", 1): (1.0, 4.6)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=5, raw_ceiling=5, capped=False)


def test_choose_block_days_exact_cap_passes() -> None:
    b = {("max", 1): (14.0, 5.0)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=14, raw_ceiling=14, capped=False)


def test_choose_block_days_over_cap_is_not_silently_capped() -> None:
    b = {("max", 1): (14.2, 5.0)}
    choice = choose_block_days(b)
    assert choice == BlockChoice(value=None, raw_ceiling=15, capped=True)


# -- first_passing_block_days -----------------------------------------------


def test_first_passing_block_days_passes_at_start() -> None:
    assert first_passing_block_days(lambda _: [0.95, 0.96], start=3) == 3


def test_first_passing_block_days_passes_later() -> None:
    def coverage_at(block_days: int) -> list[float]:
        return [0.95] if block_days >= 5 else [0.5]

    assert first_passing_block_days(coverage_at, start=1) == 5


def test_first_passing_block_days_never_passes() -> None:
    assert first_passing_block_days(lambda _: [0.1], start=1, cap=14) is None


def test_first_passing_block_days_can_return_the_cap_itself() -> None:
    def coverage_at(block_days: int) -> list[float]:
        return [0.95] if block_days == 14 else [0.5]

    assert first_passing_block_days(coverage_at, start=1, cap=14) == 14


def test_first_passing_block_days_requires_all_series_to_pass() -> None:
    def coverage_at(block_days: int) -> list[float]:
        if block_days == 3:
            return [0.95, 0.95]
        return [0.95, 0.10]

    assert first_passing_block_days(coverage_at, start=1) == 3


def test_first_passing_block_days_accepts_exact_threshold() -> None:
    assert first_passing_block_days(lambda _: [0.90], start=1) == 1


# -- replay_coverage ---------------------------------------------------------


def test_replay_coverage_white_noise_block_one_near_nominal() -> None:
    fit = ArFit(p=0, phi=np.array([]), sigma2=1.0, mean=0.7)
    rng = np.random.default_rng(20260927)

    coverage = replay_coverage(
        fit, n=200, block_days=1, n_sims=150, rng=rng, n_boot=300
    )

    assert 0.88 <= coverage <= 1.0


def test_replay_coverage_ar1_block_one_worse_than_block_fourteen() -> None:
    rho = 0.8
    fit = ArFit(p=1, phi=np.array([rho]), sigma2=(1.5**2) * (1 - rho**2), mean=0.7)
    n = 150

    coverage_block1 = replay_coverage(
        fit, n=n, block_days=1, n_sims=150, rng=np.random.default_rng(1), n_boot=300
    )
    coverage_block14 = replay_coverage(
        fit, n=n, block_days=14, n_sims=150, rng=np.random.default_rng(1), n_boot=300
    )

    assert coverage_block1 < 0.75
    assert coverage_block14 > coverage_block1


# -- contiguous_dates ---------------------------------------------------------


def test_contiguous_dates_default_start() -> None:
    segments = contiguous_dates(5)
    assert segments == [[date(2025, 1, 1) + timedelta(days=i) for i in range(5)]]


def test_contiguous_dates_custom_start() -> None:
    segments = contiguous_dates(3, start=date(2030, 6, 1))
    assert segments == [[date(2030, 6, 1), date(2030, 6, 2), date(2030, 6, 3)]]


# -- season_segments -----------------------------------------------------------


def test_season_segments_single_year_spans_jja() -> None:
    segments = season_segments(1)
    assert len(segments) == 1
    assert len(segments[0]) == 92
    assert segments[0][0] == date(2021, 6, 1)
    assert segments[0][-1] == date(2021, 8, 31)


def test_season_segments_multiple_years_in_order() -> None:
    segments = season_segments(3, first_year=2021)
    assert [segment[0].year for segment in segments] == [2021, 2022, 2023]
    assert all(len(segment) == 92 for segment in segments)
    assert all(segment[0] == date(segment[0].year, 6, 1) for segment in segments)
    assert all(segment[-1] == date(segment[0].year, 8, 31) for segment in segments)


# -- n_blocks ------------------------------------------------------------------


def test_n_blocks_contiguous_365_at_14_is_27() -> None:
    dates = contiguous_dates(365)[0]
    assert n_blocks(dates, 14) == 27


def test_n_blocks_season_segment_92_dates_at_14_is_7() -> None:
    segment = season_segments(1)[0]
    assert len(segment) == 92
    assert n_blocks(segment, 14) == 7


def test_n_blocks_counts_distinct_ids_not_dates() -> None:
    # 14 consecutive dates at block_days=14 always land in at most 2 blocks
    # (a run can straddle an id boundary); a single day is exactly 1.
    assert n_blocks([date(2025, 1, 1)], 14) == 1


# -- replay_coverage_segments ---------------------------------------------------


def test_replay_coverage_segments_white_noise_near_nominal() -> None:
    fit = ArFit(p=0, phi=np.array([]), sigma2=1.0, mean=0.7)
    segments = [
        contiguous_dates(60, start=date(2025, 1, 1))[0],
        contiguous_dates(60, start=date(2026, 1, 1))[0],
    ]
    rng = np.random.default_rng(20260929)

    coverage = replay_coverage_segments(
        fit, segments, block_days=1, n_sims=150, rng=rng, n_boot=300
    )

    assert 0.88 <= coverage <= 1.0


def test_replay_coverage_segments_draws_independently_per_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit = ArFit(p=0, phi=np.array([]), sigma2=1.0, mean=0.0)
    segment_a = contiguous_dates(5, start=date(2025, 1, 1))[0]
    segment_b = contiguous_dates(5, start=date(2026, 1, 1))[0]
    captured: list[pl.DataFrame] = []
    original_score = scoring.score

    def spy_score(frame: pl.DataFrame, *args: object, **kwargs: object) -> pl.DataFrame:
        captured.append(frame)
        return original_score(frame, *args, **kwargs)

    monkeypatch.setattr(scoring, "score", spy_score)

    replay_coverage_segments(
        fit,
        [segment_a, segment_b],
        block_days=14,
        n_sims=1,
        rng=np.random.default_rng(42),
        n_boot=5,
    )

    values = captured[0]["error_f"].to_numpy()
    first_segment_values, second_segment_values = values[:5], values[5:]
    assert not np.array_equal(first_segment_values, second_segment_values)


def test_replay_coverage_segments_calls_simulate_ar_once_per_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PREREG.md (#37): "Each simulation draws one independent `simulate_ar`
    series per segment". A shared series drawn once and sliced across
    segments would still leave the two halves numerically unequal (the
    weaker check `test_..._draws_independently_per_segment` above passes
    either way), so pin the call shape directly: one `simulate_ar` call per
    segment, each sized to that segment's own length."""
    fit = ArFit(p=0, phi=np.array([]), sigma2=1.0, mean=0.0)
    segment_a = contiguous_dates(5, start=date(2025, 1, 1))[0]
    segment_b = contiguous_dates(7, start=date(2026, 1, 1))[0]
    call_sizes: list[int] = []
    original_simulate_ar = block_length.simulate_ar

    def spy_simulate_ar(
        fit_arg: ArFit, n: int, rng: np.random.Generator, **kwargs: object
    ) -> np.ndarray:
        call_sizes.append(n)
        return original_simulate_ar(fit_arg, n, rng, **kwargs)

    monkeypatch.setattr(block_length, "simulate_ar", spy_simulate_ar)

    replay_coverage_segments(
        fit,
        [segment_a, segment_b],
        block_days=14,
        n_sims=1,
        rng=np.random.default_rng(42),
        n_boot=5,
    )

    assert call_sizes == [len(segment_a), len(segment_b)]


def test_replay_coverage_matches_replay_coverage_segments_bit_identical() -> None:
    # Pins the refactor: replay_coverage now delegates to
    # replay_coverage_segments(fit, contiguous_dates(n), ...), and must draw
    # from `rng` in the same order to reproduce the pre-refactor numbers
    # exactly for a fixed seed.
    fit = ArFit(p=1, phi=np.array([0.4]), sigma2=1.0, mean=0.3)
    n, block_days, n_sims, n_boot = 60, 14, 30, 200

    coverage_direct = replay_coverage(
        fit, n, block_days, n_sims=n_sims, rng=np.random.default_rng(7), n_boot=n_boot
    )
    coverage_segments = replay_coverage_segments(
        fit,
        contiguous_dates(n),
        block_days,
        n_sims=n_sims,
        rng=np.random.default_rng(7),
        n_boot=n_boot,
    )

    assert coverage_direct == coverage_segments


# -- ShapeResult / choose_min_sample_blocks -------------------------------------


def test_choose_min_sample_blocks_monotone_pass() -> None:
    results = [
        ShapeResult("60d", 5, (0.95, 0.95)),
        ShapeResult("90d", 10, (0.95, 0.95)),
        ShapeResult("120d", 15, (0.95, 0.95)),
    ]
    assert choose_min_sample_blocks(results) == 5


def test_choose_min_sample_blocks_nonmonotone_failure_pushes_k_up() -> None:
    results = [
        ShapeResult("60d", 5, (0.95, 0.95)),
        ShapeResult("90d", 10, (0.80, 0.95)),  # fails
        ShapeResult("120d", 15, (0.95, 0.95)),
    ]
    assert choose_min_sample_blocks(results) == 15


def test_choose_min_sample_blocks_largest_fails_returns_none() -> None:
    results = [
        ShapeResult("60d", 5, (0.95, 0.95)),
        ShapeResult("90d", 10, (0.95, 0.95)),
        ShapeResult("120d", 15, (0.80, 0.95)),  # largest shape fails
    ]
    assert choose_min_sample_blocks(results) is None


def test_choose_min_sample_blocks_exact_threshold_passes() -> None:
    results = [ShapeResult("60d", 5, (0.90, 0.90))]
    assert choose_min_sample_blocks(results) == 5


def test_choose_min_sample_blocks_ties_require_all_at_that_block_count() -> None:
    results = [
        ShapeResult("a", 10, (0.95,)),
        ShapeResult("b", 10, (0.80,)),  # ties with a at n_blocks=10, fails
        ShapeResult("c", 20, (0.95,)),
    ]
    # k=10 is rejected because b fails despite a passing at the same block
    # count; the next distinct block count, 20, is the answer.
    assert choose_min_sample_blocks(results) == 20


def test_choose_min_sample_blocks_empty_returns_none() -> None:
    assert choose_min_sample_blocks([]) is None


# -- needs_rerun -----------------------------------------------------------------


def test_needs_rerun_flags_failure_above_a_passing_smaller_shape() -> None:
    results = [
        ShapeResult("60d", 5, (0.95, 0.95)),
        ShapeResult("90d", 10, (0.80, 0.95)),  # fails; 5-block shape passed
        ShapeResult("120d", 15, (0.95, 0.95)),
    ]
    assert needs_rerun(results) == [1]


def test_needs_rerun_does_not_flag_when_no_smaller_shape_passes() -> None:
    results = [
        ShapeResult("60d", 5, (0.80, 0.95)),  # fails
        ShapeResult("90d", 10, (0.70, 0.95)),  # fails, nothing smaller passed
    ]
    assert needs_rerun(results) == []


def test_needs_rerun_all_pass_returns_empty() -> None:
    results = [ShapeResult("60d", 5, (0.95,)), ShapeResult("90d", 10, (0.95,))]
    assert needs_rerun(results) == []


def test_needs_rerun_ties_at_equal_block_count_do_not_trigger_rerun() -> None:
    """A passing shape at the *same* block count is not "strictly fewer
    n_blocks" (the docstring's own words), so it must not excuse a tied
    failing shape as noise."""
    results = [
        ShapeResult("a", 10, (0.95,)),  # passes
        ShapeResult("b", 10, (0.80,)),  # fails, ties a's block count
    ]
    assert needs_rerun(results) == []
