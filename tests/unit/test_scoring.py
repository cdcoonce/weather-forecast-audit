"""Tests for the scoring module: known answers, conservation, and the
date-block bootstrap's coverage advantage over row-level resampling.

Test 7 (coverage) is the headline test: it proves date-block CIs cover a
known bias at close to their nominal rate while naive row-level resampling
does not, on synthetic data with the day-to-day correlation real forecast
errors have. Test 8 proves the `block_days` knob helps on serially
correlated data, without picking a production default (that is a follow-up,
pre-registered measurement -- see docs/methodology.md).
"""

import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from weather_forecast_audit.scoring import (
    MIN_SAMPLE_DATES,
    _block_bootstrap,
    _no_detectable_bias,
    score,
)

pytestmark = pytest.mark.unit


# -- fixtures ------------------------------------------------------------


def _row(
    *,
    station: str = "KPHX",
    run_date: date,
    lead_day: int = 1,
    variable: str = "max",
    source: str = "raw_nbm",
    target_date: date,
    scorable: bool = True,
    error_f: float | None,
) -> dict[str, object]:
    return {
        "station": station,
        "run_date": run_date,
        "lead_day": lead_day,
        "variable": variable,
        "source": source,
        "target_date": target_date,
        "scorable": scorable,
        "error_f": error_f,
    }


def _known_answer_rows() -> list[dict[str, object]]:
    """Two sources, three raw_nbm rows, three challenger rows.

    Matched pairs (same station/run_date/lead_day/variable):
    2025-01-01 (raw 1.0 <-> challenger 0.5) and 2025-01-02 (raw -2.0 <->
    challenger -1.0). Challenger's 2025-01-05 row is unmatched: no raw_nbm
    row exists for that run_date.

    By hand:
      raw_nbm:      bias = (1 + -2 + 3) / 3 = 0.666666...
                    mae  = (1 + 2 + 3) / 3 = 2.0
                    skill = 0.0 (forced)
      challenger:   bias = (0.5 + -1 + 10) / 3 = 3.166666...
                    mae  = (0.5 + 1 + 10) / 3 = 3.833333...
                    skill (matched only) = 1 - (|0.5| + |-1|) / (|1| + |-2|)
                                         = 1 - 1.5 / 3.0 = 0.5
    """
    return [
        _row(
            run_date=date(2025, 1, 1),
            target_date=date(2025, 1, 2),
            error_f=1.0,
        ),
        _row(
            run_date=date(2025, 1, 2),
            target_date=date(2025, 1, 3),
            error_f=-2.0,
        ),
        _row(
            run_date=date(2025, 1, 3),
            target_date=date(2025, 1, 4),
            error_f=3.0,
        ),
        _row(
            run_date=date(2025, 1, 1),
            target_date=date(2025, 1, 2),
            source="challenger",
            error_f=0.5,
        ),
        _row(
            run_date=date(2025, 1, 2),
            target_date=date(2025, 1, 3),
            source="challenger",
            error_f=-1.0,
        ),
        _row(
            run_date=date(2025, 1, 5),
            target_date=date(2025, 1, 6),
            source="challenger",
            error_f=10.0,
        ),
    ]


def _known_answer_frame() -> pl.DataFrame:
    return pl.DataFrame(_known_answer_rows())


# -- 1. known answers ------------------------------------------------------


def test_known_answers_bias_mae_skill() -> None:
    result = score(_known_answer_frame(), n_boot=50).sort("source")
    raw = result.filter(pl.col("source") == "raw_nbm").row(0, named=True)
    challenger = result.filter(pl.col("source") == "challenger").row(0, named=True)

    assert raw["bias"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert raw["mae"] == pytest.approx(2.0, abs=1e-12)
    assert raw["skill"] == 0.0

    assert challenger["bias"] == pytest.approx(9.5 / 3.0, abs=1e-12)
    assert challenger["mae"] == pytest.approx(11.5 / 3.0, abs=1e-12)
    assert challenger["skill"] == pytest.approx(0.5, abs=1e-12)


# -- 2. skill exactly 0 when a source copies raw_nbm ------------------------


def test_skill_is_exactly_zero_for_identical_source() -> None:
    raw_rows = [
        _row(run_date=date(2025, 1, 1), target_date=date(2025, 1, 2), error_f=1.0),
        _row(run_date=date(2025, 1, 2), target_date=date(2025, 1, 3), error_f=-2.0),
        _row(run_date=date(2025, 1, 3), target_date=date(2025, 1, 4), error_f=3.0),
    ]
    baseline_rows = [{**r, "source": "baseline"} for r in raw_rows]
    frame = pl.DataFrame(raw_rows + baseline_rows)

    result = score(frame, n_boot=50)
    baseline = result.filter(pl.col("source") == "baseline").row(0, named=True)

    assert baseline["skill"] == 0.0
    assert baseline["skill_lo"] == 0.0
    assert baseline["skill_hi"] == 0.0


# -- 3. unmatched pairs excluded from skill but not bias/mae ----------------


def test_unmatched_row_affects_bias_mae_not_skill() -> None:
    result = score(_known_answer_frame(), n_boot=50)
    challenger = result.filter(pl.col("source") == "challenger").row(0, named=True)

    # bias/mae include the unmatched 10.0 row (n=3); skill's matched MAE
    # ratio only sees the two matched rows (1.5 / 3.0 = 0.5), not 3 of 3.
    assert challenger["n"] == 3
    assert challenger["bias"] == pytest.approx(9.5 / 3.0, abs=1e-12)
    assert challenger["skill"] == pytest.approx(0.5, abs=1e-12)


# -- 4. unscorable rows dropped ----------------------------------------------


def test_unscorable_rows_do_not_change_any_number() -> None:
    baseline = score(_known_answer_frame(), n_boot=50).sort(["source"])

    garbage = _row(
        run_date=date(2025, 1, 9),
        target_date=date(2025, 1, 10),
        source="challenger",
        scorable=False,
        error_f=9999.0,
    )
    with_garbage = score(
        pl.DataFrame(_known_answer_rows() + [garbage]), n_boot=50
    ).sort(["source"])

    assert baseline["n"].to_list() == with_garbage["n"].to_list()
    assert baseline["bias"].to_list() == pytest.approx(
        with_garbage["bias"].to_list(), abs=1e-12
    )
    assert baseline["mae"].to_list() == pytest.approx(
        with_garbage["mae"].to_list(), abs=1e-12
    )


# -- 5. conservation ---------------------------------------------------------


def _random_conservation_frame() -> pl.DataFrame:
    rng = np.random.default_rng(12345)
    stations = [f"S{i:02d}" for i in range(5)]
    leads = [1, 2, 3]
    variables = ["max", "min"]
    sources = ["raw_nbm", "challenger"]
    run_dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(40)]

    records: list[dict[str, object]] = []
    for run_date_ in run_dates:
        for station in stations:
            for lead in leads:
                for variable in variables:
                    target = run_date_ + timedelta(days=lead)
                    for source in sources:
                        scorable = bool(rng.random() > 0.1)
                        error = float(rng.normal(0, 2.0)) if scorable else None
                        records.append(
                            _row(
                                station=station,
                                run_date=run_date_,
                                lead_day=lead,
                                variable=variable,
                                source=source,
                                target_date=target,
                                scorable=scorable,
                                error_f=error,
                            )
                        )
    return pl.DataFrame(records)


@pytest.mark.parametrize(
    "by",
    [
        ("station",),
        ("month",),
        ("season",),
        ("lead_day",),
        ("variable",),
        ("station", "month", "lead_day", "variable"),
    ],
)
def test_conservation_of_n_and_error_sums(by: tuple[str, ...]) -> None:
    frame = _random_conservation_frame()
    unsliced = score(frame, by=(), n_boot=2).sort("source")
    sliced = score(frame, by=by, n_boot=2)

    conserved = (
        sliced.group_by("source")
        .agg(
            n=pl.col("n").sum(),
            n_bias=(pl.col("n") * pl.col("bias")).sum(),
            n_mae=(pl.col("n") * pl.col("mae")).sum(),
        )
        .sort("source")
    )

    for row_unsliced, row_conserved in zip(
        unsliced.iter_rows(named=True), conserved.iter_rows(named=True), strict=True
    ):
        assert row_unsliced["source"] == row_conserved["source"]
        assert row_unsliced["n"] == row_conserved["n"]
        assert row_unsliced["n"] * row_unsliced["bias"] == pytest.approx(
            row_conserved["n_bias"], abs=1e-9
        )
        assert row_unsliced["n"] * row_unsliced["mae"] == pytest.approx(
            row_conserved["n_mae"], abs=1e-9
        )


# -- 6. month/season derived from target_date --------------------------------


def test_month_and_season_derived_from_target_date_not_run_date() -> None:
    rows = [
        _row(  # lead 2: run_date Nov 30 verifies Dec 2 -> Dec / DJF
            run_date=date(2025, 11, 30),
            lead_day=2,
            target_date=date(2025, 12, 2),
            error_f=1.0,
        ),
        _row(
            run_date=date(2025, 2, 27),
            lead_day=1,
            target_date=date(2025, 2, 28),
            station="KDEN",
            error_f=1.0,
        ),
        _row(
            run_date=date(2025, 2, 28),
            lead_day=1,
            target_date=date(2025, 3, 1),
            station="KSEA",
            error_f=1.0,
        ),
    ]
    result = score(pl.DataFrame(rows), by=("month", "season"), n_boot=5)

    by_month = {row["month"]: row["season"] for row in result.iter_rows(named=True)}
    assert by_month[12] == "DJF"
    assert by_month[2] == "DJF"
    assert by_month[3] == "MAM"


# -- 6b. block bootstrap percentile pinned to the nominal alpha --------------


def test_block_bootstrap_ci_matches_numpy_quantile_at_nominal_alpha() -> None:
    """Direct, deterministic pin on the percentile endpoints.

    The stochastic coverage tests (7, 8) can pass under a systematically
    wrong percentile choice -- e.g. reading off (alpha/2, 1 - alpha/2)
    instead of (alpha, 1 - alpha) widens every interval, which either still
    lands inside the coverage band or even improves apparent coverage. This
    test instead recomputes the replicate quantiles independently (same
    multinomial draws, same seed) and pins the CI to exactly
    numpy.quantile(replicates, [0.025, 0.975]) for ci_level=0.95.
    """
    numerator = np.array([1.0, -2.0, 3.0, 0.5, -1.5])
    denominator = np.ones_like(numerator)
    block_ids = np.array([0, 1, 2, 3, 4])
    n_boot = 500
    ci_level = 0.95

    lo, hi = _block_bootstrap(
        numerator,
        denominator,
        block_ids,
        np.random.default_rng(12345),
        n_boot,
        ci_level,
    )

    n_blocks = block_ids.size
    weights = np.random.default_rng(12345).multinomial(
        n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=n_boot
    )
    replicates = (weights @ numerator) / (weights @ denominator)
    expected_lo, expected_hi = np.quantile(replicates, [0.025, 0.975])

    assert lo == pytest.approx(expected_lo, abs=1e-12)
    assert hi == pytest.approx(expected_hi, abs=1e-12)


# -- 7. coverage: the headline test ------------------------------------------


def _iid_day_effect_frame(
    sim_seed: object, mu: float, d_dates: int, s_stations: int
) -> pl.DataFrame:
    """error[d, s] = mu + day_effect[d] + noise[d, s]; one source/var/lead."""
    rng = np.random.default_rng(sim_seed)
    day_effect = rng.normal(0.0, 1.5, size=d_dates)
    noise = rng.normal(0.0, 1.0, size=(d_dates, s_stations))
    start = date(2025, 1, 1)

    stations = []
    run_dates = []
    target_dates = []
    errors = []
    for d in range(d_dates):
        run_date_ = start + timedelta(days=d)
        for s in range(s_stations):
            stations.append(f"S{s:02d}")
            run_dates.append(run_date_)
            target_dates.append(run_date_ + timedelta(days=1))
            errors.append(mu + day_effect[d] + noise[d, s])

    n = d_dates * s_stations
    return pl.DataFrame(
        {
            "station": stations,
            "run_date": run_dates,
            "lead_day": [1] * n,
            "variable": ["max"] * n,
            "source": ["raw_nbm"] * n,
            "target_date": target_dates,
            "scorable": [True] * n,
            "error_f": errors,
        }
    )


def test_date_block_coverage_beats_row_level() -> None:
    master_seed = 20260927001
    n_sims = 200
    mu = 0.7
    n_boot = 500
    ci_level = 0.95

    date_block_hits = 0
    row_level_hits = 0

    start = time.perf_counter()
    for i in range(n_sims):
        frame = _iid_day_effect_frame(
            sim_seed=[master_seed, i], mu=mu, d_dates=60, s_stations=40
        )

        # One issuance date per block: the day effects here are iid, so this
        # isolates "resample dates, not rows" from the block-length choice
        # (#31), which a 60-date slice cannot carry at the default of 14.
        result = score(frame, n_boot=n_boot, ci_level=ci_level, block_days=1)
        row = result.row(0, named=True)
        if row["bias_lo"] is not None and row["bias_lo"] <= mu <= row["bias_hi"]:
            date_block_hits += 1

        error_arr = frame["error_f"].to_numpy()
        row_ids = np.arange(error_arr.size)
        row_rng = np.random.default_rng([master_seed, i, 1])
        lo, hi = _block_bootstrap(
            error_arr, np.ones_like(error_arr), row_ids, row_rng, n_boot, ci_level
        )
        if lo is not None and lo <= mu <= hi:
            row_level_hits += 1
    elapsed = time.perf_counter() - start

    date_block_coverage = date_block_hits / n_sims
    row_level_coverage = row_level_hits / n_sims

    message = (
        f"date_block_coverage={date_block_coverage:.3f} "
        f"row_level_coverage={row_level_coverage:.3f} runtime={elapsed:.2f}s"
    )
    print(message)
    assert 0.90 <= date_block_coverage <= 0.99, message
    assert row_level_coverage < 0.75, message


# -- 8. block length on serially correlated data -----------------------------


def _ar1_day_effect_frame(
    sim_seed: object,
    mu: float,
    d_dates: int,
    s_stations: int,
    rho: float,
) -> pl.DataFrame:
    rng = np.random.default_rng(sim_seed)
    innovation_sd = 1.5 * (1 - rho**2) ** 0.5
    day_effect = np.zeros(d_dates)
    day_effect[0] = rng.normal(0.0, 1.5)
    for d in range(1, d_dates):
        day_effect[d] = rho * day_effect[d - 1] + rng.normal(0.0, innovation_sd)
    noise = rng.normal(0.0, 1.0, size=(d_dates, s_stations))
    start = date(2025, 1, 1)

    stations = []
    run_dates = []
    target_dates = []
    errors = []
    for d in range(d_dates):
        run_date_ = start + timedelta(days=d)
        for s in range(s_stations):
            stations.append(f"S{s:02d}")
            run_dates.append(run_date_)
            target_dates.append(run_date_ + timedelta(days=1))
            errors.append(mu + day_effect[d] + noise[d, s])

    n = d_dates * s_stations
    return pl.DataFrame(
        {
            "station": stations,
            "run_date": run_dates,
            "lead_day": [1] * n,
            "variable": ["max"] * n,
            "source": ["raw_nbm"] * n,
            "target_date": target_dates,
            "scorable": [True] * n,
            "error_f": errors,
        }
    )


def test_block_days_seven_beats_block_days_one_on_ar1_data() -> None:
    master_seed = 20260927002
    n_sims = 200
    mu = 0.7
    n_boot = 500
    ci_level = 0.95
    rho = 0.8

    hits_block1 = 0
    hits_block7 = 0
    for i in range(n_sims):
        frame = _ar1_day_effect_frame(
            sim_seed=[master_seed, i],
            mu=mu,
            d_dates=120,
            s_stations=40,
            rho=rho,
        )
        r1 = score(frame, n_boot=n_boot, ci_level=ci_level, block_days=1).row(
            0, named=True
        )
        if r1["bias_lo"] is not None and r1["bias_lo"] <= mu <= r1["bias_hi"]:
            hits_block1 += 1

        r7 = score(frame, n_boot=n_boot, ci_level=ci_level, block_days=7).row(
            0, named=True
        )
        if r7["bias_lo"] is not None and r7["bias_lo"] <= mu <= r7["bias_hi"]:
            hits_block7 += 1

    coverage1 = hits_block1 / n_sims
    coverage7 = hits_block7 / n_sims
    message = (
        f"block_days=1 coverage={coverage1:.3f} block_days=7 coverage={coverage7:.3f}"
    )
    print(message)
    assert coverage7 >= coverage1 + 0.08, message


# -- 9. flags at boundaries ---------------------------------------------------


def test_min_sample_flag_boundary() -> None:
    def _frame(n_dates: int) -> pl.DataFrame:
        records = []
        for d in range(n_dates):
            run_date_ = date(2025, 1, 1) + timedelta(days=d)
            for s in range(10):
                records.append(
                    _row(
                        station=f"S{s:02d}",
                        run_date=run_date_,
                        target_date=run_date_ + timedelta(days=1),
                        error_f=1.0,
                    )
                )
        return pl.DataFrame(records)

    at_threshold = score(_frame(MIN_SAMPLE_DATES), n_boot=5).row(0, named=True)
    below_threshold = score(_frame(MIN_SAMPLE_DATES - 1), n_boot=5).row(0, named=True)

    assert at_threshold["min_sample_flag"] is False
    assert below_threshold["min_sample_flag"] is True


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (-0.1, 0.3, True),
        (0.0, 0.3, True),
        (0.05, 0.3, False),
        (-0.3, -0.01, False),
    ],
)
def test_no_detectable_bias_predicate(lo: float, hi: float, expected: bool) -> None:
    assert _no_detectable_bias(lo, hi) is expected


def test_no_detectable_bias_true_when_ci_undefined() -> None:
    assert _no_detectable_bias(None, None) is True


def test_single_block_slice_has_null_ci_and_no_detectable_bias() -> None:
    rows = [
        _row(run_date=date(2025, 1, 1), target_date=date(2025, 1, 2), error_f=5.0),
        _row(
            run_date=date(2025, 1, 1),
            target_date=date(2025, 1, 2),
            station="KDEN",
            error_f=6.0,
        ),
    ]
    result = score(pl.DataFrame(rows), n_boot=50).row(0, named=True)

    assert result["bias_lo"] is None
    assert result["bias_hi"] is None
    assert result["mae_lo"] is None
    assert result["mae_hi"] is None
    assert result["no_detectable_bias"] is True


# -- 10. determinism / stability ---------------------------------------------


def test_score_is_deterministic_across_calls() -> None:
    frame = _random_conservation_frame()
    first = score(frame, by=("station",), n_boot=100)
    second = score(frame, by=("station",), n_boot=100)
    assert first.equals(second)


def test_adding_a_station_leaves_other_stations_slice_bit_identical() -> None:
    rng = np.random.default_rng(7)
    station_a_rows = [
        _row(
            station="KPHX",
            run_date=date(2025, 1, 1) + timedelta(days=d),
            target_date=date(2025, 1, 2) + timedelta(days=d),
            error_f=float(rng.normal(0, 2.0)),
        )
        for d in range(35)
    ]
    only_a = score(pl.DataFrame(station_a_rows), by=("station",), n_boot=200)

    station_b_rows = [
        _row(
            station="KDEN",
            run_date=date(2025, 1, 1) + timedelta(days=d),
            target_date=date(2025, 1, 2) + timedelta(days=d),
            error_f=float(rng.normal(0, 2.0)),
        )
        for d in range(35)
    ]
    a_and_b = score(
        pl.DataFrame(station_a_rows + station_b_rows), by=("station",), n_boot=200
    )

    a_only_row = only_a.filter(pl.col("station") == "KPHX").row(0, named=True)
    a_with_b_row = a_and_b.filter(pl.col("station") == "KPHX").row(0, named=True)
    assert a_only_row == a_with_b_row


# -- 11. validation ------------------------------------------------------------


def test_unknown_by_dimension_raises() -> None:
    with pytest.raises(ValueError, match="unknown by dimension"):
        score(_known_answer_frame(), by=("hemisphere",))


def test_missing_required_column_raises() -> None:
    frame = _known_answer_frame().drop("error_f")
    with pytest.raises(ValueError, match="error_f"):
        score(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_boot": 0},
        {"n_boot": -1},
        {"ci_level": 0.0},
        {"ci_level": 1.0},
        {"ci_level": -0.5},
        {"block_days": 0},
        {"block_days": -1},
    ],
)
def test_bad_parameters_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        score(_known_answer_frame(), **kwargs)


def test_intervals_match_golden_values_across_processes() -> None:
    # A rebuild must reproduce published intervals exactly (PRD user story 32).
    # Pinned literals catch any seeding that is only stable within one process:
    # Python's hash() is salted per process (pytest randomizes PYTHONHASHSEED),
    # so a hash()-seeded slice RNG breaks these on almost every run. Values were
    # identical under PYTHONHASHSEED=1 and =2; they change only if numpy's
    # Generator stream changes (numpy is pinned in uv.lock).
    rows = []
    for d in range(10):
        run_date = date(2025, 7, 1) + timedelta(days=d)
        for station, offset in (("KPHX", 0.5), ("KORD", -0.25)):
            rows.append(
                {
                    "station": station,
                    "run_date": run_date,
                    "lead_day": 1,
                    "variable": "max",
                    "source": "raw_nbm",
                    "target_date": run_date + timedelta(days=1),
                    "scorable": True,
                    "error_f": offset + ((d * 7) % 5 - 2) * 0.5,
                }
            )

    # Pinned to one-date blocks: these literals test cross-process seeding,
    # not the block-length default, which #31 moved to 14.
    out = score(pl.DataFrame(rows), by=("station",), n_boot=200, block_days=1)

    got = out.select("station", "bias_lo", "bias_hi", "mae_lo", "mae_hi").rows()
    expected = [
        ("KORD", -0.7, 0.15, 0.45, 0.9),
        ("KPHX", 0.09875, 0.9, 0.4, 1.0),
    ]
    for got_row, expected_row in zip(got, expected, strict=True):
        assert got_row[0] == expected_row[0]
        assert got_row[1:] == pytest.approx(expected_row[1:], abs=1e-12)


def test_block_days_default_is_the_measured_value_from_issue_31() -> None:
    """`BLOCK_DAYS` is a measured quantity, not a guess. Issue #31's
    pre-registered rule (docs/analysis/2026-09-27-block-length/PREREG.md)
    gave ceil(13.18) = 14 on 12 stations over 2025 -- the NBM max lead-2
    cross-station mean error -- and the fitted-AR coverage replay passed at
    14 (worst series 0.9075 >= 0.90) against 0.6475 at the old default of 1
    (RESULTS.md). Change it only by re-running that measurement."""
    import inspect

    from weather_forecast_audit import scoring

    assert scoring.BLOCK_DAYS == 14
    assert inspect.signature(score).parameters["block_days"].default == 14
