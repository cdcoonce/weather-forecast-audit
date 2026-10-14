"""Unit tests for the export module: the plain-language summary, per-stat
rounding, and the schema/lock versioning guard.

No DuckDB, no dbt: these exercise pure functions against hand-built stat
lists. `tests/integration/test_export_tracer.py` covers the DB-backed
export and schema validation against the tracer fixture.
"""

import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from weather_forecast_audit import export

pytestmark = pytest.mark.unit


def _stat(
    *,
    variable: str = "max",
    lead_day: int = 1,
    season: str = "JJA",
    n: int = 100,
    n_dates: int = 40,
    bias_f: float | None = None,
    bias_lo_f: float | None = None,
    bias_hi_f: float | None = None,
    mae_f: float | None = 1.0,
    min_sample_flag: bool = False,
    no_detectable_bias: bool = False,
) -> dict[str, object]:
    if bias_lo_f is None and bias_f is not None and not no_detectable_bias:
        bias_lo_f = bias_f - 0.5
    if bias_hi_f is None and bias_f is not None and not no_detectable_bias:
        bias_hi_f = bias_f + 0.5
    return {
        "variable": variable,
        "lead_day": lead_day,
        "season": season,
        "n": n,
        "n_dates": n_dates,
        "bias_f": bias_f,
        "bias_lo_f": bias_lo_f,
        "bias_hi_f": bias_hi_f,
        "mae_f": mae_f,
        "min_sample_flag": min_sample_flag,
        "no_detectable_bias": no_detectable_bias,
    }


# -- the plain-language summary -------------------------------------------


def test_summary_states_number_and_direction_when_forecast_runs_cold() -> None:
    stats = [_stat(variable="max", season="JJA", bias_f=-1.4)]
    summary = export.build_summary(stats)
    assert summary["significant"] is True
    assert summary["text"] == "Highs run 1.4°F cold in summer (day-ahead forecasts)."
    assert summary["variable"] == "max"
    assert summary["lead_day"] == 1
    assert summary["season"] == "JJA"
    assert summary["bias_f"] == -1.4


def test_summary_states_number_and_direction_when_forecast_runs_warm() -> None:
    stats = [_stat(variable="min", season="DJF", bias_f=2.3)]
    summary = export.build_summary(stats)
    assert summary["significant"] is True
    assert summary["text"] == "Lows run 2.3°F warm in winter (day-ahead forecasts)."


def test_summary_ignores_lead_2_and_lead_3_slices() -> None:
    stats = [
        _stat(variable="max", lead_day=2, season="JJA", bias_f=5.0),
        _stat(variable="max", lead_day=1, season="DJF", bias_f=1.0),
    ]
    summary = export.build_summary(stats)
    assert summary["lead_day"] == 1
    assert summary["season"] == "DJF"


def test_summary_no_candidate_but_some_lead1_slice_has_enough_sample() -> None:
    stats = [
        _stat(bias_f=0.1, no_detectable_bias=True, min_sample_flag=False),
        _stat(
            variable="min",
            bias_f=None,
            no_detectable_bias=True,
            min_sample_flag=True,
        ),
    ]
    summary = export.build_summary(stats)
    assert summary["significant"] is False
    assert summary["text"] == "No detectable bias in day-ahead forecasts."
    assert set(summary) == {"text", "significant"}


def test_summary_not_enough_data_when_every_lead1_slice_is_min_sample_flagged() -> None:
    stats = [
        _stat(bias_f=None, no_detectable_bias=True, min_sample_flag=True),
        _stat(
            variable="min",
            bias_f=None,
            no_detectable_bias=True,
            min_sample_flag=True,
        ),
    ]
    summary = export.build_summary(stats)
    assert summary["significant"] is False
    assert summary["text"] == "Not enough data yet."


def test_summary_not_enough_data_when_city_has_no_stats_at_all() -> None:
    summary = export.build_summary([])
    assert summary == {"text": "Not enough data yet.", "significant": False}


def test_summary_excludes_min_sample_flagged_candidates_even_if_biggest_bias() -> None:
    stats = [
        _stat(variable="max", season="JJA", bias_f=10.0, min_sample_flag=True),
        _stat(variable="min", season="DJF", bias_f=1.0),
    ]
    summary = export.build_summary(stats)
    assert summary["variable"] == "min"
    assert summary["bias_f"] == 1.0


def test_summary_tie_break_prefers_max_variable_over_min() -> None:
    stats = [
        _stat(variable="min", season="JJA", bias_f=2.0),
        _stat(variable="max", season="DJF", bias_f=-2.0),
    ]
    summary = export.build_summary(stats)
    assert summary["variable"] == "max"
    assert summary["season"] == "DJF"


def test_summary_tie_break_prefers_earlier_season_order() -> None:
    stats = [
        _stat(variable="max", season="JJA", bias_f=2.0),
        _stat(variable="max", season="MAM", bias_f=-2.0),
    ]
    summary = export.build_summary(stats)
    assert summary["season"] == "MAM"


def test_summary_rounds_to_slightly_warm_with_no_number() -> None:
    stats = [_stat(variable="max", season="SON", bias_f=0.04)]
    summary = export.build_summary(stats)
    assert summary["significant"] is True
    assert summary["text"] == "Highs run slightly warm in fall (day-ahead forecasts)."
    assert "1" not in summary["text"] and "0" not in summary["text"]
    assert summary["bias_f"] == 0.04


def test_summary_rounds_to_slightly_cold_with_no_number() -> None:
    stats = [_stat(variable="min", season="SON", bias_f=-0.04)]
    summary = export.build_summary(stats)
    assert summary["text"] == "Lows run slightly cold in fall (day-ahead forecasts)."


# -- per-stat rounding at export -------------------------------------------


def test_round_stat_rounds_floats_to_two_decimals() -> None:
    stat = _stat(bias_f=1.23456, bias_lo_f=0.111, bias_hi_f=2.999, mae_f=3.14159)
    rounded = export.round_stat(stat)
    assert rounded["bias_f"] == 1.23
    assert rounded["bias_lo_f"] == 0.11
    assert rounded["bias_hi_f"] == 3.0
    assert rounded["mae_f"] == 3.14


def test_round_stat_preserves_null_floats() -> None:
    stat = _stat(bias_f=None, bias_lo_f=None, bias_hi_f=None, mae_f=None)
    rounded = export.round_stat(stat)
    assert rounded["bias_f"] is None
    assert rounded["bias_lo_f"] is None
    assert rounded["bias_hi_f"] is None
    assert rounded["mae_f"] is None


def test_round_stat_does_not_touch_flags_or_counts() -> None:
    stat = _stat(n=17, n_dates=5, min_sample_flag=True, no_detectable_bias=True)
    rounded = export.round_stat(stat)
    assert rounded["n"] == 17
    assert rounded["n_dates"] == 5
    assert rounded["min_sample_flag"] is True
    assert rounded["no_detectable_bias"] is True


def test_rounding_a_borderline_bias_does_not_flip_no_detectable_bias() -> None:
    """A bias whose CI barely excludes 0 rounds to the same 2-decimal value
    whether or not it is flagged significant -- the flag must come from the
    unrounded CI, never be re-derived from the rounded display number."""
    borderline_significant = _stat(
        bias_f=0.006, bias_lo_f=0.0001, bias_hi_f=0.02, no_detectable_bias=False
    )
    borderline_not_significant = _stat(
        bias_f=0.006, bias_lo_f=-0.0001, bias_hi_f=0.02, no_detectable_bias=True
    )
    rounded_a = export.round_stat(borderline_significant)
    rounded_b = export.round_stat(borderline_not_significant)
    assert rounded_a["bias_f"] == rounded_b["bias_f"] == 0.01
    assert rounded_a["no_detectable_bias"] is False
    assert rounded_b["no_detectable_bias"] is True


# -- sign-preserving endpoint rounding (the exported data must never -------
# -- contradict its own no_detectable_bias flag) ----------------------------


def _no_detectable_bias_from_exported(stat: Mapping[str, object]) -> bool:
    lo, hi = stat["bias_lo_f"], stat["bias_hi_f"]
    assert lo is not None and hi is not None
    return lo <= 0 <= hi  # type: ignore[operator]


def _assert_stat_is_internally_consistent(rounded: Mapping[str, object]) -> None:
    """The exported (rounded) `no_detectable_bias` must agree with what a
    reader would derive from the exported (rounded) endpoints, and the
    exported point estimate must lie within the exported endpoints."""
    if rounded["bias_lo_f"] is None or rounded["bias_hi_f"] is None:
        return
    assert rounded["no_detectable_bias"] == _no_detectable_bias_from_exported(rounded)
    if rounded["bias_f"] is not None:
        assert rounded["bias_lo_f"] <= rounded["bias_f"] <= rounded["bias_hi_f"]  # type: ignore[operator]


def test_real_world_case_a_hi_barely_below_zero_rounds_toward_excluded_side() -> None:
    # Real KPHX case (min, lead 3, SON): the unrounded hi was slightly below
    # 0 and plain-rounded to 0.0, contradicting no_detectable_bias: false.
    stat = _stat(bias_f=-0.34, bias_lo_f=-0.6523, bias_hi_f=-0.0041)
    rounded = export.round_stat(stat)
    assert rounded["bias_lo_f"] == -0.65
    assert rounded["bias_hi_f"] == -0.01
    assert rounded["bias_f"] == -0.34
    _assert_stat_is_internally_consistent(rounded)


def test_mirror_case_a_lo_barely_above_zero_rounds_toward_the_excluded_side() -> None:
    stat = _stat(bias_f=0.34, bias_lo_f=0.0032, bias_hi_f=0.53)
    rounded = export.round_stat(stat)
    assert rounded["bias_lo_f"] == 0.01
    assert rounded["bias_hi_f"] == 0.53
    assert rounded["bias_f"] == 0.34
    _assert_stat_is_internally_consistent(rounded)


def test_point_estimate_rounding_that_would_escape_the_interval_is_clamped() -> None:
    # bias_f itself rounds to 0.0 (spans-zero-looking) even though the CI
    # excludes 0 on the negative side -- bias_f must be pulled back inside
    # the (also sign-preservingly rounded) interval, never left at 0.0.
    stat = _stat(bias_f=-0.001, bias_lo_f=-0.2, bias_hi_f=-0.0005)
    rounded = export.round_stat(stat)
    assert rounded["bias_hi_f"] == -0.01
    assert rounded["bias_f"] <= rounded["bias_hi_f"]
    _assert_stat_is_internally_consistent(rounded)


@pytest.mark.parametrize(
    ("bias_f", "bias_lo_f", "bias_hi_f", "no_detectable_bias"),
    [
        (-0.34, -0.6523, -0.0041, False),  # real KPHX case
        (0.34, 0.0032, 0.53, False),  # mirror
        (0.006, -0.0001, 0.02, True),  # spans zero, barely
        (0.0, -0.0001, 0.0001, True),  # spans zero, both tiny
        (-0.0, 0.0, 0.0, True),  # exactly zero everywhere
        (1.2345, 0.9, 1.6, False),  # ordinary, far from zero
        (-1.2345, -1.6, -0.9, False),  # ordinary, far from zero, negative
    ],
)
def test_rounding_never_breaks_the_spans_zero_invariant(
    bias_f: float, bias_lo_f: float, bias_hi_f: float, no_detectable_bias: bool
) -> None:
    stat = _stat(
        bias_f=bias_f,
        bias_lo_f=bias_lo_f,
        bias_hi_f=bias_hi_f,
        no_detectable_bias=no_detectable_bias,
    )
    rounded = export.round_stat(stat)
    _assert_stat_is_internally_consistent(rounded)


# -- schema / lock versioning guard ----------------------------------------


def test_schema_matches_lock_for_current_version() -> None:
    lock = export.load_lock()
    current = lock.get(export.SCHEMA_VERSION)
    assert current is not None, (
        f"export_schema/lock.json has no entry for SCHEMA_VERSION "
        f"{export.SCHEMA_VERSION!r}. If you changed a schema on disk, bump "
        f"SCHEMA_VERSION in export.py and add a new lock.json entry with "
        f"the new hashes (see export.canonical_hash)."
    )
    for name, path in export.schema_files().items():
        on_disk = json.loads(path.read_text())
        actual_hash = export.canonical_hash(on_disk)
        assert actual_hash == current[name], (
            f"{name} on disk does not match export_schema/lock.json's "
            f"{export.SCHEMA_VERSION!r} entry. If this edit was intentional, "
            f"bump SCHEMA_VERSION in export.py and add a *new* lock.json "
            f"entry for the new version -- never edit a schema without "
            f"bumping the version, and never edit an existing lock entry."
        )


# -- scorecard stats: the same sign-preserving rounding, for bias AND skill --


def _scorecard_stat(**overrides: object) -> dict[str, object]:
    stat: dict[str, object] = {
        "variable": "max",
        "lead_day": 1,
        "source": "baseline",
        "n": 100,
        "n_dates": 30,
        "n_blocks": 3,
        "mae_f": 2.0,
        "mae_lo_f": 1.5,
        "mae_hi_f": 2.5,
        "bias_f": 0.5,
        "bias_lo_f": 0.1,
        "bias_hi_f": 0.9,
        "skill": 0.1,
        "skill_lo": 0.02,
        "skill_hi": 0.2,
        "min_sample_flag": True,
        "no_detectable_bias": False,
    }
    stat.update(overrides)
    return stat


def test_round_scorecard_stat_rounds_every_float_and_keeps_counts_and_flags() -> None:
    rounded = export.round_scorecard_stat(
        _scorecard_stat(
            mae_f=2.34567,
            mae_lo_f=1.2349,
            mae_hi_f=3.5051,
            bias_f=0.51234,
            bias_lo_f=0.11111,
            bias_hi_f=0.98765,
            skill=0.123456,
            skill_lo=0.0234,
            skill_hi=0.2678,
        )
    )
    assert rounded["mae_f"] == 2.35
    assert rounded["mae_lo_f"] == 1.23
    assert rounded["mae_hi_f"] == 3.51
    assert rounded["bias_f"] == 0.51
    assert rounded["bias_lo_f"] == 0.11
    assert rounded["bias_hi_f"] == 0.99
    assert rounded["skill"] == 0.123
    assert rounded["skill_lo"] == 0.023
    assert rounded["skill_hi"] == 0.268
    assert rounded["n"] == 100
    assert rounded["n_blocks"] == 3
    assert rounded["min_sample_flag"] is True
    assert rounded["no_detectable_bias"] is False


def test_round_scorecard_stat_preserves_nulls() -> None:
    rounded = export.round_scorecard_stat(
        _scorecard_stat(
            mae_lo_f=None, mae_hi_f=None, bias_f=None, bias_lo_f=None,
            bias_hi_f=None, skill=None, skill_lo=None, skill_hi=None,
            no_detectable_bias=True,
        )
    )
    for name in (
        "mae_lo_f", "mae_hi_f", "bias_f", "bias_lo_f", "bias_hi_f",
        "skill", "skill_lo", "skill_hi",
    ):
        assert rounded[name] is None


@pytest.mark.parametrize(
    ("skill", "lo", "hi"),
    [
        (0.0004, 0.0001, 0.0009),  # excludes zero on the positive side
        (-0.0004, -0.0009, -0.0001),  # ... and on the negative side
        (0.2, -0.0004, 0.4),  # spans zero, barely
        (0.0, -0.0001, 0.0001),  # spans zero, both tiny
        (0.0, 0.0, 0.0),  # the reference source against itself
        (0.15, 0.05, 0.25),  # ordinary
        (-0.0001, -0.2, -0.00004),  # point rounds to 0 while the CI excludes it
    ],
)
def test_skill_rounding_never_changes_whether_the_interval_excludes_zero(
    skill: float, lo: float, hi: float
) -> None:
    rounded = export.round_scorecard_stat(
        _scorecard_stat(skill=skill, skill_lo=lo, skill_hi=hi)
    )
    r_lo, r_hi, r_point = rounded["skill_lo"], rounded["skill_hi"], rounded["skill"]
    assert (lo <= 0 <= hi) == (r_lo <= 0 <= r_hi)  # type: ignore[operator]
    assert r_lo <= r_point <= r_hi  # type: ignore[operator]


@pytest.mark.parametrize(
    ("bias_f", "lo", "hi", "spans"),
    [
        (-0.34, -0.6523, -0.0041, False),
        (0.34, 0.0032, 0.53, False),
        (0.006, -0.0001, 0.02, True),
    ],
)
def test_scorecard_bias_rounding_agrees_with_no_detectable_bias(
    bias_f: float, lo: float, hi: float, spans: bool
) -> None:
    rounded = export.round_scorecard_stat(
        _scorecard_stat(
            bias_f=bias_f, bias_lo_f=lo, bias_hi_f=hi, no_detectable_bias=spans
        )
    )
    assert (rounded["bias_lo_f"] <= 0 <= rounded["bias_hi_f"]) == spans  # type: ignore[operator]
    assert rounded["bias_lo_f"] <= rounded["bias_f"] <= rounded["bias_hi_f"]  # type: ignore[operator]


def test_scorecard_point_estimate_is_clamped_into_the_rounded_interval() -> None:
    """A percentile interval need not contain its own point estimate; the
    exported point is clamped so it never sits outside the exported interval."""
    above = export.round_scorecard_stat(
        _scorecard_stat(bias_f=1.0, bias_lo_f=0.2, bias_hi_f=0.9, skill=0.5,
                        skill_lo=0.1, skill_hi=0.3)
    )
    assert above["bias_f"] == 0.9
    assert above["skill"] == 0.3
    below = export.round_scorecard_stat(
        _scorecard_stat(bias_f=-0.5, bias_lo_f=0.2, bias_hi_f=0.9, skill=-0.5,
                        skill_lo=0.1, skill_hi=0.3)
    )
    assert below["bias_f"] == 0.2
    assert below["skill"] == 0.1


def test_skill_nudge_off_zero_is_one_unit_of_the_third_decimal() -> None:
    positive = export.round_scorecard_stat(
        _scorecard_stat(skill=0.0004, skill_lo=0.0001, skill_hi=0.0009)
    )
    assert (positive["skill_lo"], positive["skill"]) == (0.001, 0.001)
    assert positive["skill_hi"] == 0.001
    negative = export.round_scorecard_stat(
        _scorecard_stat(skill=-0.0004, skill_lo=-0.0009, skill_hi=-0.0001)
    )
    assert (negative["skill_lo"], negative["skill"], negative["skill_hi"]) == (
        -0.001, -0.001, -0.001,
    )


# -- typical miss (lead-1 MAE pooled over seasons) -------------------------


def test_typical_miss_pools_mae_weighted_by_row_count_not_a_simple_average() -> None:
    """The pooled MAE must be the row-count-weighted mean, not the unweighted
    mean of the season MAEs -- these differ whenever season `n` differs."""
    stats = [
        _stat(variable="max", lead_day=1, season="DJF", n=100, mae_f=2.0),
        _stat(variable="max", lead_day=1, season="JJA", n=300, mae_f=4.0),
    ]
    typical_miss = export.build_typical_miss(stats)
    weighted = (100 * 2.0 + 300 * 4.0) / 400  # 3.5
    unweighted = (2.0 + 4.0) / 2  # 3.0 -- the wrong answer a mutation would give
    assert typical_miss["max"] == pytest.approx(weighted)
    assert typical_miss["max"] != pytest.approx(unweighted)


def test_typical_miss_is_null_when_variable_has_no_lead1_slice() -> None:
    stats = [_stat(variable="min", lead_day=2, season="DJF", n=50, mae_f=2.0)]
    typical_miss = export.build_typical_miss(stats)
    assert typical_miss["max"] is None
    assert typical_miss["min"] is None


def test_typical_miss_is_null_when_every_lead1_slice_is_min_sample_flagged() -> None:
    stats = [
        _stat(
            variable="max", lead_day=1, season="DJF", n=5, mae_f=1.0,
            min_sample_flag=True,
        ),
    ]
    typical_miss = export.build_typical_miss(stats)
    assert typical_miss["max"] is None


def test_typical_miss_pools_across_all_lead1_seasons_including_flagged_ones() -> None:
    """Once at least one season has enough sample, the pooled value still
    sums over every lead-1 season slice for that variable (not only the
    unflagged ones) -- pooling only needs one season to unlock it."""
    stats = [
        _stat(
            variable="min", lead_day=1, season="DJF", n=50, mae_f=2.0,
            min_sample_flag=False,
        ),
        _stat(
            variable="min", lead_day=1, season="JJA", n=10, mae_f=5.0,
            min_sample_flag=True,
        ),
    ]
    typical_miss = export.build_typical_miss(stats)
    assert typical_miss["min"] == pytest.approx((50 * 2.0 + 10 * 5.0) / 60)


def test_typical_miss_exact_against_a_hand_computed_pooled_mae() -> None:
    stats = [
        _stat(variable="max", lead_day=1, season="DJF", n=137, mae_f=1.2345678),
        _stat(variable="max", lead_day=1, season="MAM", n=263, mae_f=2.9988776),
        # lead 2, excluded from the pool:
        _stat(variable="max", lead_day=2, season="DJF", n=999, mae_f=99.0),
        _stat(variable="min", lead_day=1, season="DJF", n=10, mae_f=0.5),
    ]
    typical_miss = export.build_typical_miss(stats)
    expected_max = (137 * 1.2345678 + 263 * 2.9988776) / (137 + 263)
    assert typical_miss["max"] == pytest.approx(expected_max, abs=1e-9)
    assert typical_miss["min"] == pytest.approx(0.5)


def test_typical_miss_ignores_lead_2_and_lead_3_slices() -> None:
    stats = [
        _stat(variable="max", lead_day=2, season="DJF", n=500, mae_f=9.0),
        _stat(variable="max", lead_day=3, season="DJF", n=500, mae_f=9.0),
    ]
    typical_miss = export.build_typical_miss(stats)
    assert typical_miss["max"] is None


# -- title-cased station labels ---------------------------------------------


def test_title_case_label_basic() -> None:
    assert export.title_case_label("MINNEAPOLIS, MN") == "Minneapolis, MN"


def test_title_case_label_slash_and_space_separated_words() -> None:
    assert (
        export.title_case_label("PHOENIX/SKY HARBOR, AZ") == "Phoenix/Sky Harbor, AZ"
    )


def test_title_case_label_keeps_trailing_state_code_upper_case() -> None:
    label = export.title_case_label("BOSTON LOGAN INTL, MA")
    assert label == "Boston Logan Intl, MA"
    state_code = label.rsplit(",", 1)[1].strip()
    assert state_code == state_code.upper()
    assert state_code == "MA"


def test_title_case_label_period_separator_real_registry_example() -> None:
    # dbt/seeds/station_registry.csv: "ST. JOHNSBURY(AMOS), VT"
    assert (
        export.title_case_label("ST. JOHNSBURY(AMOS), VT")
        == "St. Johnsbury(Amos), VT"
    )


def test_title_case_label_hyphen_separator_real_registry_example() -> None:
    # dbt/seeds/station_registry.csv: "DESERT ROCK-MERCURY NV, NV" -- the
    # state name is folded into the station name itself, directly against
    # the comma, not just the trailing tail; both "NV"s must stay upper.
    assert (
        export.title_case_label("DESERT ROCK-MERCURY NV, NV")
        == "Desert Rock-Mercury NV, NV"
    )


def test_title_case_label_apostrophe_separator_synthetic() -> None:
    # No apostrophe'd label exists in the current 573-station registry; this
    # locks in the separator rule generally.
    assert export.title_case_label("O'FALLON, MO") == "O'Fallon, MO"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ISLIP/MACARTHUR, NY", "Islip/MacArthur, NY"),
        ("LAS VEGAS/MCCARRAN, NV", "Las Vegas/McCarran, NV"),
        ("MCCOMB, MS", "McComb, MS"),
        ("MERCED/MACREADY FLD, CA", "Merced/MacReady Fld, CA"),
        ("MCCOOK, NE", "McCook, NE"),
        ("MACON/LEWIS WILSON, GA", "Macon/Lewis Wilson, GA"),
        ("MCALLEN/MILLER INTL, TX", "McAllen/Miller Intl, TX"),
        ("JACKSON/MCKELLAR, TN", "Jackson/McKellar, TN"),
        ("MCMINNVILLE MUNICIPAL AIRPORT, OR", "McMinnville Municipal Airport, OR"),
        ("YUMA MCAS, AZ", "Yuma MCAS, AZ"),
        ("SALEM/MCNARY, OR", "Salem/McNary, OR"),
        ("ATLANTA/DEKALB, GA", "Atlanta/DeKalb, GA"),
    ],
)
def test_title_case_label_known_exceptions_from_the_real_registry(
    raw: str, expected: str
) -> None:
    assert export.title_case_label(raw) == expected


# -- mid-string state codes before the trailing comma -----------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # dbt/seeds/station_registry.csv: the state name is folded into the
        # station name itself, directly against the comma -- naive casing
        # lower-cased this mid-string occurrence too (the bug).
        ("DESERT ROCK-MERCURY NV, NV", "Desert Rock-Mercury NV, NV"),
        ("WALLA WALLA WA, WA", "Walla Walla WA, WA"),
        ("BAUDETTE MN, MN", "Baudette MN, MN"),
        ("MT SHASTA CITY CA, CA", "Mt Shasta City CA, CA"),
    ],
)
def test_title_case_label_upper_cases_a_state_code_adjacent_to_the_comma(
    raw: str, expected: str
) -> None:
    assert export.title_case_label(raw) == expected


def test_title_case_label_does_not_upper_case_a_county_co_behind_a_period() -> None:
    # dbt/seeds/station_registry.csv: "ALMA/BACON CO., GA" -- "CO." is not
    # *directly* adjacent to the comma (a period sits in between), so the
    # state-code rule does not fire and naive casing stands.
    assert export.title_case_label("ALMA/BACON CO., GA") == "Alma/Bacon Co., GA"


def test_lock_entries_are_append_only() -> None:
    lock = export.load_lock()
    # 1.0.0 is the first published version; it must never be removed, and
    # every one of its schema files must still have a recorded hash. This
    # asserts a subset plus invariants, never a pinned total entry count,
    # because the lock is an append-only ledger new versions get added to.
    assert "1.0.0" in lock
    expected_files = {
        "manifest.schema.json",
        "city_index.schema.json",
        "city.schema.json",
        "completeness.schema.json",
    }
    assert expected_files <= lock["1.0.0"].keys()
    for digest in lock["1.0.0"].values():
        assert isinstance(digest, str)
        assert len(digest) == 64  # sha256 hex digest length
        int(digest, 16)  # every entry is valid hex


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # "CO" before the comma is County in these...
        ("AKRON/WASHINGTON CO, CO", "Akron/Washington Co, CO"),
        ("ANNISTON/CALHOUN CO, AL", "Anniston/Calhoun Co, AL"),
        ("BLUEFIELD/MERCER CO, WV", "Bluefield/Mercer Co, WV"),
        ("CORTEZ/MONTEZUMA CO, CO", "Cortez/Montezuma Co, CO"),
        ("DURANGO/LA PLATA CO, CO", "Durango/La Plata Co, CO"),
        ("MOSES LAKE/GRANT CO, WA", "Moses Lake/Grant Co, WA"),
        ("ROCHESTER/MONROE CO, NY", "Rochester/Monroe Co, NY"),
        # ...and Colorado in these, so no token rule can decide it.
        ("FT COLLINS/LOVELAND CO, CO", "Ft Collins/Loveland CO, CO"),
        ("GREELEY  CO, CO", "Greeley CO, CO"),
        ("NORTH_LAS_VEGAS NV, NV", "North Las Vegas NV, NV"),
    ],
)
def test_title_case_label_overrides_decide_ambiguous_labels(
    raw: str, expected: str
) -> None:
    assert export.title_case_label(raw) == expected


def test_every_ambiguous_co_label_in_the_registry_has_an_override() -> None:
    # "X CO, ST" may mean County or Colorado; only a person can tell. A new
    # registry station with this shape must fail here until it is decided.
    registry = Path(__file__).resolve().parents[2] / "dbt/seeds/station_registry.csv"
    with registry.open(newline="") as handle:
        labels = {row["label"] for row in csv.DictReader(handle)}
    ambiguous = {label for label in labels if re.search(r" CO, [A-Z]{2}$", label)}
    assert ambiguous, "expected the known County/Colorado labels in the registry"
    assert ambiguous <= set(export.LABEL_OVERRIDES), sorted(
        ambiguous - set(export.LABEL_OVERRIDES)
    )


# -- contract 1.1.0 is additive: 1.0.0 stays exactly as published ------------

_LOCK_1_0_0 = {
    "city.schema.json": (
        "d119168ee7ee8205a088a52fae26bd1f08c5ecb2a34091deb22ab6f08a8a5060"
    ),
    "city_index.schema.json": (
        "1c0e879805095eedab681e228d764514043c081519f288b37b0f52940ae94bfa"
    ),
    "completeness.schema.json": (
        "3505f4dd0ec567e42ff435abcfe22b3e66c95026be6a1121924dab4cf0ab0615"
    ),
    "manifest.schema.json": (
        "06a81136cff134911347a35289db24dad334da9ae5e24b6710889512c6420305"
    ),
}


def test_current_version_is_1_1_0() -> None:
    assert export.SCHEMA_VERSION == "1.1.0"


def test_the_1_0_0_lock_entry_is_unchanged() -> None:
    assert export.load_lock()["1.0.0"] == _LOCK_1_0_0


def test_the_four_1_0_0_schema_files_are_untouched_on_disk() -> None:
    files = export.schema_files()
    for name, digest in _LOCK_1_0_0.items():
        assert export.canonical_hash(json.loads(files[name].read_text())) == digest


def test_1_1_0_adds_exactly_the_two_scorecard_schemas_to_the_1_0_0_set() -> None:
    lock = export.load_lock()
    assert set(lock["1.1.0"]) == set(lock["1.0.0"]) | {
        "scorecard.schema.json",
        "scorecard_city.schema.json",
    }
    for name, digest in _LOCK_1_0_0.items():
        assert lock["1.1.0"][name] == digest
