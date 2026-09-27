"""Unit tests for the export module: the plain-language summary, per-stat
rounding, and the schema/lock versioning guard.

No DuckDB, no dbt: these exercise pure functions against hand-built stat
lists. `tests/integration/test_export_tracer.py` covers the DB-backed
export and schema validation against the tracer fixture.
"""

import json

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
    # dbt/seeds/station_registry.csv: "DESERT ROCK-MERCURY NV, NV"
    assert (
        export.title_case_label("DESERT ROCK-MERCURY NV, NV")
        == "Desert Rock-Mercury Nv, NV"
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
