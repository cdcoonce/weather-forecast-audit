"""Tests for the LightGBM challenger corrector (issue #19).

`ChallengerModel` is a `Model` (walk-forward Protocol) that predicts the
signed error `forecast_f - observed_f` from guidance-side features, station
metadata and lagged errors of the same (station, lead_day, variable) group,
and subtracts the prediction from the raw forecast. Two kinds of test live
here:

* the pure lagged-error builder (`pair_stats` + `asof_lag_features`) is
  checked against an independent brute-force reference, including the
  strictness at the issuance boundary that makes the features legal;
* the model is driven through the real `walk_forward` evaluator (issue #17),
  so the leakage guard, the optional `update` hook and the fit path are
  exercised together.
"""

import math
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time, timedelta

import lightgbm
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from weather_forecast_audit import challenger
from weather_forecast_audit.baseline import BaselineModel
from weather_forecast_audit.challenger import (
    ChallengerModel,
    asof_lag_features,
    pair_stats,
)
from weather_forecast_audit.walkforward import RawNbmModel, walk_forward

pytestmark = pytest.mark.unit

_LEADS_BY_VARIABLE = {"max": (1, 2), "min": (1, 2, 3)}
_SORT_KEYS = ["window_end_utc", "station", "runtime_utc", "lead_day", "variable"]
_PREDICT_COLUMNS = [
    "station",
    "run_date",
    "runtime_utc",
    "cycle_hour",
    "lead_day",
    "variable",
    "target_date",
    "forecast_f",
    "spread_f",
    "tmp_f",
    "window_start_utc",
    "window_end_utc",
]
_LAG_COLUMNS = ["lag_error", "lag_mean3", "lag_mean7", "lag_n30", "lag_age_days"]

ErrorFn = Callable[[np.random.Generator, int, float, str, int], float]


# -- synthetic data -----------------------------------------------------------


def _window_max(target: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time(12, 0), tzinfo=UTC)
    end = datetime.combine(target + timedelta(days=1), time(6, 0), tzinfo=UTC)
    return start, end


def _window_min(target: date) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time(0, 0), tzinfo=UTC)
    end = datetime.combine(target, time(18, 0), tzinfo=UTC)
    return start, end


def _consecutive_run_dates(start_date: date, n: int) -> list[date]:
    return [start_date + timedelta(days=i) for i in range(n)]


def _issuance(run_date_: date, hour: int = 13) -> datetime:
    return datetime.combine(run_date_, time(hour, 0), tzinfo=UTC)


def _default_error(
    rng: np.random.Generator, station_index: int, tmp: float, variable: str, lead: int
) -> float:
    return 2.0 + float(rng.normal(0.0, 1.0))


def _make_pairs(
    run_dates: Sequence[date],
    stations: Sequence[str],
    *,
    seed: int = 0,
    error_fn: ErrorFn = _default_error,
    keep: Callable[[str, date], bool] | None = None,
) -> pl.DataFrame:
    """Realistic verification pairs; `observed_f` is shared across runs/leads."""
    rng = np.random.default_rng(seed)
    target_dates = sorted(
        {
            run_date_ + timedelta(days=lead)
            for run_date_ in run_dates
            for leads in _LEADS_BY_VARIABLE.values()
            for lead in leads
        }
    )
    observed_by_key = {
        (station, target, variable): float(rng.normal(60.0, 12.0))
        for station in stations
        for target in target_dates
        for variable in _LEADS_BY_VARIABLE
    }
    records: list[dict[str, object]] = []
    for run_date_ in run_dates:
        runtime = _issuance(run_date_)
        for station_index, station in enumerate(stations):
            if keep is not None and not keep(station, run_date_):
                continue
            for variable, leads in _LEADS_BY_VARIABLE.items():
                for lead in leads:
                    target = run_date_ + timedelta(days=lead)
                    window_start, window_end = (
                        _window_max(target)
                        if variable == "max"
                        else _window_min(target)
                    )
                    observed = observed_by_key[(station, target, variable)]
                    tmp = float(rng.uniform(30.0, 90.0))
                    error = error_fn(rng, station_index, tmp, variable, lead)
                    records.append(
                        {
                            "station": station,
                            "run_date": run_date_,
                            "runtime_utc": runtime,
                            "cycle_hour": 13,
                            "lead_day": lead,
                            "variable": variable,
                            "target_date": target,
                            "forecast_f": observed + error,
                            "spread_f": 3.0 + float(rng.uniform(0.0, 1.0)),
                            "tmp_f": tmp,
                            "window_start_utc": window_start,
                            "window_end_utc": window_end,
                            "observed_f": observed,
                            "scorable": True,
                            "source": "raw_nbm",
                        }
                    )
    return pl.DataFrame(records)


def _station_features(stations: Sequence[str]) -> pl.DataFrame:
    regions = ("Southeast", "Northeast", "West")
    return pl.DataFrame(
        {
            "station": list(stations),
            "latitude": [30.0 + 2.0 * i for i in range(len(stations))],
            "longitude": [-100.0 + 3.0 * i for i in range(len(stations))],
            "elevation_m": [100 * i for i in range(len(stations))],
            "coastal_flag": [i % 2 == 0 for i in range(len(stations))],
            "climate_region": [regions[i % 3] for i in range(len(stations))],
        }
    )


def _scorable_sorted(pairs: pl.DataFrame) -> pl.DataFrame:
    return pairs.filter(pl.col("scorable")).sort(_SORT_KEYS)


def _planted_error(
    rng: np.random.Generator, station_index: int, tmp: float, variable: str, lead: int
) -> float:
    """+4 F when the NBS temperature guidance is warm, +0.3 F per station index."""
    return (
        (4.0 if tmp > 62.0 else 0.0)
        + 0.3 * station_index
        + float(rng.normal(0.0, 0.7))
    )


def _planted_pairs(
    n_days: int = 130, n_stations: int = 8, seed: int = 3
) -> pl.DataFrame:
    stations = [f"K{chr(65 + i)}{chr(65 + i)}{chr(65 + i)}" for i in range(n_stations)]
    return _make_pairs(
        _consecutive_run_dates(date(2026, 1, 1), n_days),
        stations,
        seed=seed,
        error_fn=_planted_error,
    )


def _mae(result: pl.DataFrame, pairs: pl.DataFrame, since: date) -> float:
    joined = result.filter(pl.col("run_date") >= since).join(
        pairs.select(["station", "run_date", "lead_day", "variable", "observed_f"]),
        on=["station", "run_date", "lead_day", "variable"],
        how="inner",
    )
    return float((joined["forecast_f"] - joined["observed_f"]).abs().mean())


def _predict_rows(pairs: pl.DataFrame, run_date_: date) -> pl.DataFrame:
    columns = [c for c in _PREDICT_COLUMNS if c in pairs.columns]
    return (
        pairs.filter(pl.col("run_date") == run_date_)
        .sort(["station", "lead_day", "variable"])
        .select(columns)
    )


# -- a. leakage property on the feature builder ---------------------------------


def _reference_lags(
    history: list[dict[str, object]],
    station: str,
    lead_day: int,
    variable: str,
    runtime: datetime,
) -> dict[str, float]:
    """Brute-force, loop-only restatement of the as-of lag semantics."""
    group = sorted(
        (
            r
            for r in history
            if (r["station"], r["lead_day"], r["variable"])
            == (station, lead_day, variable)
        ),
        key=lambda r: (r["window_end_utc"], r["runtime_utc"]),
    )
    known = [i for i, r in enumerate(group) if r["window_end_utc"] < runtime]
    if not known:
        return dict.fromkeys(_LAG_COLUMNS, math.nan)
    i = known[-1]
    errors = [r["forecast_f"] - r["observed_f"] for r in group]
    last3 = errors[max(0, i - 2) : i + 1]
    last7 = errors[max(0, i - 6) : i + 1]
    horizon = group[i]["window_end_utc"] - timedelta(days=30)
    return {
        "lag_error": errors[i],
        "lag_mean3": sum(last3) / len(last3),
        "lag_mean7": sum(last7) / len(last7),
        "lag_n30": float(
            sum(1 for j in range(i + 1) if group[j]["window_end_utc"] > horizon)
        ),
        "lag_age_days": (runtime - group[i]["window_end_utc"]).total_seconds()
        / 86400.0,
    }


def _builder_history_and_queries(
    seed: int = 11,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    base = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    history: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    for station in ("KAAA", "KBBB", "KCCC"):
        for lead in (1, 2):
            for variable in ("max", "min"):
                day = 0
                group_rows: list[dict[str, object]] = []
                for _ in range(70):
                    day += int(rng.integers(1, 5))  # gaps of 1-4 days: sparse
                    window_end = base + timedelta(
                        days=day, hours=int(rng.integers(0, 12))
                    )
                    observed = float(rng.normal(60.0, 10.0))
                    group_rows.append(
                        {
                            "station": station,
                            "lead_day": lead,
                            "variable": variable,
                            "runtime_utc": window_end - timedelta(days=lead, hours=5),
                            "window_end_utc": window_end,
                            "forecast_f": observed + float(rng.normal(1.0, 3.0)),
                            "observed_f": observed,
                        }
                    )
                # Exact ties on window_end_utc inside one group (different runs).
                for tied in rng.choice(len(group_rows), size=6, replace=False):
                    twin = dict(group_rows[int(tied)])
                    twin["runtime_utc"] = twin["runtime_utc"] + timedelta(hours=1)
                    twin["forecast_f"] = twin["forecast_f"] + float(rng.normal(0, 2))
                    group_rows.append(twin)
                history.extend(group_rows)

                ends = sorted(r["window_end_utc"] for r in group_rows)
                span_end = ends[-1]
                times: list[datetime] = [
                    base - timedelta(days=3),  # before everything
                    span_end + timedelta(days=2),  # after everything
                    span_end + timedelta(days=80),  # far after: stale
                ]
                for pick in rng.choice(len(ends), size=12, replace=False):
                    exact = ends[int(pick)]
                    times += [
                        exact,  # equals a close: must NOT be used
                        exact + timedelta(seconds=1),
                        exact - timedelta(seconds=1),
                    ]
                for _ in range(12):
                    times.append(
                        base + timedelta(minutes=int(rng.integers(0, day * 24 * 60)))
                    )
                queries.extend(
                    {
                        "station": station,
                        "lead_day": lead,
                        "variable": variable,
                        "runtime_utc": t,
                    }
                    for t in times
                )
    # A group that has no history at all.
    queries.append(
        {
            "station": "KZZZ",
            "lead_day": 1,
            "variable": "max",
            "runtime_utc": base + timedelta(days=30),
        }
    )
    return history, queries


def _is_same(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)


def test_lag_features_match_bruteforce_reference() -> None:
    history, queries = _builder_history_and_queries()
    history_frame = pl.DataFrame(history)
    query_frame = pl.DataFrame(queries)

    got = asof_lag_features(query_frame, pair_stats(history_frame))

    assert got.height == query_frame.height
    assert got.columns == _LAG_COLUMNS
    n_nan = 0
    for position, query in enumerate(queries):
        expected = _reference_lags(
            history,
            query["station"],
            query["lead_day"],
            query["variable"],
            query["runtime_utc"],
        )
        row = got.row(position, named=True)
        for column in _LAG_COLUMNS:
            assert _is_same(row[column], expected[column]), (position, column, query)
        n_nan += math.isnan(row["lag_error"])
    # The property is not vacuous: some queries have a latest pair and some do not.
    assert 0 < n_nan < len(queries)


def test_pairs_closing_at_or_after_runtime_are_never_used() -> None:
    history, queries = _builder_history_and_queries(seed=12)
    history_frame = pl.DataFrame(history)
    rng = np.random.default_rng(5)
    sample = [queries[int(i)] for i in rng.choice(len(queries), 20, replace=False)]

    for query in sample:
        runtime = query["runtime_utc"]
        one = pl.DataFrame([query])
        full = asof_lag_features(one, pair_stats(history_frame))

        # Dropping every pair that closes at or after the runtime changes nothing.
        past = history_frame.filter(pl.col("window_end_utc") < runtime)
        assert_frame_equal(
            full, asof_lag_features(one, pair_stats(past)), check_exact=True
        )

        # Neither does corrupting those pairs (a pair closing exactly at the
        # runtime is included in the corrupted set).
        poisoned = history_frame.with_columns(
            pl.when(pl.col("window_end_utc") >= runtime)
            .then(pl.col("forecast_f") + 1.0e6)
            .otherwise(pl.col("forecast_f"))
            .alias("forecast_f")
        )
        assert_frame_equal(
            full, asof_lag_features(one, pair_stats(poisoned)), check_exact=True
        )


def test_a_pair_closing_exactly_at_issuance_is_excluded() -> None:
    first_close = datetime(2026, 3, 1, 6, 0, tzinfo=UTC)
    second_close = datetime(2026, 3, 2, 6, 0, tzinfo=UTC)
    history = pl.DataFrame(
        {
            "station": ["KPHX", "KPHX"],
            "lead_day": [1, 1],
            "variable": ["max", "max"],
            "runtime_utc": [first_close - timedelta(days=1)] * 2,
            "window_end_utc": [first_close, second_close],
            "forecast_f": [70.0, 95.0],
            "observed_f": [68.0, 60.0],
        }
    )
    query = pl.DataFrame(
        {
            "station": ["KPHX"],
            "lead_day": [1],
            "variable": ["max"],
            "runtime_utc": [second_close],
        }
    )

    row = asof_lag_features(query, pair_stats(history)).row(0, named=True)

    assert row["lag_error"] == pytest.approx(2.0)
    assert row["lag_n30"] == 1.0
    assert row["lag_age_days"] == pytest.approx(1.0)


def test_asof_features_are_aligned_to_unsorted_queries() -> None:
    history, queries = _builder_history_and_queries(seed=13)
    stats = pair_stats(pl.DataFrame(history))
    query_frame = pl.DataFrame(queries)
    shuffled = query_frame.sample(fraction=1.0, shuffle=True, seed=1)

    forward = asof_lag_features(query_frame, stats)
    permuted = asof_lag_features(shuffled, stats)

    for position in (0, 7, 100, query_frame.height - 1):
        query = queries[position]
        expected = forward.row(position, named=True)
        match = shuffled.with_row_index("i").filter(
            (pl.col("station") == query["station"])
            & (pl.col("lead_day") == query["lead_day"])
            & (pl.col("variable") == query["variable"])
            & (pl.col("runtime_utc") == query["runtime_utc"])
        )["i"][0]
        got = permuted.row(match, named=True)
        assert all(_is_same(got[c], expected[c]) for c in _LAG_COLUMNS)


# -- b. fit path == predict path (incremental-state equivalence) ------------


def test_incremental_state_equals_one_shot_builder() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 125)
    stations = ["KAAA", "KBBB", "KCCC", "KSPR"]
    # KSPR reports for 10 days, vanishes for ~90 days, then returns: the gap
    # exceeds the 40-day retention horizon.
    pairs = _make_pairs(
        run_dates,
        stations,
        seed=9,
        keep=lambda s, d: (
            s != "KSPR" or d < date(2026, 1, 11) or d >= date(2026, 4, 12)
        ),
    )
    ordered = _scorable_sorted(pairs)
    model = ChallengerModel(_station_features(stations), min_train_rows=10**9)
    refit_days = {run_dates[30], run_dates[75]}

    previous = _issuance(run_dates[30])
    model.fit(ordered.filter(pl.col("window_end_utc") < previous))
    n_stale_groups_seen = 0
    n_compared = 0
    for run_date_ in run_dates[31:120]:
        t = _issuance(run_date_)
        if run_date_ in refit_days:
            model.fit(ordered.filter(pl.col("window_end_utc") < t))
        else:
            delta = ordered.filter(
                (pl.col("window_end_utc") >= previous) & (pl.col("window_end_utc") < t)
            )
            if delta.height:
                model.update(delta)
        previous = t

        rows = _predict_rows(pairs, run_date_)
        got = model.lag_features(rows)
        prefix = ordered.filter(pl.col("window_end_utc") < t)
        expected = asof_lag_features(rows, pair_stats(prefix))
        assert_frame_equal(got, expected, check_exact=True)
        n_compared += 1
        n_stale_groups_seen += int(
            (got["lag_age_days"].fill_nan(None) > 40.0).sum() or 0
        )

    assert n_compared > 80
    # The comparison covered stale-but-known groups (age > 40 days), whose
    # features must survive the retention trim, and the history stayed bounded.
    assert n_stale_groups_seen > 0
    assert model.history.height < ordered.height / 2


# -- c. walk_forward leakage guard with the challenger plugged in ----------------


class _CheatingChallenger(ChallengerModel):
    """The memorizing cheat bolted onto the challenger: predicts the observed
    value of any (station, target_date, variable) it has already been shown."""

    name = "challenger_cheat"

    def __init__(
        self, station_features: pl.DataFrame, *, min_train_rows: int
    ) -> None:
        super().__init__(station_features, min_train_rows=min_train_rows)
        self._memo: dict[tuple[str, date, str], float] = {}

    def _remember(self, frame: pl.DataFrame) -> None:
        for row in frame.iter_rows(named=True):
            self._memo[(row["station"], row["target_date"], row["variable"])] = row[
                "observed_f"
            ]

    def fit(self, training_pairs: pl.DataFrame) -> None:
        super().fit(training_pairs)
        self._memo = {}
        self._remember(training_pairs)

    def update(self, new_pairs: pl.DataFrame) -> None:
        super().update(new_pairs)
        self._remember(new_pairs)

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        detail = super().predict_detail(run_rows)
        values = [
            self._memo.get((row["station"], row["target_date"], row["variable"]), raw)
            for row, raw in zip(
                run_rows.iter_rows(named=True), detail["forecast_f"], strict=True
            )
        ]
        return detail.with_columns(pl.Series("forecast_f", values, dtype=pl.Float64))


def _leaky_daily_walk_forward(
    pairs: pl.DataFrame, model: ChallengerModel, start: date, end: date
) -> pl.DataFrame:
    """Reverse check: cut training off by `run_date`, not `window_end_utc < T`,
    built independently of `weather_forecast_audit.walkforward`."""
    scorable = pairs.filter(pl.col("scorable"))
    run_dates = sorted(
        pairs.filter((pl.col("run_date") >= start) & (pl.col("run_date") <= end))
        .get_column("run_date")
        .unique()
        .to_list()
    )
    outputs = []
    for run_date_ in run_dates:
        model.fit(scorable.filter(pl.col("run_date") < run_date_).sort(_SORT_KEYS))
        rows = _predict_rows(pairs, run_date_)
        detail = model.predict_detail(rows)
        outputs.append(
            rows.select("station", "run_date", "lead_day", "variable").with_columns(
                pl.Series("forecast_f", detail["forecast_f"])
            )
        )
    return pl.concat(outputs)


def test_trained_through_precedes_every_issuance() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 70)
    stations = ["KAAA", "KBBB", "KCCC"]
    pairs = _make_pairs(run_dates, stations, seed=4)
    model = ChallengerModel(_station_features(stations), min_train_rows=200)

    result = walk_forward(
        pairs, model, start=run_dates[5], end=run_dates[-1], retrain="monthly"
    )

    assert result.height > 0
    assert result["source"].unique().to_list() == ["challenger"]
    trained = result.filter(pl.col("trained_through").is_not_null())
    assert trained.height > 0
    assert (trained["trained_through"] < trained["runtime_utc"]).all()
    assert result["forecast_f"].null_count() == 0
    # It really trained (not all fallback) once data accumulated.
    assert (~result["fallback"]).sum() > 0


def test_cheating_challenger_gains_nothing_but_leaky_cutoff_helps() -> None:
    run_dates = _consecutive_run_dates(date(2026, 1, 1), 60)
    stations = ["KAAA", "KBBB", "KCCC"]
    pairs = _make_pairs(run_dates, stations, seed=21)
    features = _station_features(stations)
    start, end = run_dates[25], run_dates[-1]

    honest = walk_forward(
        pairs, ChallengerModel(features, min_train_rows=150), start, end, "monthly"
    )
    cheat = walk_forward(
        pairs,
        _CheatingChallenger(features, min_train_rows=150),
        start,
        end,
        "monthly",
    )
    key = ["run_date", "station", "lead_day", "variable"]
    assert cheat.sort(key)["forecast_f"].equals(honest.sort(key)["forecast_f"])

    raw = walk_forward(pairs, RawNbmModel(), start, end, "monthly")
    leaky = _leaky_daily_walk_forward(
        pairs, _CheatingChallenger(features, min_train_rows=150), start, end
    )
    honest_mae = _mae(honest, pairs, start)
    leaky_mae = _mae(leaky, pairs, start)
    assert _mae(cheat, pairs, start) == pytest.approx(honest_mae, abs=1e-9)
    assert leaky_mae < honest_mae - 0.3, (
        f"a run_date cutoff must let the cheat win: leaky={leaky_mae:.3f} "
        f"honest={honest_mae:.3f} raw={_mae(raw, pairs, start):.3f}"
    )


# -- d. determinism ---------------------------------------------------------------


def _run_challenger(pairs: pl.DataFrame, n_stations: int) -> pl.DataFrame:
    stations = sorted(pairs["station"].unique().to_list())
    assert len(stations) == n_stations
    model = ChallengerModel(_station_features(stations), min_train_rows=800)
    run_dates = sorted(pairs["run_date"].unique().to_list())
    return walk_forward(
        pairs, model, start=run_dates[45], end=run_dates[75], retrain="monthly"
    )


def test_identical_data_gives_identical_predictions_and_data_changes_them() -> None:
    pairs = _planted_pairs(n_days=80, n_stations=8)

    first = _run_challenger(pairs, 8)
    second = _run_challenger(pairs, 8)
    assert first["forecast_f"].equals(second["forecast_f"])
    assert (~first["fallback"]).sum() > 0

    changed = pairs.with_columns(
        pl.when(pl.col("station") == "KAAA")
        .then(pl.col("observed_f") - 5.0)
        .otherwise(pl.col("observed_f"))
        .alias("observed_f")
    )
    third = _run_challenger(changed, 8)
    assert not first["forecast_f"].equals(third["forecast_f"])


# -- e. fallback -------------------------------------------------------------------


def test_below_min_train_rows_every_prediction_falls_back() -> None:
    pairs = _planted_pairs(n_days=40, n_stations=4)
    stations = sorted(pairs["station"].unique().to_list())
    ordered = _scorable_sorted(pairs)
    small = ordered.head(150)
    model = ChallengerModel(_station_features(stations), min_train_rows=500)
    model.fit(small)

    rows = _predict_rows(pairs, date(2026, 2, 5))
    detail = model.predict_detail(rows)

    assert detail["forecast_f"].to_list() == rows["forecast_f"].to_list()
    assert detail["fallback"].to_list() == [True] * rows.height
    assert detail["forecast_f"].null_count() == 0
    assert model.fit_seconds_log == []

    model.fit(ordered)
    refit = model.predict_detail(rows)
    assert refit["fallback"].to_list() == [False] * rows.height
    assert refit["forecast_f"].null_count() == 0
    assert len(model.fit_seconds_log) == 1
    assert model.fit_seconds_log[0] > 0.0


def test_empty_training_set_falls_back_and_update_before_fit_is_safe() -> None:
    pairs = _planted_pairs(n_days=10, n_stations=2)
    stations = sorted(pairs["station"].unique().to_list())
    model = ChallengerModel(_station_features(stations))
    model.fit(pairs.head(0))
    rows = _predict_rows(pairs, date(2026, 1, 5))

    detail = model.predict_detail(rows)
    assert detail["fallback"].all()
    assert model.predict(rows).to_list() == rows["forecast_f"].to_list()

    fresh = ChallengerModel(_station_features(stations))
    fresh.update(_scorable_sorted(pairs).head(20))
    assert fresh.predict_detail(rows)["fallback"].all()


# -- f. learnability --------------------------------------------------------------


def test_challenger_learns_a_planted_feature_dependent_error() -> None:
    pairs = _planted_pairs(n_days=130, n_stations=8)
    stations = sorted(pairs["station"].unique().to_list())
    run_dates = sorted(pairs["run_date"].unique().to_list())
    start, end = run_dates[20], run_dates[-1]
    evaluated_since = run_dates[70]  # later months only

    raw = walk_forward(pairs, RawNbmModel(), start, end, "monthly")
    baseline = walk_forward(pairs, BaselineModel(), start, end, "monthly")
    challenger_result = walk_forward(
        pairs,
        ChallengerModel(_station_features(stations), min_train_rows=1000),
        start,
        end,
        "monthly",
    )

    raw_mae = _mae(raw, pairs, evaluated_since)
    baseline_mae = _mae(baseline, pairs, evaluated_since)
    challenger_mae = _mae(challenger_result, pairs, evaluated_since)
    assert challenger_mae < 0.5 * raw_mae, (challenger_mae, raw_mae)
    assert challenger_mae < 0.6 * baseline_mae, (challenger_mae, baseline_mae)
    assert baseline_mae < raw_mae


# -- g. no nulls, ever --------------------------------------------------------------


def test_predict_detail_never_returns_nulls_or_nans() -> None:
    pairs = _planted_pairs(n_days=60, n_stations=4)
    stations = sorted(pairs["station"].unique().to_list())
    ordered = _scorable_sorted(pairs)
    model = ChallengerModel(_station_features(stations), min_train_rows=500)
    model.fit(ordered)
    rows = _predict_rows(pairs, date(2026, 2, 20))
    assert not model.predict_detail(rows)["fallback"].any()

    # tmp_f null everywhere, a station metadata has never heard of, and a
    # (station, lead, variable) group with no history -> NaN features.
    degraded = rows.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("tmp_f"),
        pl.when(pl.col("station") == stations[0])
        .then(pl.lit("KNEW"))
        .otherwise(pl.col("station"))
        .alias("station"),
    )
    detail = model.predict_detail(degraded)
    assert detail.height == degraded.height
    assert detail["forecast_f"].null_count() == 0
    assert detail["forecast_f"].is_nan().sum() == 0
    assert detail["forecast_f"].is_finite().all()
    assert not detail["fallback"].any()

    # Optional columns absent altogether, as when the frame never carried them.
    bare = degraded.drop(["tmp_f", "cycle_hour"])
    again = model.predict_detail(bare)
    assert again["forecast_f"].null_count() == 0
    assert again["forecast_f"].is_finite().all()


def test_predictions_align_with_unsorted_run_rows() -> None:
    pairs = _planted_pairs(n_days=60, n_stations=4)
    stations = sorted(pairs["station"].unique().to_list())
    model = ChallengerModel(_station_features(stations), min_train_rows=500)
    model.fit(_scorable_sorted(pairs))
    rows = _predict_rows(pairs, date(2026, 2, 20))
    shuffled = rows.sample(fraction=1.0, shuffle=True, seed=7)

    in_order = model.predict_detail(rows)["forecast_f"].to_list()
    permuted = model.predict_detail(shuffled)["forecast_f"].to_list()

    def by_key(frame: pl.DataFrame, values: list[float]) -> dict[tuple, float]:
        keys = zip(
            frame["station"], frame["lead_day"], frame["variable"], strict=True
        )
        return dict(zip(keys, values, strict=True))

    assert by_key(rows, in_order) == pytest.approx(by_key(shuffled, permuted))


def test_station_features_are_validated() -> None:
    good = _station_features(["KAAA", "KBBB"])
    with pytest.raises(ValueError, match="elevation_m"):
        ChallengerModel(good.drop("elevation_m"))
    with pytest.raises(ValueError, match="one row per station"):
        ChallengerModel(pl.concat([good, good]))


# -- i. constants -------------------------------------------------------------------


def test_constants_equal_the_pre_registered_values() -> None:
    assert challenger.N_ESTIMATORS == 200
    assert challenger.LEARNING_RATE == 0.05
    assert challenger.NUM_LEAVES == 31
    assert challenger.MIN_DATA_IN_LEAF == 500
    assert challenger.FEATURE_FRACTION == 0.8
    assert challenger.BAGGING_FRACTION == 0.8
    assert challenger.BAGGING_FREQ == 1
    assert challenger.LAMBDA_L2 == 10.0
    assert challenger.MAX_BIN == 63
    assert challenger.SEED == 42
    assert challenger.MIN_TRAIN_ROWS == 20_000
    assert challenger.LAG_WINDOW_DAYS == 30
    assert challenger.LAG_COUNTS == (3, 7)
    assert challenger.FEATURE_COLUMNS == (
        "forecast_f",
        "spread_f",
        "tmp_f",
        "cycle_hour",
        "lead_day",
        "is_max",
        "doy_sin",
        "doy_cos",
        "latitude",
        "longitude",
        "elevation_m",
        "coastal_flag",
        "climate_region_code",
        "lag_error",
        "lag_mean3",
        "lag_mean7",
        "lag_n30",
        "lag_age_days",
    )
    assert ChallengerModel.name == "challenger"


def test_booster_params_are_the_pre_registered_ones() -> None:
    params = challenger.booster_params()

    assert params == {
        "objective": "regression_l1",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 63,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": 42,
        "verbose": -1,
    }


def test_model_params_json_carries_every_constant_and_versions() -> None:
    model = ChallengerModel(_station_features(["KAAA"]), min_train_rows=123)

    described = model.describe_params()

    assert described["n_estimators"] == 200
    assert described["min_train_rows"] == 123
    assert described["lightgbm_version"] == lightgbm.__version__
    assert described["lag_window_days"] == 30
    assert described["lag_counts"] == [3, 7]
    assert described["feature_columns"] == list(challenger.FEATURE_COLUMNS)
    assert described["objective"] == "regression_l1"
    assert described["seed"] == 42
    assert described["num_threads"] == 1


# -- j. teeth: properties the first version of this file did not pin ------------
#
# Added by the independent teeth review (tests/unit/teeth/challenger_model.
# teeth.json). Each test below is red against a named mutant in that spec and
# green on the real code.


def _history_rows(pairs: pl.DataFrame) -> list[dict[str, object]]:
    return pairs.select(
        "station",
        "lead_day",
        "variable",
        "runtime_utc",
        "window_end_utc",
        "forecast_f",
        "observed_f",
    ).to_dicts()


def _oracle_matrix(
    rows: pl.DataFrame,
    history: list[dict[str, object]],
    station_features: pl.DataFrame,
) -> np.ndarray:
    """The documented feature set (docs/methodology.md), restated row by row.

    Independent of `challenger._feature_frame`: stdlib math and a loop, the
    registry vocabulary sorted by name, the as-of lags from the brute-force
    reference. Unknown station / null input -> NaN, never zero.
    """
    meta = {r["station"]: r for r in station_features.iter_rows(named=True)}
    vocabulary = {
        name: float(i)
        for i, name in enumerate(
            sorted({m["climate_region"] for m in meta.values() if m["climate_region"]})
        )
    }
    nan = math.nan

    def number(value: object) -> float:
        return nan if value is None else float(value)

    out = []
    for row in rows.iter_rows(named=True):
        station = meta.get(row["station"])
        lag = _reference_lags(
            history,
            row["station"],
            row["lead_day"],
            row["variable"],
            row["runtime_utc"],
        )
        angle = 2.0 * math.pi * row["target_date"].timetuple().tm_yday / 365.25
        out.append(
            [
                number(row["forecast_f"]),
                number(row.get("spread_f")),
                number(row.get("tmp_f")),
                number(row.get("cycle_hour")),
                number(row["lead_day"]),
                {"max": 1.0, "min": 0.0}.get(row["variable"], nan),
                math.sin(angle),
                math.cos(angle),
                number(station["latitude"]) if station else nan,
                number(station["longitude"]) if station else nan,
                number(station["elevation_m"]) if station else nan,
                number(station["coastal_flag"]) if station else nan,
                vocabulary.get(station["climate_region"], nan) if station else nan,
                *(lag[c] for c in _LAG_COLUMNS),
            ]
        )
    return np.array(out, dtype=np.float64)


def _assert_matrix_matches(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    for j, name in enumerate(challenger.FEATURE_COLUMNS):
        np.testing.assert_allclose(
            actual[:, j].astype(np.float64),
            expected[:, j],
            rtol=2e-6,
            atol=2e-6,
            equal_nan=True,
            err_msg=f"feature column {name!r}",
        )


class _Capture:
    """Records the matrices LightGBM is handed, without changing what it does."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.train_matrix: np.ndarray | None = None
        self.predict_matrices: list[np.ndarray] = []
        real_dataset = lightgbm.Dataset
        real_predict = lightgbm.Booster.predict
        capture = self

        def dataset(data: np.ndarray, *args: object, **kwargs: object) -> object:
            capture.train_matrix = np.array(data, copy=True)
            return real_dataset(data, *args, **kwargs)

        def predict(
            booster: object, data: np.ndarray, *args: object, **kw: object
        ) -> np.ndarray:
            capture.predict_matrices.append(np.array(data, copy=True))
            return real_predict(booster, data, *args, **kw)

        monkeypatch.setattr(lightgbm, "Dataset", dataset)
        monkeypatch.setattr(lightgbm.Booster, "predict", predict)


def _small_world(
    n_days: int = 40, n_stations: int = 4, seed: int = 31
) -> tuple[pl.DataFrame, pl.DataFrame, list[str], pl.DataFrame]:
    """(pairs, scorable-sorted pairs, stations, station features)."""
    stations = [f"K{chr(65 + i)}{chr(65 + i)}{chr(65 + i)}" for i in range(n_stations)]
    pairs = _make_pairs(
        _consecutive_run_dates(date(2026, 1, 1), n_days),
        stations,
        seed=seed,
        error_fn=_planted_error,
    )
    return pairs, _scorable_sorted(pairs), stations, _station_features(stations)


def test_asof_features_ignore_other_time_columns_of_the_query() -> None:
    history, queries = _builder_history_and_queries(seed=14)
    stats = pair_stats(pl.DataFrame(history))
    plain = pl.DataFrame(queries)
    # A prediction frame also carries `window_end_utc` (the verification
    # window's close, after issuance) and `target_date`; neither is an as-of
    # time. Push both far past every pair in the history.
    decorated = plain.with_columns(
        (pl.col("runtime_utc") + pl.duration(days=400)).alias("window_end_utc"),
        pl.lit(date(2030, 1, 1)).alias("target_date"),
    )

    assert_frame_equal(
        asof_lag_features(decorated, stats),
        asof_lag_features(plain, stats),
        check_exact=True,
    )


def test_asof_features_do_not_depend_on_stats_or_history_row_order() -> None:
    history, queries = _builder_history_and_queries(seed=15)
    history_frame = pl.DataFrame(history)
    query_frame = pl.DataFrame(queries)
    reference = asof_lag_features(query_frame, pair_stats(history_frame))

    for seed in (1, 2, 3):
        shuffled_history = history_frame.sample(fraction=1.0, shuffle=True, seed=seed)
        # Shuffle the stats rows by close time, keeping the relative order of
        # rows that share one (a stable sort on a hash of the close): the
        # builder's own output order is the tie order it is documented to use.
        shuffled_stats = (
            pair_stats(history_frame)
            .with_columns(pl.col("window_end_utc").hash(seed=seed).alias("_h"))
            .sort("_h", maintain_order=True)
            .drop("_h")
        )
        # Exact ties on the close time exist inside groups (see the builder):
        # their order is by runtime_utc, never by arrival order.
        assert_frame_equal(
            asof_lag_features(query_frame, pair_stats(shuffled_history)),
            reference,
            check_exact=True,
        )
        assert_frame_equal(
            asof_lag_features(query_frame, shuffled_stats),
            reference,
            check_exact=True,
        )


def test_training_matrix_is_the_documented_feature_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 60 days: longer than RETENTION_DAYS, so the maintained history is
    # trimmed at the end of `fit` and the training lags must still have been
    # read from the full training history, not the trimmed one.
    _, ordered, _, features = _small_world(n_days=60)
    capture = _Capture(monkeypatch)
    model = ChallengerModel(features, min_train_rows=100)

    model.fit(ordered)

    assert model.is_fitted
    assert capture.train_matrix is not None
    _assert_matrix_matches(
        capture.train_matrix,
        _oracle_matrix(ordered, _history_rows(ordered), features),
    )
    # Not vacuous: lag columns are populated for later rows, missing for the
    # first, and the training rows' lags differ from the final-state lags.
    lag_error = capture.train_matrix[:, challenger.FEATURE_COLUMNS.index("lag_error")]
    assert np.isnan(lag_error).any()
    assert np.isfinite(lag_error).sum() > 0.5 * lag_error.size


def test_predict_matrix_is_the_documented_feature_set_over_the_maintained_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, ordered, stations, features = _small_world()
    fit_t = _issuance(date(2026, 1, 21))
    predict_t = _issuance(date(2026, 1, 31))
    capture = _Capture(monkeypatch)
    model = ChallengerModel(features, min_train_rows=100)
    model.fit(ordered.filter(pl.col("window_end_utc") < fit_t))
    model.update(
        ordered.filter(
            (pl.col("window_end_utc") >= fit_t) & (pl.col("window_end_utc") < predict_t)
        )
    )
    assert model.is_fitted

    rows = _predict_rows(pairs, date(2026, 1, 31)).with_columns(
        # one null tmp_f, one station the registry has never heard of
        pl.when(pl.int_range(pl.len()) == 0)
        .then(None)
        .otherwise(pl.col("tmp_f"))
        .alias("tmp_f"),
        pl.when(pl.col("station") == stations[1])
        .then(pl.lit("KNEW"))
        .otherwise(pl.col("station"))
        .alias("station"),
    )
    capture.predict_matrices.clear()
    model.predict_detail(rows)

    assert len(capture.predict_matrices) == 1
    expected = _oracle_matrix(rows, _history_rows(ordered), features)
    _assert_matrix_matches(capture.predict_matrices[0], expected)
    # Sanity of the fixture: the NaN rows really are in the oracle.
    j_tmp = challenger.FEATURE_COLUMNS.index("tmp_f")
    j_lat = challenger.FEATURE_COLUMNS.index("latitude")
    assert math.isnan(expected[0, j_tmp])
    assert np.isnan(expected[:, j_lat]).any()

    # A frame without optional columns: missing, never zero.
    bare = rows.drop(["tmp_f", "cycle_hour", "spread_f"])
    capture.predict_matrices.clear()
    model.predict_detail(bare)
    matrix = capture.predict_matrices[0]
    for name in ("tmp_f", "cycle_hour", "spread_f"):
        assert np.isnan(matrix[:, challenger.FEATURE_COLUMNS.index(name)]).all()


def test_trained_booster_carries_the_pre_registered_parameters() -> None:
    pairs = _planted_pairs(n_days=60, n_stations=4)
    stations = sorted(pairs["station"].unique().to_list())
    model = ChallengerModel(_station_features(stations), min_train_rows=500)
    model.fit(_scorable_sorted(pairs))
    assert model.is_fitted

    text = model._booster.model_to_string()  # noqa: SLF001
    header, _, tail = text.partition("parameters:")
    recorded = dict(
        line[1:-1].split(": ", 1)
        for line in tail.splitlines()
        if line.startswith("[") and ": " in line
    )

    assert "objective=regression_l1" in header
    assert f"feature_names={' '.join(challenger.FEATURE_COLUMNS)}" in header
    # The trained model's own record of what it was given: the constants
    # reached LightGBM, not merely the module that holds them.
    assert recorded["num_iterations"] == "200"
    assert recorded["learning_rate"] == "0.05"
    assert recorded["num_leaves"] == "31"
    assert recorded["min_data_in_leaf"] == "500"
    assert recorded["feature_fraction"] == "0.8"
    assert recorded["bagging_fraction"] == "0.8"
    assert recorded["bagging_freq"] == "1"
    assert recorded["lambda_l2"] == "10"
    assert recorded["max_bin"] == "63"
    assert recorded["seed"] == "42"
    assert recorded["deterministic"] == "1"
    assert recorded["force_row_wise"] == "1"
    assert recorded["num_threads"] == "1"
    # climate_region_code, and only it, is categorical.
    assert recorded["categorical_feature"] == str(
        challenger.FEATURE_COLUMNS.index("climate_region_code")
    )


def test_refit_on_too_few_rows_drops_the_previous_booster() -> None:
    pairs, ordered, _, features = _small_world(n_days=30)
    model = ChallengerModel(features, min_train_rows=300)
    rows = _predict_rows(pairs, date(2026, 1, 25))

    model.fit(ordered)
    assert model.is_fitted
    assert not model.predict_detail(rows)["fallback"].any()

    model.fit(ordered.head(50))

    assert not model.is_fitted
    detail = model.predict_detail(rows)
    assert detail["fallback"].to_list() == [True] * rows.height
    assert detail["forecast_f"].to_list() == rows["forecast_f"].to_list()


def test_min_train_rows_is_inclusive_and_counts_only_observed_pairs() -> None:
    _, ordered, _, features = _small_world(n_days=30)
    threshold = 300
    exact = ordered.head(threshold)

    at = ChallengerModel(features, min_train_rows=threshold)
    at.fit(exact)
    below = ChallengerModel(features, min_train_rows=threshold)
    below.fit(ordered.head(threshold - 1))
    # `threshold` rows of which one has no observation: 299 scorable pairs.
    holed = ChallengerModel(features, min_train_rows=threshold)
    holed.fit(
        exact.with_columns(
            pl.when(pl.int_range(pl.len()) == 7)
            .then(None)
            .otherwise(pl.col("observed_f"))
            .alias("observed_f")
        )
    )

    assert at.is_fitted
    assert not below.is_fitted
    assert not holed.is_fitted


def test_fit_count_counts_every_fit_fitted_or_not() -> None:
    _, ordered, _, features = _small_world(n_days=30)
    model = ChallengerModel(features, min_train_rows=300)

    model.fit(ordered.head(50))
    assert (model.fit_count, len(model.fit_seconds_log)) == (1, 0)
    model.fit(ordered)
    assert (model.fit_count, len(model.fit_seconds_log)) == (2, 1)


def test_non_finite_booster_output_keeps_the_raw_forecast_and_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs, ordered, _, features = _small_world(n_days=30)
    model = ChallengerModel(features, min_train_rows=300)
    model.fit(ordered)
    rows = _predict_rows(pairs, date(2026, 1, 25))
    assert rows.height >= 6
    returned = np.full(rows.height, 1.5)
    returned[1] = math.nan
    returned[2] = math.inf
    returned[4] = -math.inf
    monkeypatch.setattr(
        lightgbm.Booster, "predict", lambda self, data, *a, **kw: returned.copy()
    )

    detail = model.predict_detail(rows)

    raw = rows["forecast_f"].to_list()
    got = detail["forecast_f"].to_list()
    flags = detail["fallback"].to_list()
    for i, value in enumerate(raw):
        if i in (1, 2, 4):
            assert got[i] == value
            assert flags[i] is True
        else:
            assert got[i] == pytest.approx(value - 1.5)
            assert flags[i] is False
    assert detail["forecast_f"].null_count() == 0


def test_describe_params_records_retrain_cadence_retention_and_polars() -> None:
    import importlib.metadata

    described = ChallengerModel(_station_features(["KAAA"])).describe_params()

    assert described["retrain"] == "monthly"
    assert described["retention_days"] == challenger.RETENTION_DAYS
    assert described["polars_version"] == importlib.metadata.version("polars")
    assert described["min_train_rows"] == 20_000
    for key, value in challenger.booster_params().items():
        assert described[key] == value


def test_fit_bounds_the_maintained_history_to_what_the_features_need() -> None:
    _, ordered, _, features = _small_world(n_days=120, n_stations=4)
    model = ChallengerModel(features, min_train_rows=10**9)

    model.fit(ordered)

    # Independent restatement of the retention rule: per group, the pairs
    # within RETENTION_DAYS of the group's own latest close, plus its last
    # max(LAG_COUNTS) pairs.
    expected = 0
    for _, group in ordered.group_by(
        ["station", "lead_day", "variable"], maintain_order=True
    ):
        ends = sorted(group["window_end_utc"].to_list())
        horizon = ends[-1] - timedelta(days=challenger.RETENTION_DAYS)
        keep = {i for i, end in enumerate(ends) if end >= horizon}
        keep |= set(range(max(0, len(ends) - max(challenger.LAG_COUNTS)), len(ends)))
        expected += len(keep)
    assert expected < ordered.height / 2
    assert model.history.height == expected


def test_pairs_without_an_observation_are_ignored_by_the_builder() -> None:
    history, queries = _builder_history_and_queries(seed=16)
    frame = pl.DataFrame(history)
    holed = frame.with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 2)
        .then(None)
        .otherwise(pl.col("observed_f"))
        .alias("observed_f")
    )
    observed_only = holed.filter(pl.col("observed_f").is_not_null())
    assert 0 < observed_only.height < holed.height
    query_frame = pl.DataFrame(queries)

    assert_frame_equal(
        asof_lag_features(query_frame, pair_stats(holed)),
        asof_lag_features(query_frame, pair_stats(observed_only)),
        check_exact=True,
    )


def _signed_error(
    rng: np.random.Generator, station_index: int, tmp: float, variable: str, lead: int
) -> float:
    """+3 F when the guidance temperature is warm, -3 F when cool: both signs."""
    return (3.0 if tmp > 60.0 else -3.0) + float(rng.normal(0.0, 0.5))


def test_challenger_learns_the_sign_of_the_error_not_only_its_size() -> None:
    stations = [f"K{chr(65 + i)}{chr(65 + i)}{chr(65 + i)}" for i in range(8)]
    pairs = _make_pairs(
        _consecutive_run_dates(date(2026, 1, 1), 130),
        stations,
        seed=17,
        error_fn=_signed_error,
    )
    run_dates = sorted(pairs["run_date"].unique().to_list())
    start, end, since = run_dates[20], run_dates[-1], run_dates[70]

    raw = walk_forward(pairs, RawNbmModel(), start, end, "monthly")
    fitted = walk_forward(
        pairs,
        ChallengerModel(_station_features(stations), min_train_rows=1000),
        start,
        end,
        "monthly",
    )

    assert _mae(fitted, pairs, since) < 0.4 * _mae(raw, pairs, since)


def test_lag_features_do_not_depend_on_the_timestamp_representation() -> None:
    history, queries = _builder_history_and_queries(seed=18)
    utc_history = pl.DataFrame(history)
    utc_queries = pl.DataFrame(queries)
    reference = asof_lag_features(utc_queries, pair_stats(utc_history))
    time_columns = ("runtime_utc", "window_end_utc")

    for zone in ("America/Phoenix", "Asia/Kolkata"):
        local_history = utc_history.with_columns(
            pl.col(c).dt.convert_time_zone(zone) for c in time_columns
        )
        local_queries = utc_queries.with_columns(
            pl.col("runtime_utc").dt.convert_time_zone(zone)
        )
        # DuckDB hands TIMESTAMPTZ back in the session zone: the same instants.
        assert_frame_equal(
            asof_lag_features(local_queries, pair_stats(local_history)),
            reference,
            check_exact=True,
        )
        assert_frame_equal(
            asof_lag_features(local_queries, pair_stats(utc_history)),
            reference,
            check_exact=True,
        )

    # A plain TIMESTAMP column is already UTC wall-clock time by convention.
    naive_history = utc_history.with_columns(
        pl.col(c).dt.replace_time_zone(None) for c in time_columns
    )
    naive_queries = utc_queries.with_columns(
        pl.col("runtime_utc").dt.replace_time_zone(None)
    )
    assert_frame_equal(
        asof_lag_features(naive_queries, pair_stats(naive_history)),
        reference,
        check_exact=True,
    )
