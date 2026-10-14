"""Issuing one night's predictions into the live log (issue #20, slice B).

The acceptance oracle is `walk_forward` itself: for any sequence of issued
run dates, the rows `issue_predictions` appends must be bit-identical to what
the evaluated walk-forward produces for those same run dates (baseline with a
daily refit, challenger with a monthly retrain) -- while the live path only
refits once a month and loads only bounded history. Every night in the
scenarios below runs with a freshly opened DuckDB connection and freshly built
models, so no in-process state can carry a result across nights.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import lightgbm
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from support.synthetic_pairs import (
    make_pairs,
    runtime_of,
    station_features,
)

from weather_forecast_audit import challenger, cli, issuance, warehouse
from weather_forecast_audit.baseline import BaselineModel
from weather_forecast_audit.challenger import PAIR_COLUMNS, ChallengerModel
from weather_forecast_audit.issuance import (
    IssueResult,
    ModelArtifactError,
    NoCanonicalRunError,
    issue_predictions,
)
from weather_forecast_audit.prediction_log import (
    PartitionSealedError,
    verify_immutability,
)
from weather_forecast_audit.registry import load_registry
from weather_forecast_audit.walkforward import walk_forward

pytestmark = pytest.mark.unit

MIN_TRAIN_ROWS = 300
STATIONS = ["KAAA", "KBBB", "KCCC", "KDDD", "KEEE", "KSIL", "KSPR"]
FIRST_RUN = date(2025, 12, 15)
LAST_RUN = date(2026, 4, 8)
NULL_DATES = {date(2026, 2, 24), date(2026, 2, 25), date(2026, 2, 26)}
SKIPPED = NULL_DATES | {date(2026, 3, 1), date(2026, 3, 2)}
SENTINEL_RUN_DATE = date(1900, 1, 1)

_LOG_COLUMNS = [
    "run_date",
    "station",
    "lead_day",
    "variable",
    "forecast_f",
    "raw_forecast_f",
    "fallback",
    "retrained_on",
    "trained_through",
    "runtime_utc",
    "target_date",
]


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _active(station: str, run_date: date) -> bool:
    # KSPR reports weekly: a group with fewer than 7 pairs inside its own
    # 40-day horizon, so the "last 7 pairs" clause of the trim is binding.
    if station == "KSPR":
        return run_date.day % 7 == 0
    # KSIL reports for 48 days, goes silent for seven weeks, then returns:
    # its history is a long-lived group whose own latest close is old.
    if station == "KSIL":
        return run_date <= date(2026, 1, 31) or run_date >= date(2026, 3, 21)
    return True


def _null_forecast(station: str, run_date: date, variable: str, lead: int) -> bool:
    # Scorable pairs with no forecast, closing after the February retrain and
    # (lead 3 of 02-26 closing 03-01) before the March one.
    return station == "KAAA" and run_date in NULL_DATES


def _unscorable(station: str, run_date: date, variable: str, lead: int) -> bool:
    return (
        station == "KBBB"
        and variable == "min"
        and lead == 3
        and run_date.day % 5 == 0
    )


EDGE_RUN_DATES = {
    date(2026, 2, 9),  # closes at the February retrain's own issuance
    date(2026, 2, 14),
    date(2026, 3, 2),  # closes at the March retrain's own issuance
    date(2026, 3, 20),
    date(2026, 3, 31),  # closes at the April retrain's own issuance
}


def _closes_at_next_issue(
    station: str, run_date: date, variable: str, lead: int
) -> bool:
    # A pair closing exactly at an issuance time T is not in a fit at T
    # (strict `<`) but is in the delta that follows.
    return (
        station == "KEEE"
        and variable == "min"
        and lead == 1
        and run_date in EDGE_RUN_DATES
    )


def _pairs(
    first: date = FIRST_RUN,
    last: date = LAST_RUN,
    stations: list[str] | None = None,
) -> pl.DataFrame:
    return make_pairs(
        _days(first, last),
        stations or STATIONS,
        active=_active,
        null_forecast=_null_forecast,
        unscorable=_unscorable,
        closes_at_next_issue=_closes_at_next_issue,
    )


def _issue_dates() -> list[date]:
    """Consecutive nights, then two jumps: past March's silent-station return
    and into April (both months have to be retrained mid-scenario).
    """
    stretches = [
        (date(2026, 2, 10), date(2026, 3, 6)),
        (date(2026, 3, 21), date(2026, 3, 24)),
        (date(2026, 4, 1), date(2026, 4, 3)),
    ]
    return [
        d for first, last in stretches for d in _days(first, last) if d not in SKIPPED
    ]


def _now_for(run_date: date) -> datetime:
    return runtime_of(run_date).replace(tzinfo=UTC) + timedelta(hours=1)


def _create_db(path: Path, pairs: pl.DataFrame) -> None:
    with duckdb.connect(str(path)) as conn:
        warehouse.init_db(conn)
        conn.execute("create table int_raw_verification_pairs as select * from pairs")


def _issue(
    path: Path,
    run_date: date,
    *,
    now: datetime | None = None,
    features: pl.DataFrame | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> IssueResult:
    """One night, in a freshly opened connection with freshly built models."""
    with duckdb.connect(str(path)) as conn:
        warehouse.init_db(conn)
        return issue_predictions(
            conn,
            run_date,
            now=now or _now_for(run_date),
            station_features=(
                station_features(STATIONS) if features is None else features
            ),
            min_train_rows=min_train_rows,
        )


def _issue_all(path: Path, dates: list[date], **kwargs: Any) -> list[IssueResult]:  # noqa: ANN401
    return [_issue(path, d, **kwargs) for d in dates]


def _live(path: Path, source: str) -> pl.DataFrame:
    with duckdb.connect(str(path)) as conn:
        columns = ", ".join(_LOG_COLUMNS)
        return conn.execute(
            f"select {columns} from live.prediction_log where source = ?",  # noqa: S608
            [source],
        ).pl()


def _canon(frame: pl.DataFrame) -> pl.DataFrame:
    """The compared columns in one dtype layout and one order."""
    out = frame.select(_LOG_COLUMNS)
    for name in ("runtime_utc", "trained_through"):
        if getattr(out.schema[name], "time_zone", None) is not None:
            out = out.with_columns(
                pl.col(name).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            )
    return out.with_columns(
        pl.col("lead_day").cast(pl.Int64),
        pl.col("runtime_utc").cast(pl.Datetime("us")),
        pl.col("trained_through").cast(pl.Datetime("us")),
    ).sort("run_date", "station", "lead_day", "variable")


def _restrict(pairs: pl.DataFrame, issued: list[date]) -> pl.DataFrame:
    """The frame in which only `issued` are run dates; the rest still train.

    Re-tagging a non-issued run date's `run_date` to a sentinel leaves every
    pair (scorable, closed, observed) exactly as it was, so it keeps feeding
    training sets, but `walk_forward` no longer treats its rows as runs.
    """
    return pairs.with_columns(
        pl.when(pl.col("run_date").is_in(issued))
        .then(pl.col("run_date"))
        .otherwise(pl.lit(SENTINEL_RUN_DATE))
        .alias("run_date")
    )


def _oracle_baseline(pairs: pl.DataFrame, issued: list[date]) -> pl.DataFrame:
    result = walk_forward(
        _restrict(pairs, issued),
        BaselineModel(),
        min(issued),
        max(issued),
        retrain="daily",
    )
    return _canon(result)


def _oracle_challenger(pairs: pl.DataFrame, issued: list[date]) -> pl.DataFrame:
    model = ChallengerModel(
        station_features(STATIONS), min_train_rows=MIN_TRAIN_ROWS
    )
    result = walk_forward(
        _restrict(pairs, issued),
        model,
        min(issued),
        max(issued),
        retrain="monthly",
    )
    return _canon(result)


def _db_pairs(path: Path) -> pl.DataFrame:
    with duckdb.connect(str(path)) as conn:
        return conn.execute(
            f"select {', '.join(PAIR_COLUMNS)} from int_raw_verification_pairs"  # noqa: S608
        ).pl()


@pytest.fixture(autouse=True)
def _splittable_trees(monkeypatch: pytest.MonkeyPatch) -> None:
    # A booster with real splits on a few thousand synthetic rows; the
    # production 500-row leaves would make every tree a single leaf and the
    # lag features irrelevant, which would make the equivalence vacuous.
    monkeypatch.setattr(challenger, "MIN_DATA_IN_LEAF", 20)


class _CountTrain:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = lightgbm.train

        def counting(*args: Any, **kwargs: Any) -> lightgbm.Booster:  # noqa: ANN401
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(lightgbm, "train", counting)


@pytest.fixture
def count_train(monkeypatch: pytest.MonkeyPatch) -> _CountTrain:
    return _CountTrain(monkeypatch)


# -- the equivalence scenarios ----------------------------------------------------


@dataclass
class Scenario:
    path: Path
    issued: list[date]
    results: list[IssueResult]
    train_calls: int
    histories: list[pl.DataFrame]
    pairs: pl.DataFrame
    oracle_baseline: pl.DataFrame
    oracle_challenger: pl.DataFrame


def _run_scenario(path: Path, issued: list[date]) -> Scenario:
    """Issue `issued` night by night, then compute the walk-forward oracle."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(challenger, "MIN_DATA_IN_LEAF", 20)
        counter = _CountTrain(patch)
        histories: list[pl.DataFrame] = []
        real_restore = ChallengerModel.restore.__func__  # type: ignore[attr-defined]

        def spy(
            cls: type[ChallengerModel],
            features: pl.DataFrame,
            model_text: str | None,
            history_pairs: pl.DataFrame,
            *,
            min_train_rows: int = challenger.MIN_TRAIN_ROWS,
        ) -> ChallengerModel:
            histories.append(history_pairs)
            return real_restore(
                cls, features, model_text, history_pairs, min_train_rows=min_train_rows
            )

        patch.setattr(ChallengerModel, "restore", classmethod(spy))

        _create_db(path, _pairs())
        results = _issue_all(path, issued)
        live_train_calls = counter.calls
        pairs = _db_pairs(path)
        return Scenario(
            path=path,
            issued=issued,
            results=results,
            train_calls=live_train_calls,
            histories=histories,
            pairs=pairs,
            oracle_baseline=_oracle_baseline(pairs, issued),
            oracle_challenger=_oracle_challenger(pairs, issued),
        )


@pytest.fixture(scope="module")
def main_scenario(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    return _run_scenario(tmp_path_factory.mktemp("main") / "wfa.duckdb", _issue_dates())


@pytest.fixture(scope="module")
def gappy_scenario(tmp_path_factory: pytest.TempPathFactory) -> Scenario:
    issued = [
        date(2026, 2, 10),
        date(2026, 2, 12),
        date(2026, 2, 17),
        date(2026, 3, 5),
        date(2026, 3, 6),
        date(2026, 3, 20),
        date(2026, 4, 2),
        date(2026, 4, 7),
    ]
    return _run_scenario(tmp_path_factory.mktemp("gappy") / "wfa.duckdb", issued)


def test_challenger_rows_are_bit_identical_to_the_walk_forward(
    main_scenario: Scenario,
) -> None:
    live = _canon(_live(main_scenario.path, "challenger"))

    assert live.height == main_scenario.oracle_challenger.height > 0
    assert_frame_equal(live, main_scenario.oracle_challenger, check_exact=True)
    # Not vacuous: the booster really corrected, and fell back nowhere.
    assert (live["forecast_f"] != live["raw_forecast_f"]).mean() > 0.9
    assert not live["fallback"].any()


def test_baseline_rows_are_bit_identical_to_the_walk_forward(
    main_scenario: Scenario,
) -> None:
    live = _canon(_live(main_scenario.path, "baseline"))

    assert live.height == main_scenario.oracle_baseline.height > 0
    assert_frame_equal(live, main_scenario.oracle_baseline, check_exact=True)
    assert (live["forecast_f"] != live["raw_forecast_f"]).mean() > 0.5


def test_one_retrain_per_calendar_month_on_the_first_issued_date(
    main_scenario: Scenario,
) -> None:
    with duckdb.connect(str(main_scenario.path)) as conn:
        stored = conn.execute(
            "select retrained_on, cutoff_utc, n_train_rows, fit_seconds "
            "from live.challenger_models order by retrained_on"
        ).fetchall()

    # 03-01 and 03-02 were not issued: March's retrain is its first ISSUED date.
    assert [row[0] for row in stored] == [
        date(2026, 2, 10),
        date(2026, 3, 3),
        date(2026, 4, 1),
    ]
    assert [row[1] for row in stored] == [
        runtime_of(date(2026, 2, 10)),
        runtime_of(date(2026, 3, 3)),
        runtime_of(date(2026, 4, 1)),
    ]
    assert main_scenario.train_calls == 3
    retrained = [r.run_date for r in main_scenario.results if r.retrained]
    assert retrained == [date(2026, 2, 10), date(2026, 3, 3), date(2026, 4, 1)]
    for result in main_scenario.results:
        assert result.append.sealed
        assert result.append.n_late_rejected == 0
        assert result.fit_seconds >= 0.0
        assert result.wall_seconds > 0.0
        assert (result.challenger_retrained_on is not None) and (
            result.challenger_retrained_on.month == result.run_date.month
        )


def test_the_log_verifies_after_a_multi_night_run(main_scenario: Scenario) -> None:
    with duckdb.connect(str(main_scenario.path)) as conn:
        report = verify_immutability(conn)

    assert report.ok, report.problems
    assert report.partitions == len(main_scenario.issued)
    # Two sources per run row, nothing late, nothing dropped.
    n_run_rows = main_scenario.pairs.filter(
        pl.col("run_date").is_in(main_scenario.issued)
    ).height
    assert report.rows == 2 * n_run_rows


def test_skipped_days_equal_the_walk_forward_restricted_to_issued_dates(
    gappy_scenario: Scenario,
) -> None:
    for source, oracle in (
        ("baseline", gappy_scenario.oracle_baseline),
        ("challenger", gappy_scenario.oracle_challenger),
    ):
        live = _canon(_live(gappy_scenario.path, source))
        assert set(live["run_date"].to_list()) == set(gappy_scenario.issued)
        assert_frame_equal(live, oracle, check_exact=True)


def _expected_history_keys(
    pairs: pl.DataFrame, cutoff: datetime
) -> tuple[set[tuple[str, int, str, datetime]], int]:
    """An independent reading of fit-then-trim, group by group in Python."""
    closed = pairs.filter(
        pl.col("scorable")
        & (pl.col("window_end_utc") < cutoff)
        & pl.col("forecast_f").is_not_null()
        & pl.col("observed_f").is_not_null()
    )
    kept: set[tuple[str, int, str, datetime]] = set()
    for _, group in closed.group_by(["station", "lead_day", "variable"]):
        ordered = group.sort("window_end_utc", "runtime_utc")
        latest = ordered["window_end_utc"].max()
        horizon = latest - timedelta(days=challenger.RETENTION_DAYS)
        n = ordered.height
        for i, row in enumerate(ordered.iter_rows(named=True)):
            if row["window_end_utc"] >= horizon or i >= n - max(challenger.LAG_COUNTS):
                kept.add(
                    (
                        row["station"],
                        row["lead_day"],
                        row["variable"],
                        row["runtime_utc"],
                    )
                )
    return kept, closed.height


def test_restore_nights_load_exactly_the_trimmed_history(
    main_scenario: Scenario,
) -> None:
    restore_nights = [r for r in main_scenario.results if not r.retrained]
    assert len(main_scenario.histories) == len(restore_nights) > 20
    with duckdb.connect(str(main_scenario.path)) as conn:
        cutoffs = dict(
            conn.execute(
                "select retrained_on, cutoff_utc from live.challenger_models"
            ).fetchall()
        )

    saw_trimming = False
    reference: dict[datetime, tuple[set[tuple[str, int, str, datetime]], int]] = {}
    for night, history in zip(restore_nights, main_scenario.histories, strict=True):
        assert night.challenger_retrained_on is not None
        cutoff = cutoffs[night.challenger_retrained_on]
        if cutoff not in reference:
            reference[cutoff] = _expected_history_keys(main_scenario.pairs, cutoff)
        expected, n_closed = reference[cutoff]
        got = {
            (r["station"], r["lead_day"], r["variable"], r["runtime_utc"])
            for r in history.iter_rows(named=True)
        }
        assert got == expected, night.run_date
        assert history.height == len(expected) == night.n_history_rows
        assert history["forecast_f"].null_count() == 0
        assert history["observed_f"].null_count() == 0
        assert history["window_end_utc"].max() < cutoff
        saw_trimming = saw_trimming or len(expected) < n_closed
    # The bound was binding: older closed pairs existed and were not read.
    assert saw_trimming


def test_walk_forward_is_handed_only_the_bounded_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _scratch_db(tmp_path)
    dates = [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)]
    seen: list[tuple[str, pl.DataFrame]] = []
    real = issuance.walk_forward

    def spy(pairs: pl.DataFrame, model: Any, *args: Any, **kwargs: Any) -> pl.DataFrame:  # noqa: ANN401
        seen.append((type(model).__name__, pairs))
        return real(pairs, model, *args, **kwargs)

    monkeypatch.setattr(issuance, "walk_forward", spy)
    _issue_all(path, dates)
    everything = _db_pairs(path)
    baseline_frames = [p for name, p in seen if name == "BaselineModel"]
    challenger_frames = [p for name, p in seen if name != "BaselineModel"]
    assert len(baseline_frames) == len(challenger_frames) == len(dates)

    cutoff = runtime_of(dates[0])
    for run_date, base, chall in zip(
        dates, baseline_frames, challenger_frames, strict=True
    ):
        runtime = runtime_of(run_date)
        closed_all = everything.filter(
            pl.col("scorable") & (pl.col("window_end_utc") < runtime)
        )
        anchor = closed_all["window_end_utc"].max()
        window_start = anchor - timedelta(days=30)

        # Baseline: exactly the pairs of its window (plus the run's own rows).
        in_base = base.filter(pl.col("scorable") & (pl.col("window_end_utc") < runtime))
        assert in_base["window_end_utc"].min() > window_start
        assert in_base.height == closed_all.filter(
            pl.col("window_end_utc") > window_start
        ).height
        assert in_base.height < closed_all.height

        # Challenger: the first night reads everything closed; later nights
        # only what closed since the artifact's cutoff.
        in_chall = chall.filter(
            pl.col("scorable") & (pl.col("window_end_utc") < runtime)
        )
        if run_date == dates[0]:
            assert in_chall.height == closed_all.height
        else:
            assert in_chall["window_end_utc"].min() >= cutoff
            assert in_chall.height == closed_all.filter(
                pl.col("window_end_utc") >= cutoff
            ).height
            assert in_chall.height < closed_all.height / 4


def test_unfitted_month_persists_a_fallback_artifact_and_matches_the_walk_forward(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = tmp_path / "wfa.duckdb"
    pairs = _pairs(date(2026, 1, 5), date(2026, 2, 20))
    _create_db(path, pairs)
    dates = [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)]

    results = _issue_all(path, dates, min_train_rows=10**9)

    assert count_train.calls == 0
    with duckdb.connect(str(path)) as conn:
        text, n_rows = conn.execute(
            "select model_text, n_train_rows from live.challenger_models"
        ).fetchone()
    assert text == ""
    assert n_rows > 0
    live = _canon(_live(path, "challenger"))
    assert live["fallback"].all()
    model = ChallengerModel(station_features(STATIONS), min_train_rows=10**9)
    oracle = _canon(
        walk_forward(_restrict(_db_pairs(path), dates), model, dates[0], dates[-1])
    )
    assert_frame_equal(live, oracle, check_exact=True)
    assert [r.retrained for r in results] == [True, False, False]


# -- the booster round trip -------------------------------------------------------


def test_restored_booster_predicts_bit_identically() -> None:
    pairs = _pairs(date(2026, 1, 5), date(2026, 2, 25))
    cutoff = runtime_of(date(2026, 2, 15))
    training = pairs.filter(
        pl.col("scorable") & (pl.col("window_end_utc") < cutoff)
    ).sort("window_end_utc", "station", "runtime_utc", "lead_day", "variable")
    features = station_features(STATIONS)
    model = ChallengerModel(features, min_train_rows=MIN_TRAIN_ROWS)
    model.fit(training)
    assert model.is_fitted

    restored = ChallengerModel.restore(
        features, model.dump_booster(), training, min_train_rows=MIN_TRAIN_ROWS
    )

    run_rows = pairs.filter(pl.col("run_date") == date(2026, 2, 15)).sort(
        "station", "lead_day", "variable"
    )
    original = model.predict_detail(run_rows)
    again = restored.predict_detail(run_rows)
    assert_frame_equal(original, again, check_exact=True)
    assert (original["forecast_f"] != run_rows["forecast_f"]).any()
    assert restored.region_codes == model.region_codes
    assert restored.history.height == model.history.height
    assert restored.fit_count == 0


def test_dump_booster_refuses_an_unfitted_model_and_restore_none_is_raw() -> None:
    features = station_features(STATIONS)
    model = ChallengerModel(features, min_train_rows=10**9)
    with pytest.raises(ValueError, match="unfitted"):
        model.dump_booster()

    pairs = _pairs(date(2026, 1, 5), date(2026, 1, 20))
    restored = ChallengerModel.restore(features, None, pairs, min_train_rows=10**9)
    run_rows = pairs.filter(pl.col("run_date") == date(2026, 1, 20))
    detail = restored.predict_detail(run_rows)
    assert detail["fallback"].all()
    assert detail["forecast_f"].to_list() == run_rows["forecast_f"].to_list()


# -- tamper -----------------------------------------------------------------------


def _scratch_db(tmp_path: Path) -> Path:
    path = tmp_path / "wfa.duckdb"
    _create_db(path, _pairs(date(2025, 12, 20), date(2026, 2, 20)))
    return path


def test_a_modified_model_text_fails_its_sha_and_is_never_used(
    tmp_path: Path,
) -> None:
    path = _scratch_db(tmp_path)
    _issue_all(path, [date(2026, 2, 10), date(2026, 2, 11)])
    with duckdb.connect(str(path)) as conn:
        conn.execute("update live.challenger_models set model_text = model_text || ' '")

    with pytest.raises(ModelArtifactError, match="sha256"):
        _issue(path, date(2026, 2, 12))

    with duckdb.connect(str(path)) as conn:
        assert conn.execute(
            "select count(*) from live.partition_seals where run_date = ?",
            [date(2026, 2, 12)],
        ).fetchone() == (0,)


def test_changed_region_codes_refuse_the_stored_model(tmp_path: Path) -> None:
    path = _scratch_db(tmp_path)
    _issue_all(path, [date(2026, 2, 10)])

    # A registry change: the sorted region list gains a member, so the
    # categorical codes shift under the persisted booster.
    features = station_features(STATIONS).with_columns(
        pl.when(pl.col("station") == "KAAA")
        .then(pl.lit("Alaska"))
        .otherwise(pl.col("climate_region"))
        .alias("climate_region")
    )
    with pytest.raises(ModelArtifactError, match="climate-region codes"):
        _issue(path, date(2026, 2, 11), features=features)

    # And a stored value that was edited rather than the registry.
    with duckdb.connect(str(path)) as conn:
        conn.execute("update live.challenger_models set region_codes = '{}'")
    with pytest.raises(ModelArtifactError, match="climate-region codes"):
        _issue(path, date(2026, 2, 11))


# -- late rows --------------------------------------------------------------------


def test_rows_whose_window_already_closed_are_rejected_and_reported(
    tmp_path: Path,
) -> None:
    path = _scratch_db(tmp_path)
    run_date = date(2026, 2, 10)
    # 19:00 on the next day: a lead-1 min window (18:00 that day) has closed.
    now = runtime_of(run_date).replace(tzinfo=UTC) + timedelta(hours=30)

    result = _issue(path, run_date, now=now)

    pairs = _db_pairs(path).filter(pl.col("run_date") == run_date)
    closed = pairs.filter(pl.col("window_end_utc") <= now.replace(tzinfo=None))
    assert closed.height > 0
    expected = sorted(
        (r["station"], r["lead_day"], r["variable"], source)
        for r in closed.iter_rows(named=True)
        for source in ("baseline", "challenger")
    )
    assert list(result.append.late_keys) == expected
    assert result.append.n_late_rejected == len(expected)
    assert result.append.sealed
    assert result.append.n_appended == 2 * pairs.height - len(expected)
    with duckdb.connect(str(path)) as conn:
        assert verify_immutability(conn).ok


# -- errors, all before any fit ---------------------------------------------------


def test_no_canonical_run_raises_a_value_error(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = _scratch_db(tmp_path)

    with pytest.raises(NoCanonicalRunError, match="2030-01-01"):
        _issue(path, date(2030, 1, 1))
    assert issubclass(NoCanonicalRunError, ValueError)
    assert count_train.calls == 0


def test_mixed_runtime_utc_raises_before_any_fit(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = _scratch_db(tmp_path)
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            "update int_raw_verification_pairs set runtime_utc = runtime_utc "
            "+ interval 1 hour where run_date = '2026-02-10' and station = 'KAAA'"
        )

    with pytest.raises(ValueError, match="mixed cycles"):
        _issue(path, date(2026, 2, 10))
    assert count_train.calls == 0


def test_a_sealed_partition_raises_before_any_model_is_fit(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = _scratch_db(tmp_path)
    _issue(path, date(2026, 2, 10))
    fits = count_train.calls
    assert fits == 1

    with pytest.raises(PartitionSealedError):
        _issue(path, date(2026, 2, 10))
    assert count_train.calls == fits


def test_an_out_of_order_date_raises_before_any_fit(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = _scratch_db(tmp_path)
    _issue(path, date(2026, 2, 12))
    fits = count_train.calls

    with pytest.raises(ValueError, match="append-only"):
        _issue(path, date(2026, 2, 11))
    assert count_train.calls == fits


def test_now_before_the_runs_own_issuance_raises_before_any_fit(
    tmp_path: Path, count_train: _CountTrain
) -> None:
    path = _scratch_db(tmp_path)
    early = runtime_of(date(2026, 2, 10)).replace(tzinfo=UTC) - timedelta(hours=1)

    with pytest.raises(ValueError, match="after now"):
        _issue(path, date(2026, 2, 10), now=early)
    assert count_train.calls == 0


def test_a_naive_now_is_refused(tmp_path: Path) -> None:
    path = _scratch_db(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        _issue(path, date(2026, 2, 10), now=datetime(2026, 2, 10, 14, 0))  # noqa: DTZ001


# -- crash recovery ----------------------------------------------------------------


def _crash(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    msg = "simulated crash between persisting the booster and appending"
    raise RuntimeError(msg)


def test_an_unissued_orphan_artifact_is_skipped_and_a_rerun_reuses_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count_train: _CountTrain,
) -> None:
    path = _scratch_db(tmp_path)
    real_append = issuance.append_partition
    monkeypatch.setattr(issuance, "append_partition", _crash)
    with pytest.raises(RuntimeError, match="crash"):
        _issue(path, date(2026, 2, 10))
    assert count_train.calls == 1

    # Same date again: the orphan artifact is this date's own, so no refit.
    monkeypatch.setattr(issuance, "append_partition", real_append)
    rerun = _issue(path, date(2026, 2, 10))
    assert count_train.calls == 1
    assert rerun.retrained
    live = _canon(_live(path, "challenger"))
    oracle = _oracle_challenger(_db_pairs(path), [date(2026, 2, 10)])
    assert_frame_equal(live, oracle, check_exact=True)


def test_an_orphan_from_a_never_issued_date_does_not_govern_a_later_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count_train: _CountTrain,
) -> None:
    path = _scratch_db(tmp_path)
    real_append = issuance.append_partition
    monkeypatch.setattr(issuance, "append_partition", _crash)
    with pytest.raises(RuntimeError, match="crash"):
        _issue(path, date(2026, 2, 10))
    monkeypatch.setattr(issuance, "append_partition", real_append)

    # 02-10 never issued: the walk-forward over the issued dates retrains on
    # 02-11, so the live path must too (and ignore the 02-10 artifact).
    issued = [date(2026, 2, 11), date(2026, 2, 12)]
    results = _issue_all(path, issued)

    assert [r.retrained for r in results] == [True, False]
    assert [r.challenger_retrained_on for r in results] == [issued[0], issued[0]]
    assert count_train.calls == 2
    live = _canon(_live(path, "challenger"))
    assert_frame_equal(
        live, _oracle_challenger(_db_pairs(path), issued), check_exact=True
    )


# -- schema and wiring ------------------------------------------------------------


def test_init_db_creates_the_challenger_models_table_idempotently(
    tmp_path: Path,
) -> None:
    with duckdb.connect(str(tmp_path / "wfa.duckdb")) as conn:
        warehouse.init_db(conn)
        warehouse.init_db(conn)
        columns = conn.execute(
            "select column_name from information_schema.columns "
            "where table_schema = 'live' and table_name = 'challenger_models' "
            "order by ordinal_position"
        ).fetchall()
    assert [c[0] for c in columns] == [
        "retrained_on",
        "cutoff_utc",
        "trained_through",
        "n_train_rows",
        "model_text",
        "model_sha256",
        "region_codes",
        "params",
        "fit_seconds",
        "fitted_at",
    ]


def test_row_params_record_the_artifact_and_the_baseline_constants(
    main_scenario: Scenario,
) -> None:
    with duckdb.connect(str(main_scenario.path)) as conn:
        challenger_params = conn.execute(
            "select distinct params from live.prediction_log "
            "where source = 'challenger' and run_date = '2026-03-04'"
        ).fetchall()
        baseline_params = conn.execute(
            "select distinct params from live.prediction_log where source = 'baseline'"
        ).fetchall()
        sha = conn.execute(
            "select model_sha256 from live.challenger_models "
            "where retrained_on = '2026-03-03'"
        ).fetchone()

    assert len(challenger_params) == 1
    recorded = json.loads(challenger_params[0][0])
    assert recorded["model_sha256"] == sha[0]
    assert recorded["retrained_on"] == "2026-03-03"
    assert recorded["retrain"] == "monthly"
    assert recorded["seed"] == 42
    assert [json.loads(p[0]) for p in baseline_params] == [
        {"model": "baseline", "window_days": 30, "min_pairs": 15}
    ]


# -- the CLI ----------------------------------------------------------------------


def _live_pairs(late: bool) -> tuple[pl.DataFrame, date]:
    """Registry-station pairs around the real clock, since the CLI's `now` is."""
    stations = sorted(load_registry())[:3]
    runtime = (datetime.now(UTC) - timedelta(hours=1)).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )
    run_date = runtime.date()
    rows = make_pairs(
        [run_date - timedelta(days=k) for k in range(12, -1, -1)], stations
    )
    shift = runtime - runtime_of(run_date)
    frame = rows.with_columns(
        pl.col("runtime_utc") + shift,
        pl.col("window_start_utc") + shift,
        pl.col("window_end_utc") + shift,
    )
    # Past run dates are shifted along with today's, so the window clock stays
    # consistent; today's lead-1 min windows close in the past when `late`.
    if late:
        frame = frame.with_columns(
            pl.when(
                (pl.col("run_date") == run_date)
                & (pl.col("variable") == "min")
                & (pl.col("lead_day") == 1)
            )
            .then(runtime - timedelta(hours=1))
            .otherwise(pl.col("window_end_utc"))
            .alias("window_end_utc")
        )
    return frame, run_date


def test_cli_log_issue_smoke_prints_a_summary_and_the_log_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pairs, run_date = _live_pairs(late=False)
    path = tmp_path / "wfa.duckdb"
    _create_db(path, pairs)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(path))
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 50)

    cli.main(["log", "issue", "--run-date", run_date.isoformat()])
    out = capsys.readouterr().out

    n_today = pairs.filter(pl.col("run_date") == run_date).height
    assert f"wfa log issue {run_date}: appended={2 * n_today}" in out
    assert "late_rejected=0" in out
    assert "retrained=yes" in out
    assert "fit_seconds=" in out
    assert "wall_seconds=" in out
    cli.main(["log", "verify"])
    assert "OK partitions=1" in capsys.readouterr().out


def test_cli_log_issue_exits_1_when_rows_were_rejected_as_late(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pairs, run_date = _live_pairs(late=True)
    path = tmp_path / "wfa.duckdb"
    _create_db(path, pairs)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(path))
    monkeypatch.setattr(challenger, "MIN_TRAIN_ROWS", 50)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["log", "issue", "--run-date", run_date.isoformat()])

    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert "late_rejected=6" in out  # 3 stations x 2 sources
    assert "late: " in out
    with duckdb.connect(str(path)) as conn:
        assert verify_immutability(conn).ok
        sealed = conn.execute(
            "select n_rows, n_late_rejected from live.partition_seals"
        ).fetchone()
    assert sealed is not None
    assert sealed[1] == 6


def test_cli_log_issue_requires_run_date() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["log", "issue"])
