"""The live scorecard: window, rows, scoring, integrity gate, export contract.

The log is always built through `prediction_log.append_partition` (never raw
inserts) over a tmp DuckDB holding a minimal `int_raw_verification_pairs`.
Expected numbers are recomputed in the test from plain Python records and
`scoring.score`, so the module's own query, window and dedup logic is what is
under test, not a copy of it.
"""

import copy
import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pytest
from jsonschema import Draft202012Validator

from weather_forecast_audit import export, scorecard, warehouse
from weather_forecast_audit.prediction_log import (
    LogIntegrityError,
    append_partition,
)
from weather_forecast_audit.registry import load_registry
from weather_forecast_audit.scoring import (
    BLOCK_DAYS,
    CI_LEVEL,
    MIN_SAMPLE_BLOCKS,
    N_BOOT,
    score,
)

pytestmark = pytest.mark.unit

RUN_START = date(2026, 6, 1)
N_RUNS = 45
STATIONS = ("KPHX", "KDEN")
VARIABLES = ("max", "min")
LEADS = (1, 2)

_LOG_SCHEMA: dict[str, Any] = {
    "station": pl.String,
    "run_date": pl.Date,
    "runtime_utc": pl.Datetime("us"),
    "lead_day": pl.Int32,
    "variable": pl.String,
    "target_date": pl.Date,
    "source": pl.String,
    "forecast_f": pl.Float64,
    "raw_forecast_f": pl.Float64,
    "window_end_utc": pl.Datetime("us"),
    "retrained_on": pl.Date,
    "trained_through": pl.Datetime("us"),
    "fallback": pl.Boolean,
    "params": pl.String,
}


@dataclass(frozen=True)
class Key:
    station: str
    run_date: date
    lead_day: int
    variable: str
    raw: float
    baseline: float
    challenger: float
    observed: float | None
    late: bool = False

    @property
    def target_date(self) -> date:
        return self.run_date + timedelta(days=self.lead_day)

    @property
    def scorable(self) -> bool:
        return self.observed is not None


def _make_keys() -> list[Key]:
    rng = np.random.default_rng(20261013)
    keys: list[Key] = []
    last_run = RUN_START + timedelta(days=N_RUNS - 1)
    for i in range(N_RUNS):
        run_date = RUN_START + timedelta(days=i)
        for station in STATIONS:
            for variable in VARIABLES:
                for lead in LEADS:
                    observed = 70 + 10 * np.sin(i / 6) + rng.normal(0, 3)
                    raw = observed + 1.5 + rng.normal(0, 2.5)
                    baseline = raw - 1.2 + rng.normal(0, 0.4)
                    challenger = raw - 1.6 + rng.normal(0, 0.8)
                    unscorable = (i * 7 + lead + (variable == "min")) % 13 == 0
                    keys.append(
                        Key(
                            station=station,
                            run_date=run_date,
                            lead_day=lead,
                            variable=variable,
                            raw=round(float(raw), 1),
                            baseline=round(float(baseline), 1),
                            challenger=round(float(challenger), 1),
                            observed=None if unscorable else round(float(observed), 1),
                            # one key of the final partition is refused as late
                            late=(
                                run_date == last_run
                                and station == "KPHX"
                                and variable == "max"
                                and lead == 1
                            ),
                        )
                    )
    return keys


def _now_for(run_date: date) -> datetime:
    return datetime.combine(run_date, time(12, 0), tzinfo=UTC)


def _log_records(run_date: date, keys: list[Key]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in keys:
        end = datetime.combine(key.target_date + timedelta(days=1), time(6, 0))
        if key.late:
            end = datetime.combine(run_date, time(0, 0))  # closed before `now`
        for source, forecast in (
            ("baseline", key.baseline),
            ("challenger", key.challenger),
        ):
            records.append(
                {
                    "station": key.station,
                    "run_date": run_date,
                    "runtime_utc": datetime.combine(run_date, time(6, 0)),
                    "lead_day": key.lead_day,
                    "variable": key.variable,
                    "target_date": key.target_date,
                    "source": source,
                    "forecast_f": forecast,
                    "raw_forecast_f": key.raw,
                    "window_end_utc": end,
                    "retrained_on": run_date - timedelta(days=1),
                    "trained_through": datetime.combine(run_date, time(0, 0)),
                    "fallback": False,
                    "params": None,
                }
            )
    return records


def _create_warehouse_tables(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        create table int_raw_verification_pairs (
            station varchar, run_date date, lead_day integer, variable varchar,
            target_date date, forecast_f double, observed_f double,
            scorable boolean
        )
        """
    )
    conn.execute(
        """
        create table fct_forecast_verification (
            station varchar, run_date date, lead_day integer, variable varchar,
            source varchar, target_date date, scorable boolean, error_f double
        )
        """
    )


def _populate(
    conn: duckdb.DuckDBPyConnection, keys: list[Key], *, log: bool = True
) -> None:
    _create_warehouse_tables(conn)
    for key in keys:
        conn.execute(
            "insert into int_raw_verification_pairs values (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                key.station,
                key.run_date,
                key.lead_day,
                key.variable,
                key.target_date,
                key.raw,
                key.observed,
                key.scorable,
            ],
        )
    if not log:
        return
    for run_date in sorted({key.run_date for key in keys}):
        batch = [key for key in keys if key.run_date == run_date]
        frame = pl.DataFrame(_log_records(run_date, batch), schema=_LOG_SCHEMA)
        append_partition(conn, run_date, frame, now=_now_for(run_date))


@pytest.fixture
def keys() -> list[Key]:
    return _make_keys()


@pytest.fixture
def conn(tmp_path: Path, keys: list[Key]) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(tmp_path / "scorecard.duckdb"))
    warehouse.init_db(connection)
    _populate(connection, keys)
    return connection


def _logged(keys: list[Key]) -> list[Key]:
    return [key for key in keys if not key.late]


def _expected_window(keys: list[Key]) -> tuple[date, date]:
    end = max(key.target_date for key in _logged(keys) if key.scorable)
    return end - timedelta(days=29), end


def _expected_frame(keys: list[Key], start: date, end: date) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in _logged(keys):
        if not start <= key.target_date <= end:
            continue
        for source, forecast in (
            ("raw_nbm", key.raw),
            ("baseline", key.baseline),
            ("challenger", key.challenger),
        ):
            rows.append(
                {
                    "station": key.station,
                    "run_date": key.run_date,
                    "lead_day": key.lead_day,
                    "variable": key.variable,
                    "source": source,
                    "target_date": key.target_date,
                    "scorable": key.scorable,
                    "error_f": (
                        forecast - key.observed if key.observed is not None else None
                    ),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "station": pl.String,
            "run_date": pl.Date,
            "lead_day": pl.Int32,
            "variable": pl.String,
            "source": pl.String,
            "target_date": pl.Date,
            "scorable": pl.Boolean,
            "error_f": pl.Float64,
        },
    )


_RENAMES = {"mae": "mae_f", "mae_lo": "mae_lo_f", "mae_hi": "mae_hi_f"} | {
    "bias": "bias_f",
    "bias_lo": "bias_lo_f",
    "bias_hi": "bias_hi_f",
}


def _by_key(
    stats: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {(s["variable"], s["lead_day"], s["source"]): s for s in stats}


def _assert_stats_match(
    actual: list[dict[str, Any]], expected: pl.DataFrame
) -> None:
    got = _by_key(actual)
    assert len(got) == len(actual), "duplicate (variable, lead_day, source)"
    assert len(got) == expected.height
    for row in expected.iter_rows(named=True):
        stat = got[(row["variable"], row["lead_day"], row["source"])]
        for name in ("n", "n_dates", "n_blocks", "min_sample_flag"):
            assert stat[name] == row[name], (name, row)
        assert stat["no_detectable_bias"] == row["no_detectable_bias"]
        for source_name, target_name in [
            *_RENAMES.items(),
            ("skill", "skill"),
            ("skill_lo", "skill_lo"),
            ("skill_hi", "skill_hi"),
        ]:
            if row[source_name] is None:
                assert stat[target_name] is None, (source_name, row)
            else:
                assert stat[target_name] == pytest.approx(row[source_name], rel=1e-12)


# -- (a) the window ---------------------------------------------------------


def test_window_is_the_last_thirty_target_dates_inclusive(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    start, end = _expected_window(keys)
    card = scorecard.build_scorecard(conn)

    assert card.window == (start, end)
    assert (end - start).days == scorecard.WINDOW_DAYS - 1

    # Non-vacuous edges: scorable keys sit exactly on the first day and on the
    # day before it, and they land on opposite sides.
    assert any(k.target_date == start and k.scorable for k in _logged(keys))
    assert any(
        k.target_date == start - timedelta(days=1) and k.scorable
        for k in _logged(keys)
    )
    expected_raw_n = sum(
        1
        for k in _logged(keys)
        if k.scorable and start <= k.target_date <= end
    )
    raw_n = sum(s["n"] for s in card.national if s["source"] == "raw_nbm")
    assert raw_n == expected_raw_n
    assert expected_raw_n < sum(1 for k in _logged(keys) if k.scorable)


def test_window_end_is_the_latest_scorable_target_date_not_the_latest_logged(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    latest_logged = max(k.target_date for k in _logged(keys))
    _, end = _expected_window(keys)
    assert end <= latest_logged
    assert scorecard.build_scorecard(conn).window is not None
    assert scorecard.build_scorecard(conn).window[1] == end  # type: ignore[index]


_TAIL_CUTOFF = RUN_START + timedelta(days=38)  # last target_date still observed


@pytest.mark.parametrize("mode", ["unobserved", "unscorable", "unpaired"])
def test_window_ignores_a_tail_of_dates_that_cannot_be_graded_yet(
    tmp_path: Path, keys: list[Key], mode: str
) -> None:
    """Logged target_dates after the cutoff have no usable observation, in
    three different ways. None may anchor, shift or shorten the window, and
    none may be graded: the scorecard equals the one for keys that simply
    end at the cutoff."""
    graded = [
        replace(k, observed=None) if k.target_date > _TAIL_CUTOFF else k
        for k in keys
    ]
    latest_logged = max(k.target_date for k in _logged(keys))
    start, end = _expected_window(graded)
    assert end <= _TAIL_CUTOFF < latest_logged
    assert any(k.target_date > _TAIL_CUTOFF and k.observed is not None for k in keys)

    connection = duckdb.connect(str(tmp_path / f"{mode}.duckdb"))
    warehouse.init_db(connection)
    _populate(connection, graded if mode == "unobserved" else keys)
    if mode == "unscorable":
        # an observation is present but the pair is not scorable
        connection.execute(
            "update int_raw_verification_pairs set scorable = false "
            "where target_date > ?",
            [_TAIL_CUTOFF],
        )
    elif mode == "unpaired":
        connection.execute(
            "delete from int_raw_verification_pairs where target_date > ?",
            [_TAIL_CUTOFF],
        )

    card = scorecard.build_scorecard(connection)
    assert card.window == (start, end)
    expected = score(
        _expected_frame(graded, start, end),
        by=("variable", "lead_day"),
        reference="raw_nbm",
    )
    _assert_stats_match(card.national, expected)
    per_station = score(
        _expected_frame(graded, start, end),
        by=("station", "variable", "lead_day"),
        reference="raw_nbm",
    )
    for station in STATIONS:
        _assert_stats_match(
            card.stations[station], per_station.filter(pl.col("station") == station)
        )


# -- (b) equals score() computed independently ------------------------------


def test_national_stats_equal_score_computed_independently(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    start, end = _expected_window(keys)
    expected = score(
        _expected_frame(keys, start, end),
        by=("variable", "lead_day"),
        reference="raw_nbm",
    )
    card = scorecard.build_scorecard(conn)
    _assert_stats_match(card.national, expected)
    # The project's own floor speaks: 30 days is too few blocks for a call.
    assert card.national
    assert all(s["min_sample_flag"] for s in card.national)
    assert all(s["n_blocks"] < MIN_SAMPLE_BLOCKS for s in card.national)


def test_per_station_stats_equal_score_computed_independently(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    start, end = _expected_window(keys)
    expected = score(
        _expected_frame(keys, start, end),
        by=("station", "variable", "lead_day"),
        reference="raw_nbm",
    )
    card = scorecard.build_scorecard(conn)
    assert set(card.stations) == set(STATIONS)
    for station in STATIONS:
        _assert_stats_match(
            card.stations[station], expected.filter(pl.col("station") == station)
        )


def test_stats_are_ordered_deterministically(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    card = scorecard.build_scorecard(conn)
    order = {"raw_nbm": 0, "baseline": 1, "challenger": 2}
    variable_order = {"max": 0, "min": 1}
    for stats in (card.national, *card.stations.values()):
        keys_in_order = [
            (variable_order[s["variable"]], s["lead_day"], order[s["source"]])
            for s in stats
        ]
        assert keys_in_order == sorted(keys_in_order)


def test_skill_is_against_raw_and_raw_scores_exactly_zero(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    card = scorecard.build_scorecard(conn)
    for stat in card.national:
        if stat["source"] == "raw_nbm":
            assert stat["skill"] == 0.0
            assert stat["skill_lo"] == 0.0
            assert stat["skill_hi"] == 0.0
        else:
            assert stat["skill"] is not None


# -- (c) raw is what the log issued -----------------------------------------


def test_raw_nbm_comes_from_the_log_not_the_warehouse_forecast(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    before = scorecard.build_scorecard(conn)
    conn.execute("update int_raw_verification_pairs set forecast_f = forecast_f + 50")
    after = scorecard.build_scorecard(conn)
    assert after == before


def test_raw_nbm_is_one_row_per_logged_key_across_sources(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    card = scorecard.build_scorecard(conn)
    start, end = _expected_window(keys)
    by = _by_key(card.national)
    for variable in VARIABLES:
        for lead in LEADS:
            expected_n = sum(
                1
                for k in _logged(keys)
                if k.variable == variable
                and k.lead_day == lead
                and k.scorable
                and start <= k.target_date <= end
            )
            assert by[(variable, lead, "raw_nbm")]["n"] == expected_n
            assert by[(variable, lead, "baseline")]["n"] == expected_n
            assert by[(variable, lead, "challenger")]["n"] == expected_n


# -- (d) unscorable rows ----------------------------------------------------


def test_unscorable_rows_are_excluded_and_counts_match(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    start, end = _expected_window(keys)
    in_window = [k for k in _logged(keys) if start <= k.target_date <= end]
    assert any(not k.scorable for k in in_window), "fixture must include some"

    card = scorecard.build_scorecard(conn)
    total = sum(s["n"] for s in card.national if s["source"] == "baseline")
    assert total == sum(1 for k in in_window if k.scorable)
    assert total < len(in_window)


def test_a_log_with_no_graded_row_is_an_empty_scorecard(
    tmp_path: Path, keys: list[Key]
) -> None:
    ungraded = [
        Key(
            k.station, k.run_date, k.lead_day, k.variable, k.raw, k.baseline,
            k.challenger, None, k.late,
        )
        for k in keys
    ]
    connection = duckdb.connect(str(tmp_path / "ungraded.duckdb"))
    warehouse.init_db(connection)
    _populate(connection, ungraded)

    card = scorecard.build_scorecard(connection)
    assert card.window is None
    assert card.national == []
    assert card.stations == {}
    assert card.log["rows"] > 0
    assert card.log["verified"] is True


# -- the log block ----------------------------------------------------------


def test_log_block_reports_the_chain_and_the_late_refusals(
    conn: duckdb.DuckDBPyConnection, keys: list[Key]
) -> None:
    card = scorecard.build_scorecard(conn)
    seal = conn.execute(
        "select chain_sha256 from live.partition_seals order by run_date desc limit 1"
    ).fetchone()
    assert seal is not None
    last_run = RUN_START + timedelta(days=N_RUNS - 1)

    assert card.log == {
        "partitions": N_RUNS,
        "rows": 2 * len(_logged(keys)),
        "first_run_date": RUN_START.isoformat(),
        "last_run_date": last_run.isoformat(),
        "late_rejected_total": 2,
        "head_chain_sha256": seal[0],
        "verified": True,
    }


def test_an_observed_but_unscorable_pair_inside_the_window_is_not_graded(
    tmp_path: Path, keys: list[Key]
) -> None:
    """`scorable` is the warehouse's verdict, not "an observation exists": a
    pair that carries an observed_f but is flagged not scorable (e.g. an
    incomplete observation day) must not be graded, anywhere in the window."""
    start, end = _expected_window(keys)

    def _flagged(k: Key) -> bool:
        return (
            k.station == "KPHX"
            and k.variable == "max"
            and k.lead_day == 1
            and start <= k.target_date <= end
            and k.run_date.toordinal() % 3 == 0
            and k.observed is not None
        )

    flagged = [k for k in keys if _flagged(k)]
    assert len(flagged) >= 5
    graded = [replace(k, observed=None) if _flagged(k) else k for k in keys]

    connection = duckdb.connect(str(tmp_path / "flagged.duckdb"))
    warehouse.init_db(connection)
    _populate(connection, keys)  # observed_f is present for every flagged key
    for k in flagged:
        connection.execute(
            "update int_raw_verification_pairs set scorable = false "
            "where station = ? and run_date = ? and lead_day = ? and variable = ?",
            [k.station, k.run_date, k.lead_day, k.variable],
        )

    card = scorecard.build_scorecard(connection)
    assert card.window == _expected_window(graded)
    _assert_stats_match(
        card.national,
        score(
            _expected_frame(graded, start, end),
            by=("variable", "lead_day"),
            reference="raw_nbm",
        ),
    )
    expected_n = sum(
        1 for k in graded if k.scorable and start <= k.target_date <= end
        and (k.station, k.variable, k.lead_day) == ("KPHX", "max", 1)
    )
    by = _by_key(card.stations["KPHX"])
    for source in ("raw_nbm", "baseline", "challenger"):
        assert by[("max", 1, source)]["n"] == expected_n


def test_build_scorecard_called_directly_refuses_a_tampered_log(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """The verification lives in `build_scorecard` too, not only in `export`:
    a caller that passes no report must still never score a tampered log."""
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1.0 "
        "where run_date = ? and station = 'KPHX'",
        [RUN_START + timedelta(days=10)],
    )
    with pytest.raises(LogIntegrityError):
        scorecard.build_scorecard(conn)


def test_intervals_note_carries_the_window_the_blocks_and_the_floor() -> None:
    note = scorecard.intervals_note()
    assert f"{scorecard.WINDOW_DAYS}-day window" in note
    assert f"{BLOCK_DAYS}-day blocks" in note
    assert f"min_sample_blocks ({MIN_SAMPLE_BLOCKS})" in note
    assert "too short for a significance call" in note
    assert scorecard.intervals_block()["note"] == note


# -- (e) empty log ----------------------------------------------------------


def _registry() -> dict[str, Any]:
    full = load_registry()
    return {icao: full[icao] for icao in ("KPHX", "KDEN", "KSEA")}


def test_empty_log_exports_valid_empty_scorecard_files(tmp_path: Path) -> None:
    connection = duckdb.connect(str(tmp_path / "empty.duckdb"))
    warehouse.init_db(connection)
    _create_warehouse_tables(connection)

    out = tmp_path / "out"
    result = export.export(connection, out, registry=_registry())

    card = json.loads((out / "scorecard.json").read_text())
    assert card["window"] is None
    assert card["national"] == []
    assert card["log"]["partitions"] == 0
    assert card["log"]["rows"] == 0
    assert card["log"]["first_run_date"] is None
    assert card["log"]["verified"] is True
    assert "scorecard.json" in result.files
    schemas = export.schema_files()
    Draft202012Validator(
        json.loads(schemas["scorecard.schema.json"].read_text())
    ).validate(card)
    for icao in _registry():
        city = json.loads((out / "scorecard_cities" / f"{icao}.json").read_text())
        assert city["stats"] == []
        assert city["window"] is None
        Draft202012Validator(
            json.loads(schemas["scorecard_city.schema.json"].read_text())
        ).validate(city)


def test_database_without_the_live_schema_exports_an_empty_scorecard(
    tmp_path: Path,
) -> None:
    connection = duckdb.connect(str(tmp_path / "old.duckdb"))
    _create_warehouse_tables(connection)
    out = tmp_path / "out"
    export.export(connection, out, registry=_registry())
    card = json.loads((out / "scorecard.json").read_text())
    assert card["window"] is None
    assert card["log"]["rows"] == 0


# -- exported values: layout, rounding, invariants ---------------------------


def _export(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> tuple[Path, export.ExportResult]:
    out = tmp_path / "out"
    return out, export.export(conn, out, registry=_registry())


def test_export_writes_a_city_file_for_every_registry_station(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path, keys: list[Key]
) -> None:
    out, result = _export(conn, tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    start, end = _expected_window(keys)

    assert manifest["schema_version"] == export.SCHEMA_VERSION == "1.1.0"
    assert "scorecard.json" in manifest["files"]
    for icao in _registry():
        assert f"scorecard_cities/{icao}.json" in manifest["files"]
    assert manifest["files"] == result.files

    card = json.loads((out / "scorecard.json").read_text())
    assert card["schema_version"] == "1.1.0"
    assert card["window"] == {
        "days": 30,
        "basis": "target_date",
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    assert card["intervals"]["method"] == "date-block bootstrap"
    assert card["intervals"]["block_days"] == BLOCK_DAYS
    assert card["intervals"]["n_boot"] == N_BOOT
    assert card["intervals"]["level"] == CI_LEVEL
    assert card["intervals"]["min_sample_blocks"] == MIN_SAMPLE_BLOCKS
    assert card["intervals"]["note"]

    kphx = json.loads((out / "scorecard_cities" / "KPHX.json").read_text())
    assert kphx["icao"] == "KPHX"
    assert kphx["label"] == "Phoenix/Sky Harbor, AZ"
    assert kphx["window"] == card["window"]
    assert kphx["stats"]
    ksea = json.loads((out / "scorecard_cities" / "KSEA.json").read_text())
    assert ksea["stats"] == []  # a registry station with no logged rows


def _spans_zero(lo: float | None, hi: float | None) -> bool:
    assert lo is not None and hi is not None
    return lo <= 0 <= hi


def test_exported_bias_and_skill_intervals_agree_with_their_unrounded_flags(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    out, _ = _export(conn, tmp_path)
    card = scorecard.build_scorecard(conn)
    exported = json.loads((out / "scorecard.json").read_text())["national"]
    raw = _by_key(card.national)

    assert len(exported) == len(card.national)
    for stat in exported:
        unrounded = raw[(stat["variable"], stat["lead_day"], stat["source"])]
        if stat["bias_lo_f"] is not None:
            assert _spans_zero(stat["bias_lo_f"], stat["bias_hi_f"]) == (
                stat["no_detectable_bias"]
            ), stat
            assert stat["no_detectable_bias"] == unrounded["no_detectable_bias"]
            assert stat["bias_lo_f"] <= stat["bias_f"] <= stat["bias_hi_f"], stat
        if stat["skill_lo"] is not None:
            assert _spans_zero(stat["skill_lo"], stat["skill_hi"]) == _spans_zero(
                unrounded["skill_lo"], unrounded["skill_hi"]
            ), stat
            assert stat["skill_lo"] <= stat["skill"] <= stat["skill_hi"], stat
        assert stat["min_sample_flag"] is True


def test_exported_numbers_are_the_rounded_unrounded_numbers(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    out, _ = _export(conn, tmp_path)
    card = scorecard.build_scorecard(conn)
    exported = json.loads((out / "scorecard.json").read_text())["national"]
    raw = _by_key(card.national)
    for stat in exported:
        unrounded = raw[(stat["variable"], stat["lead_day"], stat["source"])]
        assert stat["mae_f"] == pytest.approx(unrounded["mae_f"], abs=0.0051)
        assert stat["bias_f"] == pytest.approx(unrounded["bias_f"], abs=0.0151)
        assert stat["n"] == unrounded["n"]
        assert set(stat) == set(unrounded)


# -- (g) a scorecard is never published over a log that fails its own check --


def test_tampered_log_raises_and_writes_nothing(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute(
        "update live.prediction_log set forecast_f = forecast_f + 1.0 "
        "where run_date = ? and station = 'KPHX'",
        [RUN_START + timedelta(days=10)],
    )
    out = tmp_path / "out"
    with pytest.raises(LogIntegrityError):
        export.export(conn, out, registry=_registry())
    assert not out.exists() or not any(out.rglob("*"))


def test_a_missing_seal_blocks_the_export_too(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    conn.execute(
        "delete from live.partition_seals where run_date = ?",
        [RUN_START + timedelta(days=3)],
    )
    out = tmp_path / "out"
    with pytest.raises(LogIntegrityError):
        export.export(conn, out, registry=_registry())
    assert not out.exists() or not any(out.rglob("*"))


# -- (h) contract -----------------------------------------------------------


def test_both_new_schemas_validate_real_output(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    out, _ = _export(conn, tmp_path)
    schemas = export.schema_files()
    national = Draft202012Validator(
        json.loads(schemas["scorecard.schema.json"].read_text())
    )
    city = Draft202012Validator(
        json.loads(schemas["scorecard_city.schema.json"].read_text())
    )
    manifest = Draft202012Validator(
        json.loads(schemas["manifest.schema.json"].read_text())
    )
    payload = json.loads((out / "scorecard.json").read_text())
    assert payload["national"], "needs real stats to be a meaningful validation"
    national.validate(payload)
    for icao in _registry():
        city.validate(
            json.loads((out / "scorecard_cities" / f"{icao}.json").read_text())
        )
    manifest.validate(json.loads((out / "manifest.json").read_text()))


def test_schemas_reject_unknown_and_missing_fields(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    out, _ = _export(conn, tmp_path)
    schemas = export.schema_files()
    validator = Draft202012Validator(
        json.loads(schemas["scorecard.schema.json"].read_text())
    )
    payload = json.loads((out / "scorecard.json").read_text())

    extra = json.loads(json.dumps(payload))
    extra["national"][0]["surprise"] = 1
    assert list(validator.iter_errors(extra))

    missing = json.loads(json.dumps(payload))
    del missing["national"][0]["skill_hi"]
    assert list(validator.iter_errors(missing))

    bad_date = json.loads(json.dumps(payload))
    bad_date["window"]["start"] = "June 1"
    assert list(validator.iter_errors(bad_date))


def _object_paths(
    node: object, path: tuple[object, ...] = ()
) -> list[tuple[object, ...]]:
    paths: list[tuple[object, ...]] = []
    if isinstance(node, dict):
        paths.append(path)
        for key, value in node.items():
            paths.extend(_object_paths(value, (*path, key)))
    elif isinstance(node, list) and node:
        paths.extend(_object_paths(node[0], (*path, 0)))
    return paths


@pytest.mark.parametrize(
    ("schema_name", "payload_path"),
    [
        ("scorecard.schema.json", "scorecard.json"),
        ("scorecard_city.schema.json", "scorecard_cities/KPHX.json"),
    ],
)
def test_schemas_reject_an_unknown_field_at_every_object_level(
    conn: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    schema_name: str,
    payload_path: str,
) -> None:
    out, _ = _export(conn, tmp_path)
    validator = Draft202012Validator(
        json.loads(export.schema_files()[schema_name].read_text())
    )
    payload = json.loads((out / payload_path).read_text())
    validator.validate(payload)
    paths = _object_paths(payload)
    assert len(paths) >= 3, paths  # top level, window and a stat at least
    for path in paths:
        mutated = copy.deepcopy(payload)
        target: Any = mutated
        for step in path:
            target = target[step]
        target["surprise"] = 1
        assert list(validator.iter_errors(mutated)), f"open at {path}"


def test_lock_entry_matches_the_canonical_hash_of_all_six_files() -> None:
    lock = export.load_lock()
    entry = lock["1.1.0"]
    assert set(entry) == set(export.schema_files())
    assert len(entry) == 6
    for name, path in export.schema_files().items():
        assert entry[name] == export.canonical_hash(json.loads(path.read_text()))


def test_every_manifest_file_exists(
    conn: duckdb.DuckDBPyConnection, tmp_path: Path
) -> None:
    out, _ = _export(conn, tmp_path)
    manifest = json.loads((out / "manifest.json").read_text())
    for relative in manifest["files"]:
        assert (out / relative).is_file(), relative


def test_sources_disagreeing_on_the_raw_forecast_are_refused_not_tie_broken(
    tmp_path: Path, keys: list[Key]
) -> None:
    connection = duckdb.connect(str(tmp_path / "disagree.duckdb"))
    warehouse.init_db(connection)
    _populate(connection, keys, log=False)
    run_dates = sorted({key.run_date for key in keys})
    for run_date in run_dates:
        batch = [key for key in keys if key.run_date == run_date and not key.late]
        records = _log_records(run_date, batch)
        if run_date == run_dates[-1]:
            # a stored baseline row (inside the window) now claims a different
            # raw value than its challenger twin
            records[0]["raw_forecast_f"] += 1.0
        frame = pl.DataFrame(records, schema=_LOG_SCHEMA)
        append_partition(connection, run_date, frame, now=_now_for(run_date))

    with pytest.raises(ValueError, match="raw_forecast_f"):
        scorecard.build_scorecard(connection)


def test_a_non_finite_observation_is_not_graded_and_never_reaches_the_json(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """A NaN observed_f flagged scorable passed `is_not_null` and poisoned
    every MAE of its group with NaN; strict JSON parsers (a browser's
    `JSON.parse`) then reject the whole file."""
    window = scorecard.build_scorecard(conn).window
    assert window is not None
    conn.execute(
        "update int_raw_verification_pairs set observed_f = 'NaN'::double, "
        "scorable = true where station = 'KPHX' and variable = 'max' "
        "and lead_day = 1 and target_date between ? and ?",
        [window[0], window[1]],
    )
    poisoned = conn.execute(
        "select count(*) from int_raw_verification_pairs "
        "where isnan(observed_f) and station = 'KPHX'"
    ).fetchone()
    assert poisoned is not None and poisoned[0] > 0  # the fixture is not vacuous

    card = scorecard.build_scorecard(conn)

    json.dumps(
        {"national": card.national, "stations": card.stations}, allow_nan=False
    )


def test_export_refuses_to_write_non_finite_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON"):
        export._write_json(tmp_path / "x.json", {"a": float("nan")})
