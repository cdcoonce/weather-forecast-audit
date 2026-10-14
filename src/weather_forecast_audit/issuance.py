"""Issue one night's predictions into the append-only live log (issue #20, B).

`issue_predictions` produces, for ONE run date, exactly the rows the evaluated
walk-forward would have produced for that date: baseline with a daily refit,
challenger with a monthly retrain. The difference is cost. The walk-forward
loads every pair and refits on every retrain date; the nightly job shares a
600 s run slot on a 5 GiB host. So:

* **baseline** reads only the 30-day window it can use (bounded load);
* **challenger** retrains once per calendar month, on the first issued run
  date of that month, persists the booster (`live.challenger_models`, written
  once, sha256-verified on load), and on every other night restores it and
  hands it the pairs closed since, through the same `update` path the
  walk-forward uses. The maintained history it needs is exactly what
  `ChallengerModel._trim` keeps, which SQL computes group by group, so only
  that is read.

The invariant (the acceptance oracle in `tests/unit/test_issuance.py`): for any
sequence of issued run dates, the appended `forecast_f`, `raw_forecast_f`,
`fallback`, `retrained_on`, `trained_through`, `runtime_utc` and `target_date`
are bit-identical to `walk_forward` over the full pair frame restricted to
those run dates. No row selection, ordering or null checking is
re-implemented: both sources go through `walk_forward` itself, on a frame
holding only what the night needs.

Information cutoff: training pairs are scorable pairs whose
`window_end_utc` is strictly before the run's `runtime_utc` (never before
`now`). `now` only stamps `issued_at` and drives the late rule.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import duckdb
import polars as pl

from weather_forecast_audit import challenger as challenger_module
from weather_forecast_audit.baseline import BaselineModel
from weather_forecast_audit.challenger import (
    LAG_COUNTS,
    PAIR_COLUMNS,
    RETENTION_DAYS,
    ChallengerModel,
)
from weather_forecast_audit.prediction_log import (
    AppendResult,
    PartitionSealedError,
    append_partition,
    init_log,
)
from weather_forecast_audit.walkforward import walk_forward

_PAIR_TABLE = "int_raw_verification_pairs"
_PAIR_SELECT = ", ".join(PAIR_COLUMNS)
_GROUP_SQL = "station, lead_day, variable"
# Sentinel a persisted `model_text` takes for a model that was unfitted (below
# `min_train_rows`) at its retrain: it falls back to raw all month, and the
# column is NOT NULL.
_UNFITTED_TEXT = ""


class NoCanonicalRunError(ValueError):
    """The pair table holds no rows for the requested run date."""


class ModelArtifactError(RuntimeError):
    """A persisted challenger booster cannot be trusted (hash or codes differ)."""


@dataclass(frozen=True)
class IssueResult:
    """What one `issue_predictions` call did."""

    run_date: date
    append: AppendResult
    retrained: bool
    challenger_retrained_on: date | None
    fit_seconds: float
    n_history_rows: int
    wall_seconds: float


@dataclass(frozen=True)
class _Artifact:
    retrained_on: date
    cutoff_utc: datetime
    trained_through: datetime | None
    n_train_rows: int
    model_text: str
    model_sha256: str
    region_codes: str
    params: str
    fit_seconds: float


def init_issuance(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the `live` tables (idempotent).

    The `live.challenger_models` DDL lives in `prediction_log` so that
    `warehouse.init_db` can create it without importing this module, which
    pulls in LightGBM: every ingest connection calls `init_db`.
    """
    init_log(conn)


def _naive_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment
    return moment.astimezone(UTC).replace(tzinfo=None)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- fail fast ---------------------------------------------------------------


def _precheck(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    runtime: datetime,
    naive_now: datetime,
) -> None:
    """Refuse, before any fit, what `append_partition` would refuse after it."""
    sealed = conn.execute(
        "select 1 from live.partition_seals where run_date = ?", [run_date]
    ).fetchone()
    logged = conn.execute(
        "select 1 from live.prediction_log where run_date = ? limit 1", [run_date]
    ).fetchone()
    if sealed is not None or logged is not None:
        msg = f"partition {run_date} is already sealed or populated; never re-appended"
        raise PartitionSealedError(msg)
    later = conn.execute(
        "select min(run_date) from live.partition_seals where run_date > ?",
        [run_date],
    ).fetchone()
    if later is not None and later[0] is not None:
        msg = (
            f"cannot issue {run_date}: {later[0]} is already sealed and the "
            "log is append-only in run_date order"
        )
        raise ValueError(msg)
    if runtime > naive_now:
        msg = f"run_date {run_date} has runtime_utc {runtime} after now ({naive_now})"
        raise ValueError(msg)


# -- loading -----------------------------------------------------------------


def _query(
    conn: duckdb.DuckDBPyConnection, sql: str, params: list[Any]
) -> pl.DataFrame:
    return conn.execute(sql, params).pl()


def _load_run_rows(conn: duckdb.DuckDBPyConnection, run_date: date) -> pl.DataFrame:
    rows = _query(
        conn,
        f"select {_PAIR_SELECT} from {_PAIR_TABLE} where run_date = ?",  # noqa: S608
        [run_date],
    )
    if rows.height == 0:
        msg = f"no rows in {_PAIR_TABLE} for run_date {run_date}"
        raise NoCanonicalRunError(msg)
    return rows


def _issuance_time(rows: pl.DataFrame, run_date: date) -> datetime:
    runtimes = rows.get_column("runtime_utc").unique().to_list()
    if len(runtimes) != 1:
        msg = (
            f"run_date {run_date} has rows with disagreeing runtime_utc values "
            "(mixed cycles)"
        )
        raise ValueError(msg)
    return _naive_utc(runtimes[0])


def _closed_pairs(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    *,
    before: datetime,
    since: datetime | None = None,
    after: datetime | None = None,
) -> pl.DataFrame:
    """Scorable pairs with `since <= window_end_utc < before` (and `> after`).

    The run date's own rows are excluded: they arrive with the run rows, so a
    pair of the run date that happens to be closed is seen once, as in the
    walk-forward. NO null filter: `walk_forward` and `update` count a pair
    with a null forecast (the baseline's `n_pairs`, the history's trim), so
    the bound must keep it.
    """
    clauses = ["scorable", "window_end_utc < ?", "run_date <> ?"]
    params: list[Any] = [before, run_date]
    if since is not None:
        clauses.append("window_end_utc >= ?")
        params.append(since)
    if after is not None:
        clauses.append("window_end_utc > ?")
        params.append(after)
    return _query(
        conn,
        f"select {_PAIR_SELECT} from {_PAIR_TABLE} where "  # noqa: S608
        + " and ".join(clauses),
        params,
    )


def _with_run_rows(closed: pl.DataFrame, run_rows: pl.DataFrame) -> pl.DataFrame:
    # No rechunk: walk_forward sorts (and so copies) anyway; the full-history
    # retrain night is the one that is tight on memory.
    return pl.concat([closed, run_rows], how="vertical", rechunk=False)


def _latest_close_before(
    conn: duckdb.DuckDBPyConnection, before: datetime
) -> datetime | None:
    row = conn.execute(
        f"select max(window_end_utc) from {_PAIR_TABLE} "  # noqa: S608
        "where scorable and window_end_utc < ?",
        [before],
    ).fetchone()
    return None if row is None or row[0] is None else _naive_utc(row[0])


def _bounded_history(
    conn: duckdb.DuckDBPyConnection, cutoff: datetime
) -> pl.DataFrame:
    """Exactly the pairs `ChallengerModel.fit` + `_trim` keep for `cutoff`.

    `fit` sees the scorable pairs closed before the retrain's issuance time,
    minus those with a null forecast or observation; `_trim` then keeps, per
    group, the pairs within `RETENTION_DAYS` of the group's own latest close
    plus the group's last `max(LAG_COUNTS)` by (window_end_utc, runtime_utc).
    Both are evaluated here, group by group, so a silent group keeps its last
    pairs however old they are.
    """
    last_n = max(LAG_COUNTS)
    return _query(
        conn,
        f"select {_PAIR_SELECT} from {_PAIR_TABLE} "  # noqa: S608
        "where scorable and window_end_utc < ? "
        "and forecast_f is not null and observed_f is not null "
        f"qualify window_end_utc >= max(window_end_utc) over "
        f"(partition by {_GROUP_SQL}) - interval {RETENTION_DAYS} day "
        f"or row_number() over (partition by {_GROUP_SQL} "
        f"order by window_end_utc desc, runtime_utc desc) <= {last_n}",
        [cutoff],
    )


# -- the artifact ------------------------------------------------------------


def _month_artifact(
    conn: duckdb.DuckDBPyConnection, run_date: date
) -> _Artifact | None:
    """The persisted booster governing `run_date`'s calendar month, if any.

    The month's retrain is its first ISSUED run date; an artifact only counts
    when that date was issued (sealed) or is `run_date` itself (a rerun after
    a crash between persisting and appending). An orphan from a crashed night
    that was never issued is skipped, so the next issued date retrains, as
    the walk-forward restricted to issued dates would.
    """
    row = conn.execute(
        "select retrained_on, cutoff_utc, trained_through, n_train_rows, "
        "model_text, model_sha256, region_codes, params, fit_seconds "
        "from live.challenger_models "
        "where retrained_on >= ? and retrained_on <= ? "
        "and (retrained_on = ? or retrained_on in "
        "(select run_date from live.partition_seals)) "
        "order by retrained_on limit 1",
        [run_date.replace(day=1), run_date, run_date],
    ).fetchone()
    if row is None:
        return None
    return _Artifact(*row)


def _persist(
    conn: duckdb.DuckDBPyConnection,
    *,
    run_date: date,
    cutoff: datetime,
    trained_through: datetime | None,
    model: ChallengerModel,
    now: datetime,
) -> _Artifact:
    text = model.dump_booster() if model.is_fitted else _UNFITTED_TEXT
    fit_seconds = float(sum(model.fit_seconds_log))
    artifact = _Artifact(
        retrained_on=run_date,
        cutoff_utc=cutoff,
        trained_through=trained_through,
        n_train_rows=int(model.n_train_rows or 0),
        model_text=text,
        model_sha256=_sha256(text),
        region_codes=json.dumps(model.region_codes, sort_keys=True),
        params=json.dumps(model.describe_params(), sort_keys=True),
        fit_seconds=fit_seconds,
    )
    conn.execute(
        "insert into live.challenger_models (retrained_on, cutoff_utc, "
        "trained_through, n_train_rows, model_text, model_sha256, region_codes, "
        "params, fit_seconds, fitted_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            artifact.retrained_on,
            artifact.cutoff_utc,
            artifact.trained_through,
            artifact.n_train_rows,
            artifact.model_text,
            artifact.model_sha256,
            artifact.region_codes,
            artifact.params,
            artifact.fit_seconds,
            now,
        ],
    )
    return artifact


def _verify_artifact(artifact: _Artifact, model_codes: dict[str, float]) -> None:
    if _sha256(artifact.model_text) != artifact.model_sha256:
        msg = (
            f"challenger model of {artifact.retrained_on} fails its sha256 "
            "check: the stored booster text is not what was persisted"
        )
        raise ModelArtifactError(msg)
    if json.loads(artifact.region_codes) != model_codes:
        msg = (
            f"challenger model of {artifact.retrained_on} was trained with "
            f"climate-region codes {artifact.region_codes}, but the registry "
            f"now gives {json.dumps(model_codes, sort_keys=True)}; reusing it "
            "would silently reinterpret the categorical feature"
        )
        raise ModelArtifactError(msg)


# -- the two sources ---------------------------------------------------------


class _DeltaAdapter:
    """Presents a restored model to `walk_forward` as a one-retrain model.

    `walk_forward` calls `fit` once, on the only (retrain) date, with the
    scorable pairs closed before `T` that are in the frame it was given: the
    delta since the artifact's cutoff. Applying it through the model's own
    `update` is the walk-forward's non-retrain path; prediction, null checks
    and output construction stay in `walk_forward`.
    """

    name = ChallengerModel.name

    def __init__(self, model: ChallengerModel, cutoff: datetime) -> None:
        self._model = model
        self._cutoff = cutoff

    def fit(self, training_pairs: pl.DataFrame) -> None:
        # Anything before the cutoff is already in the restored history.
        self._model.update(
            training_pairs.filter(pl.col("window_end_utc") >= self._cutoff)
        )

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        return self._model.predict_detail(run_rows)


def _log_rows(
    result: pl.DataFrame,
    run_rows: pl.DataFrame,
    params: str,
    *,
    retrained_on: date | None = None,
    trained_through: datetime | None = None,
    override: bool = False,
) -> pl.DataFrame:
    """A `walk_forward` result in the layout `append_partition` expects."""
    closes = run_rows.select(
        pl.col("station"),
        pl.col("lead_day").cast(pl.Int64),
        pl.col("variable"),
        pl.col("window_end_utc"),
    )
    frame = result.with_columns(pl.col("lead_day").cast(pl.Int64)).join(
        closes, on=["station", "lead_day", "variable"], how="left"
    )
    if frame.height != result.height:
        msg = "run rows are not unique per (station, lead_day, variable)"
        raise ValueError(msg)
    if override:
        frame = frame.with_columns(
            pl.lit(retrained_on, dtype=pl.Date).alias("retrained_on"),
            pl.lit(trained_through, dtype=pl.Datetime("us")).alias("trained_through"),
        )
    for name in ("runtime_utc", "trained_through", "window_end_utc"):
        dtype = frame.schema[name]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
            frame = frame.with_columns(
                pl.col(name).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            )
    return frame.with_columns(pl.lit(params).alias("params"))


def _baseline_rows(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    run_rows: pl.DataFrame,
    runtime: datetime,
) -> pl.DataFrame:
    model = BaselineModel()
    anchor = _latest_close_before(conn, runtime)
    if anchor is None:
        closed = run_rows.clear()
    else:
        # `fit` keeps `window_end_utc > anchor - window_days` (strict) of the
        # prefix whose last element is `anchor`; nothing older can matter.
        closed = _closed_pairs(
            conn,
            run_date,
            before=runtime,
            after=anchor - timedelta(days=model.window_days),
        )
    frame = _with_run_rows(closed, run_rows)
    result = walk_forward(frame, model, run_date, run_date, retrain="daily")
    params = json.dumps(
        {
            "model": model.name,
            "window_days": model.window_days,
            "min_pairs": model.min_pairs,
        },
        sort_keys=True,
    )
    return _log_rows(result, run_rows, params)


@dataclass(frozen=True)
class _ChallengerNight:
    rows: pl.DataFrame
    retrained: bool
    retrained_on: date
    fit_seconds: float
    n_history_rows: int


def _challenger_rows(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    run_rows: pl.DataFrame,
    runtime: datetime,
    *,
    now: datetime,
    station_features: pl.DataFrame,
    min_train_rows: int,
) -> _ChallengerNight:
    artifact = _month_artifact(conn, run_date)
    if artifact is None:
        return _retrain_night(
            conn,
            run_date,
            run_rows,
            runtime,
            now=now,
            station_features=station_features,
            min_train_rows=min_train_rows,
        )

    # Both checks come before any history is read or any text is parsed.
    codes = ChallengerModel(
        station_features, min_train_rows=min_train_rows
    ).region_codes
    _verify_artifact(artifact, codes)
    history = _bounded_history(conn, artifact.cutoff_utc)
    text = None if artifact.model_text == _UNFITTED_TEXT else artifact.model_text
    model = ChallengerModel.restore(
        station_features, text, history, min_train_rows=min_train_rows
    )
    delta = _closed_pairs(
        conn, run_date, before=runtime, since=artifact.cutoff_utc
    )
    frame = _with_run_rows(delta, run_rows)
    result = walk_forward(
        frame,
        _DeltaAdapter(model, artifact.cutoff_utc),
        run_date,
        run_date,
        retrain="monthly",
    )
    params = _challenger_params(artifact)
    rows = _log_rows(
        result,
        run_rows,
        params,
        retrained_on=artifact.retrained_on,
        trained_through=artifact.trained_through,
        override=True,
    )
    just_retrained = artifact.retrained_on == run_date
    return _ChallengerNight(
        rows=rows,
        retrained=just_retrained,
        retrained_on=artifact.retrained_on,
        fit_seconds=artifact.fit_seconds if just_retrained else 0.0,
        n_history_rows=history.height,
    )


def _challenger_params(artifact: _Artifact) -> str:
    recorded = json.loads(artifact.params)
    recorded["model_sha256"] = artifact.model_sha256
    recorded["retrained_on"] = artifact.retrained_on.isoformat()
    return json.dumps(recorded, sort_keys=True)


def _retrain_night(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    run_rows: pl.DataFrame,
    runtime: datetime,
    *,
    now: datetime,
    station_features: pl.DataFrame,
    min_train_rows: int,
) -> _ChallengerNight:
    model = ChallengerModel(station_features, min_train_rows=min_train_rows)
    closed = _closed_pairs(conn, run_date, before=runtime)
    n_history = closed.height
    frame = _with_run_rows(closed, run_rows)
    del closed
    result = walk_forward(frame, model, run_date, run_date, retrain="monthly")
    del frame

    trained_through = result.get_column("trained_through")[0]
    trained_naive = None if trained_through is None else _naive_utc(trained_through)
    artifact = _persist(
        conn,
        run_date=run_date,
        cutoff=runtime,
        trained_through=trained_naive,
        model=model,
        now=now,
    )
    rows = _log_rows(result, run_rows, _challenger_params(artifact))
    return _ChallengerNight(
        rows=rows,
        retrained=True,
        retrained_on=run_date,
        fit_seconds=artifact.fit_seconds,
        n_history_rows=n_history,
    )


# -- the night ---------------------------------------------------------------


def issue_predictions(
    conn: duckdb.DuckDBPyConnection,
    run_date: date,
    *,
    now: datetime,
    station_features: pl.DataFrame,
    min_train_rows: int = challenger_module.MIN_TRAIN_ROWS,
) -> IssueResult:
    """Append `run_date`'s baseline and challenger predictions, and seal it.

    Everything that can fail cheaply does so before a model is fit: an
    unknown or mixed-cycle run date, an already-sealed partition, a later
    seal, a `now` before the run's own issuance. Errors from the append (a
    non-finite prediction, a null) are not swallowed; nothing is written for
    a failed append, but a challenger retrain's booster stays persisted, and
    a rerun of the same date reuses it (identical by determinism).

    Rows whose window already closed at `now` are rejected by the log's
    strict late rule and reported in `result.append.late_keys`.
    """
    started = time.monotonic()
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        msg = f"`now` must be a timezone-aware UTC datetime, got {now!r}"
        raise ValueError(msg)
    naive_now = _naive_utc(now)

    run_rows = _load_run_rows(conn, run_date)
    runtime = _issuance_time(run_rows, run_date)
    _precheck(conn, run_date, runtime, naive_now)

    baseline = _baseline_rows(conn, run_date, run_rows, runtime)
    night = _challenger_rows(
        conn,
        run_date,
        run_rows,
        runtime,
        now=naive_now,
        station_features=station_features,
        min_train_rows=min_train_rows,
    )
    combined = pl.concat([baseline, night.rows], how="vertical")
    appended = append_partition(conn, run_date, combined, now=now)
    return IssueResult(
        run_date=run_date,
        append=appended,
        retrained=night.retrained,
        challenger_retrained_on=night.retrained_on,
        fit_seconds=night.fit_seconds,
        n_history_rows=night.n_history_rows,
        wall_seconds=time.monotonic() - started,
    )
