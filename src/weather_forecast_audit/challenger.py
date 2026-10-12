"""LightGBM challenger corrector (issue #19): the model that has to beat the
rolling-bias baseline (PRD module 7).

`ChallengerModel` regresses the signed error `forecast_f - observed_f` on
guidance-side features known at issuance (the NBS temperature and spread,
cycle, lead, variable, season), station metadata (latitude, longitude,
elevation, coastal flag, climate region) and lagged errors of the same
(station, lead_day, variable) group. The corrected forecast is
`forecast_f - predicted_error`. Every constant below is pre-registered and
fixed before any challenger skill is computed; nothing is tuned on
evaluation skill (see docs/methodology.md, "Challenger correction
(LightGBM)").

Lagged errors are the only features that touch verification outcomes, so
they carry the leakage risk. They are built by exactly two pure functions,
`pair_stats` and `asof_lag_features`, used by both `fit` (each training row,
as of its own issuance time) and `predict_detail` (as of the run's issuance
time, over the history the model maintains through the evaluator's `update`
hook). A feature is that of the group's latest pair whose verification
window CLOSED strictly before the row's `runtime_utc`; the pair's age is a
feature too, so a stale value is visible to the model rather than disguised
as fresh.

No I/O, no DuckDB, no dbt: a plain `weather_forecast_audit.walkforward.Model`
implementation, driven only through `walk_forward`.
"""

import importlib.metadata
import time
from typing import Any

import lightgbm
import numpy as np
import polars as pl

# -- pre-registered constants (never tuned on evaluation skill) ----------------

N_ESTIMATORS = 200
"""Boosting rounds. Modest capacity for a noisy ~2-3 F target; no early
stopping, because a validation split of a time-ordered, serially correlated
history is itself a leakage surface."""

LEARNING_RATE = 0.05
"""Small step so 200 rounds are a shrunk, smooth correction rather than a fit
to noise."""

NUM_LEAVES = 31
"""LightGBM's default tree size; interactions of a few features are enough."""

MIN_DATA_IN_LEAF = 500
"""Large leaves: the target is mostly noise, and a station-lead-season cell
must be supported by hundreds of pairs before it earns its own correction."""

FEATURE_FRACTION = 0.8
"""Column subsampling per tree, to decorrelate trees given the strongly
related temperature features."""

BAGGING_FRACTION = 0.8
"""Row subsampling per iteration (with BAGGING_FREQ), a variance control."""

BAGGING_FREQ = 1
"""Resample rows every iteration."""

LAMBDA_L2 = 10.0
"""L2 shrinkage of leaf values; with an L1 objective it damps the leaf
medians of sparse cells."""

MAX_BIN = 63
"""Histogram resolution: 63 bins resolve a temperature in about 1.5 F steps
and keep the Dataset small on a memory-constrained host."""

SEED = 42
"""Single seed for bagging and feature sampling (the #4 platform-proof
settings: deterministic, row-wise, single thread)."""

MIN_TRAIN_ROWS = 20_000
"""Below this many scorable pairs the model is unfitted and every row falls
back to raw NBM, flagged."""

LAG_WINDOW_DAYS = 30
"""Span of the pair-count feature, matching the baseline's window W."""

LAG_COUNTS = (3, 7)
"""Window sizes (in pairs) of the lagged mean-error features."""

# Not a pre-registered modelling constant: how much closed history the
# maintained state keeps per group beyond what the features strictly need
# (30 days of count and the last 7 pairs), as a margin.
RETENTION_DAYS = 40

FEATURE_COLUMNS = (
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
"""The model's input columns, in order. `climate_region_code` is the only
categorical feature; `is_max` and `coastal_flag` are 0/1."""

_CATEGORICAL_FEATURES = ("climate_region_code",)

GROUP_KEYS = ("station", "lead_day", "variable")

_LAG_COLUMNS = ("lag_error", "lag_mean3", "lag_mean7", "lag_n30", "lag_age_days")

_STATION_FEATURE_COLUMNS = (
    "station",
    "latitude",
    "longitude",
    "elevation_m",
    "coastal_flag",
    "climate_region",
)

# The only columns of a training pair the model touches. Selecting them first
# keeps the working set small on the full national history.
_TRAIN_COLUMNS = (
    "station",
    "runtime_utc",
    "cycle_hour",
    "lead_day",
    "variable",
    "target_date",
    "forecast_f",
    "spread_f",
    "tmp_f",
    "window_end_utc",
    "observed_f",
)

_HISTORY_SCHEMA = {
    "station": pl.Utf8,
    "lead_day": pl.Int64,
    "variable": pl.Utf8,
    "runtime_utc": pl.Datetime("us"),
    "window_end_utc": pl.Datetime("us"),
    "forecast_f": pl.Float64,
    "observed_f": pl.Float64,
}

_NAN = float("nan")
_MICROSECONDS_PER_DAY = 86_400_000_000


def booster_params() -> dict[str, Any]:
    """The exact LightGBM parameter dict (excluding the round count)."""
    return {
        "objective": "regression_l1",
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "feature_fraction": FEATURE_FRACTION,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": BAGGING_FREQ,
        "lambda_l2": LAMBDA_L2,
        "max_bin": MAX_BIN,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": SEED,
        "verbose": -1,
    }


# -- the legal lagged-error features -----------------------------------------


def _naive_utc(frame: pl.DataFrame, column: str) -> pl.Expr:
    """`column` as a timezone-naive UTC microsecond datetime.

    DuckDB hands back naive or UTC-aware timestamps depending on the column
    type; `join_asof` needs both sides to be the same dtype.
    """
    expr = pl.col(column)
    if getattr(frame.schema[column], "time_zone", None) is not None:
        expr = expr.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return expr.cast(pl.Datetime("us")).alias(column)


def _mean_of_last(n: int) -> pl.Expr:
    """Mean of the group's last `n` errors, the pair itself included.

    Built from shifted copies rather than a sliding-sum window, so the value
    for a pair depends only on its own `n` predecessors and never on how much
    older history happens to be in the frame (exactly reproducible between a
    trimmed maintained history and a full one). Fewer than `n` predecessors
    give the mean of those available.
    """
    error = pl.col("lag_error")
    terms = [error] + [error.shift(k).over(list(GROUP_KEYS)) for k in range(1, n)]
    return pl.mean_horizontal(terms)


def pair_stats(history: pl.DataFrame) -> pl.DataFrame:
    """Per-pair "through-p" error statistics for every pair of `history`.

    `history` needs `station`, `lead_day`, `variable`, `runtime_utc`,
    `window_end_utc`, `forecast_f` and `observed_f`; a pair with no observed
    value is ignored. For each pair p the frame carries, as of the moment p's
    window closed (p itself included):

    * `lag_error`: p's signed error `forecast_f - observed_f`;
    * `lag_mean3`, `lag_mean7`: mean of the group's last 3 / 7 errors;
    * `lag_n30`: number of the group's pairs in the `LAG_WINDOW_DAYS` days
      ending at p's `window_end_utc` (the window is `(end - 30d, end]`).

    A group is (station, lead_day, variable). Within a group pairs are ordered
    by (`window_end_utc`, `runtime_utc`); "last n" is positional in that
    order, so pairs sharing a close time are ordered deterministically.

    The result is sorted by `window_end_utc` (then the group keys and
    `runtime_utc`), the order `asof_lag_features` joins on. `_known_utc`
    repeats `window_end_utc` so the matched pair's close time survives the
    as-of join (polars drops the right-hand `on` column).
    """
    ordered = (
        history.select(
            pl.col("station"),
            pl.col("lead_day").cast(pl.Int64),
            pl.col("variable"),
            _naive_utc(history, "runtime_utc"),
            _naive_utc(history, "window_end_utc"),
            (pl.col("forecast_f") - pl.col("observed_f"))
            .cast(pl.Float64)
            .alias("lag_error"),
        )
        .filter(pl.col("lag_error").is_not_null())
        .sort([*GROUP_KEYS, "window_end_utc", "runtime_utc"], maintain_order=True)
    )

    n30 = np.empty(ordered.height, dtype=np.float64)
    if ordered.height:
        closes = ordered.get_column("window_end_utc").to_numpy()
        lengths = (
            ordered.group_by(list(GROUP_KEYS), maintain_order=True)
            .len()
            .get_column("len")
            .to_numpy()
        )
        window = np.timedelta64(LAG_WINDOW_DAYS, "D")
        position = 0
        for length in lengths:
            segment = closes[position : position + length]
            first_inside = np.searchsorted(segment, segment - window, side="right")
            n30[position : position + length] = np.arange(1, length + 1) - first_inside
            position += int(length)

    stats = ordered.with_columns(
        _mean_of_last(LAG_COUNTS[0]).alias("lag_mean3"),
        _mean_of_last(LAG_COUNTS[1]).alias("lag_mean7"),
        pl.Series("lag_n30", n30, dtype=pl.Float64),
        pl.col("window_end_utc").alias("_known_utc"),
    )
    return stats.select(
        *GROUP_KEYS,
        "runtime_utc",
        "window_end_utc",
        "_known_utc",
        "lag_error",
        "lag_mean3",
        "lag_mean7",
        "lag_n30",
    ).sort(
        ["window_end_utc", *GROUP_KEYS, "runtime_utc"], maintain_order=True
    )


def asof_lag_features(queries: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    """Lag features of each query row, as of its `runtime_utc`, strictly.

    `queries` needs `station`, `lead_day`, `variable` and `runtime_utc`;
    `stats` is `pair_stats`' output. For each query the matched pair is the
    group's latest pair whose `window_end_utc` is STRICTLY before the query's
    `runtime_utc` (a pair closing exactly at issuance is not known yet).
    Returns, aligned positionally with `queries`: `lag_error`, `lag_mean3`,
    `lag_mean7`, `lag_n30` (that pair's through-p statistics) and
    `lag_age_days` (`runtime_utc` minus that pair's close, in days). All five
    are NaN when the group has no earlier pair.

    `join_asof` requires both sides globally sorted on the join key, so both
    are sorted explicitly here (and the query order restored from a row
    index), whatever order the caller holds them in.
    """
    left = (
        queries.select(
            pl.col("station"),
            pl.col("lead_day").cast(pl.Int64),
            pl.col("variable"),
            _naive_utc(queries, "runtime_utc"),
        )
        .with_row_index("_row")
        .sort("runtime_utc", maintain_order=True)
    )
    right = stats.select(
        *GROUP_KEYS,
        "window_end_utc",
        "_known_utc",
        "lag_error",
        "lag_mean3",
        "lag_mean7",
        "lag_n30",
    ).sort("window_end_utc", maintain_order=True)

    joined = left.join_asof(
        right,
        left_on="runtime_utc",
        right_on="window_end_utc",
        by=list(GROUP_KEYS),
        strategy="backward",
        allow_exact_matches=False,
        # Both sides are sorted explicitly above; polars cannot verify
        # sortedness itself when `by` groups are given.
        check_sortedness=False,
    ).sort("_row")

    age_days = (
        (pl.col("runtime_utc") - pl.col("_known_utc")).dt.total_microseconds()
        / _MICROSECONDS_PER_DAY
    ).alias("lag_age_days")
    return joined.select(
        pl.col("lag_error").cast(pl.Float64).fill_null(_NAN),
        pl.col("lag_mean3").cast(pl.Float64).fill_null(_NAN),
        pl.col("lag_mean7").cast(pl.Float64).fill_null(_NAN),
        pl.col("lag_n30").cast(pl.Float64).fill_null(_NAN),
        age_days.cast(pl.Float64).fill_null(_NAN),
    )


# -- the model -----------------------------------------------------------------


class ChallengerModel:
    """LightGBM error corrector with legal lagged-error features, with fallback."""

    name = "challenger"

    def __init__(
        self,
        station_features: pl.DataFrame,
        *,
        min_train_rows: int = MIN_TRAIN_ROWS,
    ) -> None:
        for column in _STATION_FEATURE_COLUMNS:
            if column not in station_features.columns:
                msg = f"station_features is missing required column: {column}"
                raise ValueError(msg)
        if station_features["station"].n_unique() != station_features.height:
            msg = "station_features must have one row per station"
            raise ValueError(msg)

        self.min_train_rows = min_train_rows
        regions = sorted(
            {r for r in station_features["climate_region"].to_list() if r is not None}
        )
        self._region_codes = {region: float(i) for i, region in enumerate(regions)}
        self._station_table = pl.DataFrame(
            {
                "station": station_features["station"],
                "latitude": station_features["latitude"].cast(pl.Float32),
                "longitude": station_features["longitude"].cast(pl.Float32),
                "elevation_m": station_features["elevation_m"].cast(pl.Float32),
                "coastal_flag": station_features["coastal_flag"].cast(pl.Float32),
                "climate_region_code": pl.Series(
                    [
                        self._region_codes.get(r)
                        for r in station_features["climate_region"].to_list()
                    ],
                    dtype=pl.Float32,
                ),
            }
        )
        self._booster: lightgbm.Booster | None = None
        self._history = pl.DataFrame(schema=_HISTORY_SCHEMA)
        self._stats: pl.DataFrame | None = None
        self.fit_seconds_log: list[float] = []
        self.fit_count = 0

    # -- state ---------------------------------------------------------------

    @property
    def history(self) -> pl.DataFrame:
        """The maintained closed-pair history the lag features are read from."""
        return self._history

    @property
    def is_fitted(self) -> bool:
        return self._booster is not None

    def describe_params(self) -> dict[str, Any]:
        """Every pre-registered constant plus the runtime versions, for `params`."""
        return {
            "model": self.name,
            "n_estimators": N_ESTIMATORS,
            **booster_params(),
            "min_train_rows": self.min_train_rows,
            "lag_window_days": LAG_WINDOW_DAYS,
            "lag_counts": list(LAG_COUNTS),
            "retention_days": RETENTION_DAYS,
            "feature_columns": list(FEATURE_COLUMNS),
            "retrain": "monthly",
            "lightgbm_version": lightgbm.__version__,
            "polars_version": importlib.metadata.version("polars"),
        }

    @staticmethod
    def _as_history(pairs: pl.DataFrame) -> pl.DataFrame:
        return pairs.select(
            pl.col("station"),
            pl.col("lead_day").cast(pl.Int64),
            pl.col("variable"),
            _naive_utc(pairs, "runtime_utc"),
            _naive_utc(pairs, "window_end_utc"),
            pl.col("forecast_f").cast(pl.Float64),
            pl.col("observed_f").cast(pl.Float64),
        ).filter(pl.col("observed_f").is_not_null())

    def _trim(self) -> None:
        """Bound the maintained history without changing any feature.

        Keeps, per group, the pairs closed within `RETENTION_DAYS` of the
        group's own latest close, plus its last `max(LAG_COUNTS)` pairs. The
        features of a group's latest pair need only its last 7 pairs and 30
        days, so this is exact for every query at or after the latest close
        (the evaluator's contract) even for a sparse or silent group; a
        station that stopped reporting keeps its last pairs and their age.
        """
        history = self._history.sort(
            [*GROUP_KEYS, "window_end_utc", "runtime_utc"], maintain_order=True
        )
        group = list(GROUP_KEYS)
        within_horizon = pl.col("window_end_utc") >= (
            pl.col("window_end_utc").max().over(group)
            - pl.duration(days=RETENTION_DAYS)
        )
        among_last = pl.int_range(pl.len(), 0, -1).over(group) <= max(LAG_COUNTS)
        keep = history.with_columns((within_horizon | among_last).alias("_keep"))
        self._history = keep.filter(pl.col("_keep")).drop("_keep")
        self._stats = None

    def _current_stats(self) -> pl.DataFrame:
        if self._stats is None:
            self._stats = pair_stats(self._history)
        return self._stats

    # -- Model protocol ------------------------------------------------------

    def fit(self, training_pairs: pl.DataFrame) -> None:
        """Reset state, rebuild the history, and train if there is enough data.

        `training_pairs` are the scorable pairs with `window_end_utc < T`,
        sorted ascending by `window_end_utc` (the evaluator's contract).
        Below `min_train_rows` the model is left unfitted: every prediction
        falls back to raw `forecast_f`, flagged.
        """
        started = time.monotonic()
        self.fit_count += 1
        self._booster = None

        columns = [c for c in _TRAIN_COLUMNS if c in training_pairs.columns]
        rows = training_pairs.select(columns).filter(
            pl.col("observed_f").is_not_null()
        )
        self._history = self._as_history(rows)
        self._stats = None

        if rows.height >= self.min_train_rows:
            self._booster = self._train(rows)
            self.fit_seconds_log.append(time.monotonic() - started)
        self._trim()

    def update(self, new_pairs: pl.DataFrame) -> None:
        """Absorb newly closed pairs into the maintained history; no refit."""
        if new_pairs.height == 0:
            return
        addition = self._as_history(new_pairs)
        self._history = pl.concat([self._history, addition])
        self._trim()

    def lag_features(self, rows: pl.DataFrame) -> pl.DataFrame:
        """The five lag features of `rows` over the maintained history."""
        return asof_lag_features(rows, self._current_stats())

    def predict_detail(self, run_rows: pl.DataFrame) -> pl.DataFrame:
        """Corrected `forecast_f` and `fallback`, aligned to `run_rows`.

        Unfitted: raw `forecast_f` and `fallback = true` everywhere. Fitted:
        `forecast_f - predicted_error`, `fallback = false`. Missing inputs
        (an absent `tmp_f` or `cycle_hour`, an unknown station, a group with
        no earlier pair) are NaN for the booster to route as missing, never
        fabricated zeros. A non-finite prediction (not expected) leaves that
        row at raw and flags it.
        """
        raw = run_rows.get_column("forecast_f").cast(pl.Float64)
        n = run_rows.height
        if self._booster is None or n == 0:
            return pl.DataFrame(
                {
                    "forecast_f": raw,
                    "fallback": pl.Series(
                        [self._booster is None] * n, dtype=pl.Boolean
                    ),
                }
            )

        features = self._feature_frame(run_rows, self.lag_features(run_rows))
        predicted_error = self._booster.predict(features.to_numpy(order="c"))
        predicted = pl.Series("predicted_error", predicted_error, dtype=pl.Float64)
        usable = predicted.is_finite()
        corrected = pl.select(
            pl.when(usable).then(raw - predicted).otherwise(raw).alias("forecast_f")
        ).get_column("forecast_f")
        return pl.DataFrame({"forecast_f": corrected, "fallback": ~usable})

    def predict(self, run_rows: pl.DataFrame) -> pl.Series:
        return self.predict_detail(run_rows)["forecast_f"]

    # -- features and training -------------------------------------------------

    def _feature_frame(self, rows: pl.DataFrame, lags: pl.DataFrame) -> pl.DataFrame:
        """`FEATURE_COLUMNS` as float32, aligned to `rows`; missing -> NaN."""

        def numeric(name: str) -> pl.Expr:
            if name not in rows.columns:
                return pl.lit(_NAN, dtype=pl.Float32).alias(name)
            return pl.col(name).cast(pl.Float32).fill_null(_NAN)

        if "target_date" in rows.columns:
            angle = (
                pl.col("target_date").dt.ordinal_day().cast(pl.Float64)
                * (2.0 * np.pi / 365.25)
            )
            doy_sin = angle.sin().cast(pl.Float32).fill_null(_NAN).alias("doy_sin")
            doy_cos = angle.cos().cast(pl.Float32).fill_null(_NAN).alias("doy_cos")
        else:
            doy_sin = pl.lit(_NAN, dtype=pl.Float32).alias("doy_sin")
            doy_cos = pl.lit(_NAN, dtype=pl.Float32).alias("doy_cos")

        is_max = (
            pl.when(pl.col("variable") == "max")
            .then(pl.lit(1.0, dtype=pl.Float32))
            .when(pl.col("variable") == "min")
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(pl.lit(_NAN, dtype=pl.Float32))
            .alias("is_max")
        )
        guidance = rows.select(
            pl.col("station"),
            numeric("forecast_f"),
            numeric("spread_f"),
            numeric("tmp_f"),
            numeric("cycle_hour"),
            numeric("lead_day"),
            is_max,
            doy_sin,
            doy_cos,
        )
        with_station = guidance.join(
            self._station_table, on="station", how="left", maintain_order="left"
        ).with_columns(
            pl.col("latitude").fill_null(_NAN),
            pl.col("longitude").fill_null(_NAN),
            pl.col("elevation_m").fill_null(_NAN),
            pl.col("coastal_flag").fill_null(_NAN),
            pl.col("climate_region_code").fill_null(_NAN),
        )
        lag_frame = lags.select(pl.col(c).cast(pl.Float32) for c in _LAG_COLUMNS)
        return with_station.hstack(lag_frame).select(list(FEATURE_COLUMNS))

    def _train(self, rows: pl.DataFrame) -> lightgbm.Booster:
        """Fit on `rows`, each with its lag features as of its own issuance."""
        lags = asof_lag_features(rows, pair_stats(self._history))
        features = self._feature_frame(rows, lags)
        del lags
        matrix = features.to_numpy(order="c")
        del features
        target = (
            (rows.get_column("forecast_f") - rows.get_column("observed_f"))
            .cast(pl.Float64)
            .to_numpy()
        )
        dataset = lightgbm.Dataset(
            matrix,
            label=target,
            feature_name=list(FEATURE_COLUMNS),
            categorical_feature=list(_CATEGORICAL_FEATURES),
            free_raw_data=True,
        )
        return lightgbm.train(booster_params(), dataset, num_boost_round=N_ESTIMATORS)
