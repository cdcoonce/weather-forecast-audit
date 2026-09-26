"""Platform smoke checks for the production host.

rammingspeed is an Ivy Bridge i5 with no AVX2. Wheels built for a newer
baseline die there with an illegal instruction, usually at import or first
use. Each check here exercises the native code path the pipeline depends on:
DuckDB (window query plus Parquet round trip), LightGBM (fit and predict),
and Polars (group-by, via the `rtcompat` build).
"""

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm
import numpy as np
import polars as pl

CPUINFO = Path("/proc/cpuinfo")


class PlatformSmokeError(AssertionError):
    """A platform check ran but produced a wrong result."""


@dataclass(frozen=True)
class DuckDBResult:
    rows: int
    running_totals: list[int]


@dataclass(frozen=True)
class LightGBMResult:
    deterministic: bool
    r2: float


@dataclass(frozen=True)
class SmokeReport:
    duckdb: DuckDBResult
    lightgbm: LightGBMResult
    polars: dict[str, int]
    polars_runtime: str
    avx2: bool | None
    versions: dict[str, str]


def duckdb_window_parquet_roundtrip(workdir: Path) -> DuckDBResult:
    database = workdir / "platform_smoke.duckdb"
    parquet = workdir / "platform_smoke.parquet"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            """
            create or replace table readings as
            select * from (values
                ('a', 1, 1), ('a', 2, 2), ('a', 3, 3), ('a', 4, 4),
                ('b', 1, 20), ('b', 2, 30)
            ) as t(station, day, value)
            """
        )
        connection.execute(
            """
            create or replace table running as
            select
                station,
                day,
                -- sum() of INTEGER is HUGEINT, which Parquet stores as DOUBLE.
                cast(sum(value) over (order by station, day) as bigint)
                    as running_total
            from readings
            """
        )
        connection.execute(f"copy running to '{parquet}' (format parquet)")
        rows = connection.execute(
            f"select running_total from read_parquet('{parquet}') order by station, day"
        ).fetchall()
    return DuckDBResult(rows=len(rows), running_totals=[row[0] for row in rows])


def assert_identical(first: np.ndarray, second: np.ndarray) -> None:
    if not np.array_equal(first, second):
        max_diff = float(np.max(np.abs(first - second)))
        msg = f"LightGBM predictions not deterministic (max diff {max_diff:.3g})"
        raise PlatformSmokeError(msg)


def lightgbm_deterministic_fit(seed: int = 20260926) -> LightGBMResult:
    rng = np.random.default_rng(seed)
    features = rng.uniform(-5, 5, size=(500, 3))
    target = 3 * features[:, 0] + rng.normal(0, 0.5, size=500)
    params = {
        "objective": "regression",
        "learning_rate": 0.1,
        "num_leaves": 15,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbose": -1,
    }

    predictions = [
        lightgbm.train(
            params, lightgbm.Dataset(features, label=target), num_boost_round=50
        ).predict(features)
        for _ in range(2)
    ]
    assert_identical(*predictions)

    residual = np.sum((target - predictions[0]) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return LightGBMResult(deterministic=True, r2=float(1 - residual / total))


def polars_group_by() -> dict[str, int]:
    frame = pl.DataFrame({"key": ["a", "a", "b", "b"], "value": [1, 2, 5, 7]})
    totals = frame.group_by("key").agg(pl.col("value").sum()).sort("key")
    return dict(zip(totals["key"].to_list(), totals["value"].to_list(), strict=True))


def polars_runtime() -> str:
    """Which Polars native runtime loaded: `compat`, `64`, or `32`."""
    module = pl._plr.__name__  # e.g. _polars_runtime_compat._polars_runtime
    return module.split(".", 1)[0].removeprefix("_polars_runtime_")


def cpu_has_avx2(cpuinfo: Path = CPUINFO) -> bool | None:
    """Whether the CPU advertises AVX2; None where /proc/cpuinfo is absent."""
    if not cpuinfo.exists():
        return None
    for line in cpuinfo.read_text().splitlines():
        if line.startswith("flags"):
            return "avx2" in line.split(":", 1)[1].split()
    return None


def run_platform_smoke(workdir: Path) -> SmokeReport:
    return SmokeReport(
        duckdb=duckdb_window_parquet_roundtrip(workdir),
        lightgbm=lightgbm_deterministic_fit(),
        polars=polars_group_by(),
        polars_runtime=polars_runtime(),
        avx2=cpu_has_avx2(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("duckdb", "lightgbm", "polars", "numpy")
        },
    )
