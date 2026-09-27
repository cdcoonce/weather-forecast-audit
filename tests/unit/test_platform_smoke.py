from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from weather_forecast_audit import platform_smoke

pytestmark = pytest.mark.unit


@pytest.mark.io
def test_duckdb_window_query_round_trips_through_parquet(tmp_path: Path) -> None:
    result = platform_smoke.duckdb_window_parquet_roundtrip(tmp_path)

    assert result.rows == 6
    assert result.running_totals == [1, 3, 6, 10, 30, 60]
    assert all(type(total) is int for total in result.running_totals)
    assert (tmp_path / "platform_smoke.duckdb").exists()
    assert (tmp_path / "platform_smoke.parquet").exists()


@pytest.mark.io
def test_polars_duckdb_insert_round_trips_types_and_nulls(tmp_path: Path) -> None:
    result = platform_smoke.polars_duckdb_insert(tmp_path)

    assert result.rows == 2
    assert result.stations == ["DEN", "PHX"]
    assert result.valid_utc == [
        datetime(2026, 9, 26, 19, 0, 0),  # noqa: DTZ001 (naive on purpose)
        datetime(2026, 9, 26, 18, 0, 0),  # noqa: DTZ001 (naive on purpose)
    ]
    assert result.local_date == [date(2026, 9, 26), date(2026, 9, 26)]
    # DEN's tmpf is the deliberate null; PHX's must round-trip exactly.
    assert result.tmpf == [None, 98.6]
    assert result.n_obs == [18, 24]
    assert (tmp_path / "platform_smoke_insert.duckdb").exists()


def test_lightgbm_fit_is_deterministic_and_learns() -> None:
    result = platform_smoke.lightgbm_deterministic_fit(seed=7)

    assert result.deterministic
    # A fit on y = 3x + noise must beat predicting the mean by a wide margin.
    assert result.r2 > 0.9


def test_determinism_check_rejects_differing_predictions() -> None:
    a = np.array([1.0, 2.0, 3.0])

    with pytest.raises(platform_smoke.PlatformSmokeError, match="not deterministic"):
        platform_smoke.assert_identical(a, a + 1e-9)


def test_polars_group_by_aggregates() -> None:
    assert platform_smoke.polars_group_by() == {"a": 3, "b": 12}


def test_polars_loads_the_compat_runtime() -> None:
    # The default runtimes assume AVX2 and crash on rammingspeed's Ivy Bridge.
    assert platform_smoke.polars_runtime() == "compat"


def test_cpu_flags_report_avx2_from_cpuinfo(tmp_path: Path) -> None:
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu sse4_2 avx\n")

    assert platform_smoke.cpu_has_avx2(cpuinfo) is False
    cpuinfo.write_text("processor\t: 0\nflags\t\t: fpu sse4_2 avx avx2\n")
    assert platform_smoke.cpu_has_avx2(cpuinfo) is True


def test_cpu_flags_unknown_without_cpuinfo(tmp_path: Path) -> None:
    assert platform_smoke.cpu_has_avx2(tmp_path / "missing") is None


@pytest.mark.io
def test_run_platform_smoke_reports_every_check(tmp_path: Path) -> None:
    report = platform_smoke.run_platform_smoke(tmp_path)

    assert report.duckdb.rows == 6
    assert report.lightgbm.deterministic
    assert report.polars == {"a": 3, "b": 12}
    assert report.polars_runtime == "compat"
    assert report.polars_duckdb_insert.rows == 2
    assert report.polars_duckdb_insert.tmpf == [None, 98.6]
    assert set(report.versions) == {
        "duckdb",
        "lightgbm",
        "polars",
        "numpy",
        "pyarrow",
    }
