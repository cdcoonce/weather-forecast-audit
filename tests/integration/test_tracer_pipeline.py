"""End-to-end tracer: fixture-served ingest -> dbt build -> known answers.

FakeFetcher-served fixtures stand in for the three live runs the tracer
script makes for real (`scripts/tracer_kphx.sh`): the hand-checked KPHX
2023-07-14 run, plus one run either side of the 2026-04-30 cycle changeover.
"""

import csv
import json
import os
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import pytest
from dbt.cli.main import dbtRunner

from weather_forecast_audit import warehouse
from weather_forecast_audit.iem.http import HttpResponse
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry

pytestmark = [pytest.mark.integration, pytest.mark.io, pytest.mark.dbt]

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "iem"
DBT_DIR = REPO / "dbt"

ASOS_HEADER_ONLY = b"station,valid,tmpf,metar\n"
CLI_EMPTY = json.dumps({"results": []}).encode("utf-8")

# -- independent derivation of the known answers ------------------------------
#
# This decodes the ASOS fixture's METAR 6-hour groups with a fresh regex, not
# via weather_forecast_audit.iem.metar.parse_six_hour_groups, so this table
# is not just an echo of the production parser it verifies.

_MAX_GROUP = re.compile(r"^1[01]\d{3}$")
_MIN_GROUP = re.compile(r"^2[01]\d{3}$")
_LOOKBACK = timedelta(minutes=60)

ASOS_FIXTURE = FIXTURES / "asos_kphx_2023-07-13_2023-07-17.csv"


def _group_value_f(token: str) -> float:
    sign = -1.0 if token[1] == "1" else 1.0
    tenths = int(token[2:5])
    return sign * tenths / 10.0 * 9 / 5 + 32


def _decode_metar_groups(metar: str) -> tuple[float | None, float | None]:
    tokens = metar.split()
    if "RMK" not in tokens:
        return None, None
    remarks = tokens[tokens.index("RMK") + 1 :]
    max_matches = [t for t in remarks if _MAX_GROUP.match(t)]
    min_matches = [t for t in remarks if _MIN_GROUP.match(t)]
    max_f = _group_value_f(max_matches[0]) if max_matches else None
    min_f = _group_value_f(min_matches[0]) if min_matches else None
    return max_f, min_f


def _load_fixture_reports() -> list[tuple[datetime, float | None, float | None]]:
    reports: list[tuple[datetime, float | None, float | None]] = []
    with ASOS_FIXTURE.open(newline="") as handle:
        for record in csv.DictReader(handle):
            valid = datetime.strptime(record["valid"], "%Y-%m-%d %H:%M").replace(
                tzinfo=UTC
            )
            max_f, min_f = _decode_metar_groups(record["metar"])
            reports.append((valid, max_f, min_f))
    return reports


def _six_hour_periods(window_start: datetime) -> tuple[datetime, datetime, datetime]:
    return (
        window_start + timedelta(hours=6),
        window_start + timedelta(hours=12),
        window_start + timedelta(hours=18),
    )


def _derive_six_hour_extreme(
    reports: list[tuple[datetime, float | None, float | None]],
    window_start: datetime,
    variable: str,
) -> tuple[float | None, int]:
    """The metar_6h extreme over `window_start`'s three synoptic periods.

    Independent re-implementation of the matching rule (latest report in
    `[H-60min, H)` wins) for the known-answer table below, not a call into
    `weather_forecast_audit.resolver`.
    """
    found: list[float] = []
    for hour in _six_hour_periods(window_start):
        lower = hour - _LOOKBACK
        latest_valid: datetime | None = None
        latest_value: float | None = None
        for valid, max_f, min_f in reports:
            if not (lower <= valid < hour):
                continue
            candidate = max_f if variable == "max" else min_f
            if candidate is None:
                continue
            if latest_valid is None or valid > latest_valid:
                latest_valid = valid
                latest_value = candidate
        if latest_value is not None:
            found.append(latest_value)
    if len(found) != 3:
        return None, len(found)
    return (max(found) if variable == "max" else min(found)), len(found)


def _build_known_answers() -> list[
    tuple[date, int, str, float, float | None, float | None, int]
]:
    """Independently derive the five 13Z-2023-07-14 targets from the fixture.

    target_date, lead, variable, forecast_f (from the fixture's guidance
    rows, unchanged by issue #6), observed (metar_6h), error, periods_found.
    """
    reports = _load_fixture_reports()
    targets = [
        (date(2023, 7, 15), 1, "min", 93.0),
        (date(2023, 7, 15), 1, "max", 118.0),
        (date(2023, 7, 16), 2, "min", 93.0),
        (date(2023, 7, 16), 2, "max", 118.0),
        (date(2023, 7, 17), 3, "min", 94.0),
    ]
    answers = []
    for target_date, lead, variable, forecast in targets:
        window_start = (
            datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
            if variable == "min"
            else datetime.combine(target_date, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=12)
        )
        observed, periods_found = _derive_six_hour_extreme(
            reports, window_start, variable
        )
        error = round(forecast - observed, 6) if observed is not None else None
        answers.append(
            (target_date, lead, variable, forecast, observed, error, periods_found)
        )
    return answers


KNOWN_ANSWERS = _build_known_answers()

# Cross-check against the build spec's own hand-derived values (all but the
# 2023-07-16 max, which the spec asked us to derive independently).
_SPEC_CHECKS = {
    (date(2023, 7, 15), "min"): (91.94, 1.06),
    (date(2023, 7, 15), "max"): (118.04, -0.04),
    (date(2023, 7, 16), "min"): (93.92, -0.92),
}
for _target, _lead, _variable, _forecast, _observed, _error, _ in KNOWN_ANSWERS:
    _check = _SPEC_CHECKS.get((_target, _variable))
    if _check is not None:
        _expected_observed, _expected_error = _check
        assert _observed == pytest.approx(_expected_observed, abs=1e-6), (
            f"{_target} {_variable}: derived {_observed}, "
            f"spec says {_expected_observed}"
        )
        assert _error == pytest.approx(_expected_error, abs=1e-2), (
            f"{_target} {_variable}: derived error {_error}, "
            f"spec says {_expected_error}"
        )


def _read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _asos_in_requested_range(url: str) -> bytes:
    """The ASOS fixture clipped to the request's [sts, ets), as the service does.

    Serving the whole file regardless of range would hide a pipeline that
    fetches too narrow a span: its lead-2/3 windows would still find their
    observations here, though the live service would not return them.
    """
    query = parse_qs(urlparse(url).query)
    sts = datetime.fromisoformat(query["sts"][0].replace("Z", "+00:00"))
    ets = datetime.fromisoformat(query["ets"][0].replace("Z", "+00:00"))
    header, *rows = _read("asos_kphx_2023-07-13_2023-07-17.csv").decode().splitlines()
    kept = [
        row
        for row in rows
        if sts
        <= datetime.strptime(row.split(",")[1], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        < ets
    ]
    return ("\n".join([header, *kept]) + "\n").encode()


class FixtureFetcher:
    """Routes NBS/ASOS/CLI requests to the recorded fixtures by URL shape.

    ASOS/CLI serve the 2023 fixture for 2023 requests and an empty payload
    for 2026 requests, matching the build spec's integration-test recipe.
    """

    def get(self, url: str) -> HttpResponse:
        if "mos.py" in url:
            if "sts=2023-07" in url:
                return HttpResponse(200, _read("nbs_kphx_2023-07-14.csv"))
            if "sts=2026-04" in url:
                return HttpResponse(200, _read("nbs_kphx_2026-04-29.csv"))
            if "sts=2026-05" in url:
                return HttpResponse(200, _read("nbs_kphx_2026-05-06.csv"))
            raise AssertionError(f"unexpected mos.py request: {url}")
        if "asos.py" in url:
            if "sts=2023-07" in url:
                return HttpResponse(200, _asos_in_requested_range(url))
            if "sts=2026" in url:
                return HttpResponse(200, ASOS_HEADER_ONLY)
            raise AssertionError(f"unexpected asos.py request: {url}")
        if "cli.py" in url:
            if "year=2023" in url:
                return HttpResponse(200, _read("cli_kphx_2023.json"))
            if "year=2026" in url:
                return HttpResponse(200, CLI_EMPTY)
            raise AssertionError(f"unexpected cli.py request: {url}")
        raise AssertionError(f"unexpected request: {url}")


def _run_ingest(db_path: Path) -> None:
    station = load_registry()["KPHX"]
    regimes = load_cycle_regimes()
    fetcher = FixtureFetcher()
    with duckdb.connect(str(db_path)) as conn:
        warehouse.init_db(conn)
        for start, end in [
            (date(2023, 7, 14), date(2023, 7, 14)),
            (date(2026, 4, 29), date(2026, 4, 29)),
            (date(2026, 5, 6), date(2026, 5, 6)),
        ]:
            ingest_station(conn, station, start, end, fetcher, regimes)


def _dbt_build(tmp_path: Path, db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("DBT_TARGET_PATH", str(tmp_path / "target"))
    monkeypatch.setenv("DBT_LOG_PATH", str(tmp_path / "logs"))
    for name in [n for n in os.environ if n.startswith("WFA_SNOWFLAKE_")]:
        monkeypatch.delenv(name)
    result = dbtRunner().invoke(
        ["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    )
    assert result.success, result.exception


def test_tracer_pipeline_known_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        rows = conn.execute(
            "select target_date, lead_day, variable, forecast_f, observed_f, "
            "error_f, periods_found, scorable, extreme_source "
            "from fct_forecast_verification "
            "where station = 'KPHX' and run_date = '2023-07-14'"
        ).fetchall()

    by_key = {(r[0], r[1], r[2]): r for r in rows}
    assert len(by_key) == len(KNOWN_ANSWERS)
    for target, lead, variable, forecast, observed, error, periods in KNOWN_ANSWERS:
        row = by_key[(target, lead, variable)]
        assert row[3] == pytest.approx(forecast, abs=1e-6)
        if observed is None:
            assert row[4] is None
            assert row[5] is None
            assert row[7] is False
            assert row[8] == "none"
        else:
            assert row[4] == pytest.approx(observed, abs=1e-6)
            assert row[5] == pytest.approx(error, abs=1e-6)
            assert row[7] is True
            assert row[8] == "metar_6h"
        assert row[6] == periods


def test_tracer_pipeline_2026_rows_carry_cycle_hour_and_are_unscorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        rows_0429 = conn.execute(
            "select cycle_hour, scorable, error_f, extreme_source "
            "from fct_forecast_verification where run_date = '2026-04-29'"
        ).fetchall()
        rows_0506 = conn.execute(
            "select cycle_hour, scorable, error_f, extreme_source "
            "from fct_forecast_verification where run_date = '2026-05-06'"
        ).fetchall()

    assert rows_0429
    assert all(row[0] == 13 for row in rows_0429)
    assert all(
        row[1] is False and row[2] is None and row[3] == "none" for row in rows_0429
    )

    assert rows_0506
    assert all(row[0] == 12 for row in rows_0506)
    assert all(
        row[1] is False and row[2] is None and row[3] == "none" for row in rows_0506
    )


def test_tracer_pipeline_gaps_has_missing_observations_for_2026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracer.duckdb"
    _run_ingest(db_path)
    _dbt_build(tmp_path, db_path, monkeypatch)

    with duckdb.connect(str(db_path)) as conn:
        gaps = conn.execute(
            "select expected, reason from stg_ingest_gaps "
            "where source = 'asos' and reason = 'missing_observations' "
            "and expected like '2026%'"
        ).fetchall()

    assert gaps


def test_ingest_twice_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "tracer.duckdb"
    tables = [
        "nbs_guidance",
        "asos_hourly",
        "cli_daily",
        "ingest_gaps",
        "resolved_windows",
    ]

    _run_ingest(db_path)
    with duckdb.connect(str(db_path)) as conn:
        counts_first = {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()
            for table in tables
        }

    _run_ingest(db_path)
    with duckdb.connect(str(db_path)) as conn:
        counts_second = {
            table: conn.execute(f"select count(*) from raw.{table}").fetchone()
            for table in tables
        }

    assert counts_first == counts_second
    assert all(count[0] > 0 for count in counts_first.values())
