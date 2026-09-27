"""NBM version seed: contiguous windows, and header-verified boundaries."""

import csv
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SEED = REPO / "dbt" / "seeds" / "nbm_versions.csv"
ARCHIVE_SEED = REPO / "dbt" / "seeds" / "nbs_archive.csv"


def _rows() -> list[dict[str, str]]:
    with SEED.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)  # naive UTC, as dbt loads it


def test_version_windows_are_contiguous_and_only_the_last_is_open() -> None:
    # fct_forecast_verification's half-open join tags every row exactly once
    # only if each window ends where the next begins.
    rows = sorted(_rows(), key=lambda row: _ts(row["valid_from_utc"]))
    for earlier, later in zip(rows, rows[1:], strict=False):
        assert earlier["valid_to_utc"] == later["valid_from_utc"], earlier
    assert [row["valid_to_utc"] for row in rows].count("") == 1
    assert rows[-1]["valid_to_utc"] == ""


def test_first_version_starts_at_the_pinned_archive_start() -> None:
    with ARCHIVE_SEED.open(newline="") as handle:
        archive_start = next(csv.DictReader(handle))
    first = min(_rows(), key=lambda row: _ts(row["valid_from_utc"]))
    first_day = _ts(first["valid_from_utc"]).date().isoformat()
    assert first_day == archive_start["archive_start_date"]


def test_v5_boundary_is_the_header_verified_run_not_the_announced_one() -> None:
    # SCN 26-24 announced 2026-04-30 13Z; NBS bulletin headers stay V4.3
    # until a stable V5.0 at 2026-05-05 12Z
    # (docs/analysis/2026-09-27-nbm-version-headers/RESULTS.md).
    by_version = {row["nbm_version"]: row for row in _rows()}
    assert by_version["v5.0"]["valid_from_utc"] == "2026-05-05 12:00:00"
    assert by_version["v4.2"]["valid_from_utc"] == "2024-05-15 11:00:00"
    assert all(row["verified"] == "true" for row in by_version.values())
