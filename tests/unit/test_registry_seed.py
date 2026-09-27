"""Validate every row of the built CONUS station registry (issue #7).

No network: this only reads the checked-in seeds and evidence README that
`scripts/registry/build_registry.py` produces.
"""

import csv
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from weather_forecast_audit.registry_sources import CLIMATE_REGIONS

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
REGISTRY_SEED = REPO / "dbt" / "seeds" / "station_registry.csv"
EXCLUSIONS_SEED = REPO / "dbt" / "seeds" / "station_exclusions.csv"
EVIDENCE_README = REPO / "docs" / "spikes" / "2026-09-26-station-registry" / "README.md"

# CONUS bounding box (build spec #7 acceptance criteria).
LAT_RANGE = (24.0, 49.5)
LON_RANGE = (-125.0, -66.5)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def test_registry_rows_all_resolve_timezone_and_stay_in_conus_bbox() -> None:
    rows = _read_rows(REGISTRY_SEED)
    assert rows, "station_registry.csv must not be empty"

    for row in rows:
        ZoneInfo(row["iana_tz"])  # raises ZoneInfoNotFoundError if unresolvable

        lat = float(row["lat"])
        lon = float(row["lon"])
        assert LAT_RANGE[0] <= lat <= LAT_RANGE[1], f"{row['icao']} lat={lat} OOB"
        assert LON_RANGE[0] <= lon <= LON_RANGE[1], f"{row['icao']} lon={lon} OOB"

        assert row["climate_region"] in CLIMATE_REGIONS, row["icao"]

        coastal_raw = row["coastal_flag"].strip().lower()
        assert coastal_raw in ("true", "false"), f"{row['icao']} coastal_flag OOB"


def test_no_icao_in_both_registry_and_exclusions() -> None:
    registry_icaos = {row["icao"] for row in _read_rows(REGISTRY_SEED)}
    excluded_icaos = {row["icao"] for row in _read_rows(EXCLUSIONS_SEED)}
    assert registry_icaos & excluded_icaos == set()


def test_registry_row_count_matches_evidence_readme() -> None:
    rows = _read_rows(REGISTRY_SEED)
    content = EVIDENCE_README.read_text(encoding="utf-8")
    match = re.search(r"Included in `station_registry\.csv`: (\d+)", content)
    assert match, "evidence README must state the included row count"
    assert len(rows) == int(match.group(1))
