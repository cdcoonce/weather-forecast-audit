"""Station registry: the seed CSV as the single source of station metadata."""

import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "station_registry.csv"


@dataclass(frozen=True)
class Station:
    cli_station: str
    icao: str
    nbm_station_id: str
    iana_tz: str
    lat: float
    lon: float
    elevation_m: int
    label: str
    climate_region: str
    coastal_flag: bool


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_registry(path: Path = DEFAULT_SEED_PATH) -> dict[str, Station]:
    """Load the station registry seed, keyed by ICAO identifier."""
    with path.open(newline="") as handle:
        stations = [
            Station(
                cli_station=row["cli_station"],
                icao=row["icao"],
                nbm_station_id=row["nbm_station_id"],
                iana_tz=row["iana_tz"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                elevation_m=int(row["elevation_m"]),
                label=row["label"],
                climate_region=row["climate_region"],
                coastal_flag=_parse_bool(row["coastal_flag"]),
            )
            for row in csv.DictReader(handle)
        ]
    return {station.icao: station for station in stations}
