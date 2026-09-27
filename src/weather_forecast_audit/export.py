"""Export contract v1: manifest, city index, per-city stats, completeness stub.

Turns `fct_forecast_verification` (source `raw_nbm` only in v1) and the
station registry into the versioned static JSON files the site reads.
`SCHEMA_VERSION` is embedded in every file. The four schemas under
`export_schema/v1/` and their content-hash lock (`export_schema/lock.json`)
are the versioning guard: editing a schema without bumping
`SCHEMA_VERSION` and appending a new lock entry fails
`tests/unit/test_export.py::test_schema_matches_lock_for_current_version`.

No Dagster asset here by design -- another builder is editing
`definitions.py` in parallel (issue #10); the export asset lands with #16.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import polars as pl

from weather_forecast_audit.registry import Station, load_registry
from weather_forecast_audit.scoring import score

SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "export_schema" / "v1"
LOCK_PATH = REPO_ROOT / "export_schema" / "lock.json"

_SCHEMA_FILENAMES = (
    "manifest.schema.json",
    "city_index.schema.json",
    "city.schema.json",
    "completeness.schema.json",
)

_RAW_SOURCE = "raw_nbm"
_SEASON_WORDS = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "fall"}
_SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
_VARIABLE_WORDS = {"max": "Highs", "min": "Lows"}

_VERIFICATION_QUERY = """
    select station, run_date, lead_day, variable, source, target_date,
           scorable, error_f
    from fct_forecast_verification
    where source = 'raw_nbm'
"""


# -- schema / lock versioning guard -----------------------------------------


def schema_files() -> dict[str, Path]:
    """The four v1 schema files, keyed by filename (as `lock.json` keys them)."""
    return {name: SCHEMA_DIR / name for name in _SCHEMA_FILENAMES}


def canonical_hash(obj: object) -> str:
    """SHA-256 hex digest of `obj`'s canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_lock() -> dict[str, dict[str, str]]:
    """`export_schema/lock.json`: schema_version -> {schema filename -> hash}."""
    return json.loads(LOCK_PATH.read_text())


# -- the plain-language summary ---------------------------------------------


def build_summary(stats: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """A city's plain-language bias summary from its (unrounded) stat rows.

    See the build spec's "The plain-language summary" section. A candidate
    is a lead-1 slice with a real, sample-backed bias (`no_detectable_bias`
    and `min_sample_flag` both false); the candidate with the largest
    `|bias_f|` wins, ties broken by variable (max before min) then season
    order (DJF, MAM, JJA, SON). With no candidate, the text falls back to
    "No detectable bias..." if some lead-1 slice at least had enough sample,
    else "Not enough data yet." A significant bias that rounds to 0.0 at one
    decimal still names its direction ("slightly warm"/"slightly cold")
    rather than printing a misleading "+0.0".
    """
    lead1 = [s for s in stats if s["lead_day"] == 1]
    candidates = [
        s for s in lead1 if not s["no_detectable_bias"] and not s["min_sample_flag"]
    ]

    if not candidates:
        any_enough_sample = any(not s["min_sample_flag"] for s in lead1)
        text = (
            "No detectable bias in day-ahead forecasts."
            if any_enough_sample
            else "Not enough data yet."
        )
        return {"text": text, "significant": False}

    def _sort_key(stat: Mapping[str, object]) -> tuple[float, int, int]:
        bias = float(stat["bias_f"])  # type: ignore[arg-type]
        return (
            -abs(bias),
            0 if stat["variable"] == "max" else 1,
            _SEASON_ORDER[str(stat["season"])],
        )

    chosen = min(candidates, key=_sort_key)
    bias = float(chosen["bias_f"])  # type: ignore[arg-type]
    direction = "warm" if bias > 0 else "cold"
    variable_word = _VARIABLE_WORDS[str(chosen["variable"])]
    season_word = _SEASON_WORDS[str(chosen["season"])]
    rounded_abs = round(abs(bias), 1)

    if rounded_abs == 0.0:
        text = (
            f"{variable_word} run slightly {direction} in {season_word} "
            "(day-ahead forecasts)."
        )
    else:
        text = (
            f"{variable_word} run {rounded_abs:.1f}°F {direction} in "
            f"{season_word} (day-ahead forecasts)."
        )

    return {
        "text": text,
        "significant": True,
        "variable": chosen["variable"],
        "lead_day": chosen["lead_day"],
        "season": chosen["season"],
        "bias_f": round(bias, 2),
    }


# -- per-stat rounding at export ---------------------------------------------


def _round2(value: object) -> float | None:
    return None if value is None else round(float(value), 2)  # type: ignore[arg-type]


def round_stat(stat: Mapping[str, object]) -> dict[str, object]:
    """Round a stat's float fields to 2 decimals for JSON output.

    Flags (`min_sample_flag`, `no_detectable_bias`) are computed by
    `scoring.score` from unrounded values and passed through unchanged here:
    rounding must never be able to flip a flag.
    """
    rounded = dict(stat)
    rounded["bias_f"] = _round2(stat["bias_f"])
    rounded["bias_lo_f"] = _round2(stat["bias_lo_f"])
    rounded["bias_hi_f"] = _round2(stat["bias_hi_f"])
    rounded["mae_f"] = _round2(stat["mae_f"])
    return rounded


# -- DB-backed export ---------------------------------------------------------


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    files: list[str]
    station_count: int
    data_through: date | None


def _load_verification_rows(conn: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    return conn.execute(_VERIFICATION_QUERY).pl()


def _data_through(conn: duckdb.DuckDBPyConnection) -> date | None:
    row = conn.execute(
        "select max(target_date) from fct_forecast_verification "
        "where source = 'raw_nbm' and scorable"
    ).fetchone()
    return row[0] if row else None


def _station_stats(scored: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    by_station: dict[str, list[dict[str, object]]] = {}
    for row in scored.iter_rows(named=True):
        by_station.setdefault(str(row["station"]), []).append(
            {
                "variable": row["variable"],
                "lead_day": row["lead_day"],
                "season": row["season"],
                "n": row["n"],
                "n_dates": row["n_dates"],
                "bias_f": row["bias"],
                "bias_lo_f": row["bias_lo"],
                "bias_hi_f": row["bias_hi"],
                "mae_f": row["mae"],
                "min_sample_flag": row["min_sample_flag"],
                "no_detectable_bias": row["no_detectable_bias"],
            }
        )
    return by_station


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def export(
    conn: duckdb.DuckDBPyConnection,
    out_dir: Path,
    *,
    registry: dict[str, Station] | None = None,
) -> ExportResult:
    """Write the v1 export contract under `out_dir`.

    Reads `fct_forecast_verification` (source `raw_nbm` only) and scores it
    per `station x variable x lead_day x season`
    (`scoring.score(rows, by=(...))`), then writes `manifest.json`,
    `city_index.json`, `cities/{ICAO}.json` for every registry station, and
    a `completeness.json` stub. `registry` defaults to the checked-in seed
    (`weather_forecast_audit.registry.load_registry`); tests may pass a
    smaller one.
    """
    if registry is None:
        registry = load_registry()

    rows = _load_verification_rows(conn)
    scored = score(rows, by=("station", "variable", "lead_day", "season"))
    by_station = _station_stats(scored)

    out_dir.mkdir(parents=True, exist_ok=True)
    cities_dir = out_dir / "cities"
    cities_dir.mkdir(exist_ok=True)

    city_index_entries: list[dict[str, object]] = []
    for icao in sorted(registry):
        station = registry[icao]
        raw_stats = by_station.get(icao, [])
        summary = build_summary(raw_stats)
        rounded_stats = [round_stat(stat) for stat in raw_stats]

        _write_json(
            cities_dir / f"{icao}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "icao": station.icao,
                "label": station.label,
                "stats": rounded_stats,
            },
        )

        city_index_entries.append(
            {
                "icao": station.icao,
                "label": station.label,
                "lat": station.lat,
                "lon": station.lon,
                "climate_region": station.climate_region,
                "summary": summary,
            }
        )

    _write_json(
        out_dir / "city_index.json",
        {"schema_version": SCHEMA_VERSION, "cities": city_index_entries},
    )
    _write_json(
        out_dir / "completeness.json",
        {"schema_version": SCHEMA_VERSION, "stations": []},
    )

    data_through = _data_through(conn)
    files = [
        "manifest.json",
        "city_index.json",
        "completeness.json",
        *(f"cities/{icao}.json" for icao in sorted(registry)),
    ]
    _write_json(
        out_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "data_through": data_through.isoformat() if data_through else None,
            "station_count": len(registry),
            "files": files,
        },
    )

    return ExportResult(
        out_dir=out_dir,
        files=files,
        station_count=len(registry),
        data_through=data_through,
    )
