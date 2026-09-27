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
import re
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


# -- typical miss: lead-1 MAE pooled over seasons, per variable -------------


def build_typical_miss(
    stats: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    """Lead-1 MAE pooled over seasons, per variable, from unrounded stats.

    Pooled MAE = sum(n * mae) / sum(n) over that variable's lead-1 season
    slices -- the row-count-weighted mean, which is mathematically the same
    as the mean of `|error|` over every row in those slices (never the
    unweighted mean of the season MAEs, which is wrong whenever season `n`
    differs). All lead-1 season slices for the variable are pooled once
    pooling is unlocked; a variable's value is null only when *no* lead-1
    slice for it has `min_sample_flag` false (i.e. there is no sample-backed
    season to unlock pooling at all).
    """
    result: dict[str, float | None] = {}
    for variable in ("max", "min"):
        season_slices = [
            s for s in stats if s["variable"] == variable and s["lead_day"] == 1
        ]
        if not any(not s["min_sample_flag"] for s in season_slices):
            result[variable] = None
            continue
        total_n = sum(int(s["n"]) for s in season_slices)  # type: ignore[arg-type]
        if total_n == 0:
            result[variable] = None
            continue
        weighted_sum = sum(
            float(s["n"]) * float(s["mae_f"])  # type: ignore[arg-type]
            for s in season_slices
        )
        result[variable] = weighted_sum / total_n
    return result


# -- title-cased station labels -----------------------------------------------

# Known multi-cap names a naive "capitalize first letter, lowercase the rest"
# pass would mangle: Mc-/Mac- surnames (McAllen, MacArthur, ...) and the
# MCAS acronym (Marine Corps Air Station), which is not a name at all and
# reads as broken ("Mcas") if title-cased like one. Found by grepping
# dbt/seeds/station_registry.csv for labels containing "MC", "MAC", "DE ",
# "LA ", "O'" -- see the build report for the full list considered and why
# entries like MACON and DENVER needed no exception (naive casing is already
# correct for them).
_TITLE_EXCEPTIONS = {
    "MACARTHUR": "MacArthur",
    "MACREADY": "MacReady",
    "MCALLEN": "McAllen",
    "MCAS": "MCAS",
    "MCCARRAN": "McCarran",
    "MCCOMB": "McComb",
    "MCCOOK": "McCook",
    "MCGRATH": "McGrath",
    "MCKELLAR": "McKellar",
    "MCMINNVILLE": "McMinnville",
    "MCNARY": "McNary",
    "DEKALB": "DeKalb",
}

_LABEL_WORD_SPLIT_RE = re.compile(r"([ /\-'.])")
_LABEL_DELIMITERS = frozenset(" /-'.")


def _title_case_token(token: str) -> str:
    """Title-case one delimiter-split token, capitalizing the first letter
    of every maximal run of letters within it (so a parenthesized suffix
    like "(AMOS)" in "JOHNSBURY(AMOS)" -- glued to its neighbor with no
    space/slash/hyphen/apostrophe/period between them -- still reads
    "Johnsbury(Amos)" rather than lowercasing it outright)."""
    exception = _TITLE_EXCEPTIONS.get(token.upper())
    if exception is not None:
        return exception
    chars: list[str] = []
    prev_was_letter = False
    for ch in token:
        if ch.isalpha():
            chars.append(ch.upper() if not prev_was_letter else ch.lower())
            prev_was_letter = True
        else:
            chars.append(ch)
            prev_was_letter = False
    return "".join(chars)


def title_case_label(label: str) -> str:
    """Title-case a station label, keeping its trailing state code upper-case.

    Splits on spaces, slashes, hyphens, apostrophes and periods, capitalizing
    each token's first letter and lowercasing the rest, except for
    `_TITLE_EXCEPTIONS`. The two-letter state code after the label's last
    comma is left untouched (it is already upper-case in the source seed).
    """
    head, sep, tail = label.rpartition(",")
    if not sep:
        head, tail = tail, ""
    parts = _LABEL_WORD_SPLIT_RE.split(head)
    cased_head = "".join(
        part if part in _LABEL_DELIMITERS else _title_case_token(part)
        for part in parts
    )
    return f"{cased_head},{tail}" if sep else cased_head


# -- per-stat rounding at export ---------------------------------------------


def _round2(value: object) -> float | None:
    return None if value is None else round(float(value), 2)  # type: ignore[arg-type]


def _round_sign_preserving(value: float, *, exclude_zero: bool) -> float:
    """Round one bias value to 2 dp, never letting the rounding land on or
    cross 0 when the *unrounded* CI excludes 0 (`exclude_zero`, i.e.
    `no_detectable_bias` is false).

    Plain rounding can flip a barely-nonzero endpoint (or the point
    estimate) to 0.00, which reads as "spans zero" even though the
    unrounded value was, say, -0.0041 -- the exact opposite of what
    `no_detectable_bias: false` claims. When that would happen, this emits
    the smallest 2-dp value with the value's own sign instead (+0.01 or
    -0.01), nudging the published number by less than 0.01F toward the
    excluded side. The published data must never contradict its own flag.

    When `exclude_zero` is false (the CI spans 0), plain rounding can never
    break that: a value <= 0 rounds to something <= 0, and a value >= 0
    rounds to something >= 0 (nearest-2dp rounding never crosses 0 for an
    input that is already on the non-strict side of it) -- asserted by
    `test_rounding_never_breaks_the_spans_zero_invariant` rather than
    assumed here.
    """
    rounded = round(value, 2)
    if not exclude_zero:
        return rounded
    if value > 0 and rounded <= 0:
        return 0.01
    if value < 0 and rounded >= 0:
        return -0.01
    return rounded


def round_stat(stat: Mapping[str, object]) -> dict[str, object]:
    """Round a stat's float fields to 2 decimals for JSON output.

    Flags (`min_sample_flag`, `no_detectable_bias`) are computed by
    `scoring.score` from unrounded values and passed through unchanged here:
    rounding must never be able to flip a flag. Endpoints (and the point
    estimate) are rounded sign-preservingly instead of plainly, via
    `_round_sign_preserving`, precisely so that reading the flag back off
    the *rounded, exported* numbers (`bias_lo_f <= 0 <= bias_hi_f`) always
    agrees with `no_detectable_bias` -- see its docstring.

    Sign-preserving rounding of `bias_lo_f`/`bias_hi_f` alone would fix the
    endpoints but could still leave the rounded `bias_f` point estimate
    outside `[bias_lo_f, bias_hi_f]` (e.g. a `bias_f` that plain-rounds to
    0.00 while a nudged `bias_hi_f` sits at -0.01). `bias_f` is rounded the
    same sign-preserving way, and as a last resort clamped into the rounded
    interval, so the point estimate never contradicts its own CI either.
    """
    exclude_zero = not stat["no_detectable_bias"]

    bias_lo_f = stat["bias_lo_f"]
    bias_hi_f = stat["bias_hi_f"]
    bias_f = stat["bias_f"]

    rounded_lo = (
        None
        if bias_lo_f is None
        else _round_sign_preserving(float(bias_lo_f), exclude_zero=exclude_zero)  # type: ignore[arg-type]
    )
    rounded_hi = (
        None
        if bias_hi_f is None
        else _round_sign_preserving(float(bias_hi_f), exclude_zero=exclude_zero)  # type: ignore[arg-type]
    )
    rounded_bias = (
        None
        if bias_f is None
        else _round_sign_preserving(float(bias_f), exclude_zero=exclude_zero)  # type: ignore[arg-type]
    )
    if rounded_bias is not None and rounded_lo is not None:
        rounded_bias = max(rounded_bias, rounded_lo)
    if rounded_bias is not None and rounded_hi is not None:
        rounded_bias = min(rounded_bias, rounded_hi)

    rounded = dict(stat)
    rounded["bias_f"] = rounded_bias
    rounded["bias_lo_f"] = rounded_lo
    rounded["bias_hi_f"] = rounded_hi
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
        typical_miss = build_typical_miss(raw_stats)
        rounded_typical_miss = {
            variable: _round2(value) for variable, value in typical_miss.items()
        }
        label = title_case_label(station.label)

        _write_json(
            cities_dir / f"{icao}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "icao": station.icao,
                "label": label,
                "stats": rounded_stats,
                "typical_miss": rounded_typical_miss,
            },
        )

        city_index_entries.append(
            {
                "icao": station.icao,
                "label": label,
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
