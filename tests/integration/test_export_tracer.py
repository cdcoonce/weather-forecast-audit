"""DB-backed export tests: build the tracer fixture warehouse, run the v1
export, and validate every file against its committed JSON Schema.

Reuses `FixtureFetcher` (`tests/support/fixture_fetcher.py`) and the same
ingest-then-dbt-build recipe as `tests/integration/test_tracer_pipeline.py`,
so this is not a second, drifting definition of "the tracer fixture."

The registry passed to `export.export` is a one-station subset (KPHX only),
not the full 573-station seed: the fixture DB only ever ingests KPHX, and a
573-file fixture export would be committed to `site/src/test/fixtures/export/`
for no test benefit. Production `wfa export` (no `registry=` override)
defaults to the full seed.
"""

import copy
import json
import os
from pathlib import Path

import duckdb
import pytest
from dbt.cli.main import dbtRunner
from jsonschema import Draft202012Validator, ValidationError
from support.fixture_fetcher import FixtureFetcher

from weather_forecast_audit import export, warehouse
from weather_forecast_audit.pipeline import ingest_station
from weather_forecast_audit.regimes import load_cycle_regimes
from weather_forecast_audit.registry import load_registry

pytestmark = [pytest.mark.integration, pytest.mark.io, pytest.mark.dbt]

REPO = Path(__file__).resolve().parents[2]
DBT_DIR = REPO / "dbt"
SITE_FIXTURE_DIR = REPO / "site" / "src" / "test" / "fixtures" / "export"


def _kphx_registry() -> dict[str, object]:
    full = load_registry()
    return {"KPHX": full["KPHX"]}


def _build_tracer_db(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import date

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

    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))
    for name in [n for n in os.environ if n.startswith("WFA_SNOWFLAKE_")]:
        monkeypatch.delenv(name)
    result = dbtRunner().invoke(
        ["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    )
    assert result.success, result.exception


def _run_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "tracer.duckdb"
    _build_tracer_db(db_path, monkeypatch)
    out_dir = tmp_path / "export"
    with duckdb.connect(str(db_path)) as conn:
        export.export(conn, out_dir, registry=_kphx_registry())
    return out_dir


def _validator_for(schema_path: Path) -> Draft202012Validator:
    schema = json.loads(schema_path.read_text())
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


def test_export_validates_against_schemas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _run_export(tmp_path, monkeypatch)
    schemas = export.schema_files()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    _validator_for(schemas["manifest.schema.json"]).validate(manifest)

    city_index = json.loads((out_dir / "city_index.json").read_text())
    _validator_for(schemas["city_index.schema.json"]).validate(city_index)

    completeness = json.loads((out_dir / "completeness.json").read_text())
    _validator_for(schemas["completeness.schema.json"]).validate(completeness)

    city_validator = _validator_for(schemas["city.schema.json"])
    for relative_path in manifest["files"]:
        if relative_path.startswith("cities/"):
            city = json.loads((out_dir / relative_path).read_text())
            city_validator.validate(city)

    assert (out_dir / "cities" / "KPHX.json").exists()


def test_dropping_a_required_field_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = _run_export(tmp_path, monkeypatch)
    schemas = export.schema_files()

    manifest = json.loads((out_dir / "manifest.json").read_text())
    manifest_validator = _validator_for(schemas["manifest.schema.json"])
    manifest_validator.validate(manifest)  # sanity: the real export is valid

    broken_manifest = copy.deepcopy(manifest)
    del broken_manifest["data_through"]
    with pytest.raises(ValidationError):
        manifest_validator.validate(broken_manifest)

    kphx = json.loads((out_dir / "cities" / "KPHX.json").read_text())
    city_validator = _validator_for(schemas["city.schema.json"])
    city_validator.validate(kphx)  # sanity: the real export is valid
    assert kphx["stats"], "KPHX must have at least one stat row for this test"

    broken_city = copy.deepcopy(kphx)
    del broken_city["stats"][0]["bias_f"]
    with pytest.raises(ValidationError):
        city_validator.validate(broken_city)


@pytest.mark.parametrize(
    "bad",
    ["yesterday", "2026-09-27", "2026-09-27 15:41:55", "2026-09-27T15:41:55+02:00"],
)
def test_malformed_generated_at_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    # `format: date-time` is a no-op under plain jsonschema (it needs the
    # optional rfc3339-validator), so the contract pins the UTC shape with a
    # pattern instead; this proves the guard is not decorative.
    out_dir = _run_export(tmp_path, monkeypatch)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    validator = _validator_for(export.schema_files()["manifest.schema.json"])
    manifest["generated_at"] = bad
    with pytest.raises(ValidationError):
        validator.validate(manifest)


def test_exported_stats_never_contradict_their_own_no_detectable_bias_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For every exported stat with non-null endpoints, re-deriving
    `no_detectable_bias` from the *exported* (rounded) numbers must agree
    with the exported flag, and the exported point estimate must lie within
    the exported endpoints -- the sign-preserving rounding invariant, run
    against a real DB-backed export of the tracer fixture rather than only
    hand-built stats (see tests/unit/test_export.py for those exercising
    the actual near-zero cases -- the tracer fixture's few distinct
    issuance dates mean its own bootstrap CIs are usually null, so this
    pass is a drift guard more than a source of new near-zero cases)."""
    out_dir = _run_export(tmp_path, monkeypatch)
    kphx = json.loads((out_dir / "cities" / "KPHX.json").read_text())
    assert kphx["stats"], "KPHX must have at least one stat row for this test"

    for stat in kphx["stats"]:
        if stat["bias_lo_f"] is None or stat["bias_hi_f"] is None:
            continue
        derived = stat["bias_lo_f"] <= 0 <= stat["bias_hi_f"]
        assert stat["no_detectable_bias"] == derived, stat
        if stat["bias_f"] is not None:
            assert stat["bias_lo_f"] <= stat["bias_f"] <= stat["bias_hi_f"], stat


def test_site_fixture_export_matches_a_fresh_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed `site/src/test/fixtures/export/` must equal a fresh
    export of the same tracer fixture DB, ignoring `generated_at`. This
    keeps the site's test data from drifting away from what the exporter
    actually produces."""
    out_dir = _run_export(tmp_path, monkeypatch)

    committed_manifest = json.loads((SITE_FIXTURE_DIR / "manifest.json").read_text())
    fresh_manifest = json.loads((out_dir / "manifest.json").read_text())
    committed_manifest.pop("generated_at", None)
    fresh_manifest.pop("generated_at", None)
    assert committed_manifest == fresh_manifest

    committed_city_index = json.loads(
        (SITE_FIXTURE_DIR / "city_index.json").read_text()
    )
    fresh_city_index = json.loads((out_dir / "city_index.json").read_text())
    assert committed_city_index == fresh_city_index

    committed_completeness = json.loads(
        (SITE_FIXTURE_DIR / "completeness.json").read_text()
    )
    fresh_completeness = json.loads((out_dir / "completeness.json").read_text())
    assert committed_completeness == fresh_completeness

    for relative_path in fresh_manifest["files"]:
        if not relative_path.startswith("cities/"):
            continue
        committed_city = json.loads((SITE_FIXTURE_DIR / relative_path).read_text())
        fresh_city = json.loads((out_dir / relative_path).read_text())
        assert committed_city == fresh_city, relative_path


def _exported_json(out_dir: Path) -> dict[str, object]:
    """Every exported JSON file keyed by relative path, `generated_at` dropped."""
    exported: dict[str, object] = {}
    for path in sorted(out_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload.pop("generated_at", None)
        exported[str(path.relative_to(out_dir))] = payload
    return exported


def test_correction_model_rows_never_reach_the_published_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The export publishes raw NBM only (v1). Once `wfa predict baseline`
    (#18) writes `baseline` rows into `fct_forecast_verification`, they must
    not move a single published number, including `data_through`: add
    wildly wrong baseline rows, one of them later than any raw row, and the
    export must be byte-for-byte the same apart from `generated_at`."""
    db_path = tmp_path / "tracer.duckdb"
    _build_tracer_db(db_path, monkeypatch)
    registry = _kphx_registry()

    before_dir = tmp_path / "before"
    with duckdb.connect(str(db_path)) as conn:
        export.export(conn, before_dir, registry=registry)
        raw_rows = conn.execute(
            "select count(*) from fct_forecast_verification where source = 'raw_nbm'"
        ).fetchone()
        assert raw_rows is not None and raw_rows[0] > 0
        conn.execute(
            """
            insert into fct_forecast_verification by name
            select * replace (
                'baseline' as source,
                error_f + 40 as error_f,
                target_date + interval 400 day as target_date
            )
            from fct_forecast_verification
            where source = 'raw_nbm'
            """
        )

    after_dir = tmp_path / "after"
    with duckdb.connect(str(db_path)) as conn:
        export.export(conn, after_dir, registry=registry)

    assert _exported_json(after_dir) == _exported_json(before_dir)
