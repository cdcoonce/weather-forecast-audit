"""sqlfluff in the Snowflake dialect is the dbt models' portability guard."""

from pathlib import Path

import pytest
from sqlfluff.core import FluffConfig, Linter

pytestmark = pytest.mark.dbt

REPO = Path(__file__).resolve().parents[2]
DBT_DIR = REPO / "dbt"
SCRATCH_MODEL = DBT_DIR / "models" / "scratch_portability_probe.sql"


def _linter() -> Linter:
    return Linter(config=FluffConfig.from_path(str(DBT_DIR)))


def _rule_codes(sql: str) -> set[str]:
    linted = _linter().lint_string(sql, fname=str(SCRATCH_MODEL))
    return {violation.rule_code() for violation in linted.get_violations()}


def test_config_pins_snowflake_dialect() -> None:
    assert FluffConfig.from_path(str(DBT_DIR)).get("dialect") == "snowflake"


def test_dbt_models_pass_lint() -> None:
    result = _linter().lint_paths((str(DBT_DIR / "models"),))

    assert result.as_records(), "no model files were linted"
    assert result.num_violations() == 0, result.as_records()


def test_duckdb_list_comprehension_fails_to_parse() -> None:
    sql = "select [x * 2 for x in xs] as doubled from {{ ref('scaffold_heartbeat') }}\n"

    assert "PRS" in _rule_codes(sql)


def test_portable_equivalent_parses() -> None:
    # Positive control: the same statement minus the DuckDB-only construct parses,
    # so the PRS above comes from the list comprehension, not from templating.
    sql = "select xs as doubled from {{ ref('scaffold_heartbeat') }}\n"

    assert "PRS" not in _rule_codes(sql)
