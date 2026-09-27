import csv
import os
from pathlib import Path

import duckdb
import pytest
import yaml
from dbt.cli.main import dbtRunner
from dbt.flags import get_flags

from weather_forecast_audit import warehouse

pytestmark = [pytest.mark.dbt, pytest.mark.io]

REPO = Path(__file__).resolve().parents[2]
DBT_DIR = REPO / "dbt"


def test_profile_defaults_to_duckdb_and_carries_unused_snowflake_target() -> None:
    profiles = yaml.safe_load((DBT_DIR / "profiles.yml").read_text())
    profile = profiles["weather_forecast_audit"]

    assert profile["target"] == "duckdb"
    assert profile["outputs"]["duckdb"]["type"] == "duckdb"
    assert "WFA_DUCKDB_PATH" in profile["outputs"]["duckdb"]["path"]
    assert profile["outputs"]["snowflake"]["type"] == "snowflake"


def test_dbt_build_writes_to_env_configured_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "warehouse.duckdb"
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(database))
    monkeypatch.setenv("DBT_TARGET_PATH", str(tmp_path / "target"))
    monkeypatch.setenv("DBT_LOG_PATH", str(tmp_path / "logs"))
    # The unused snowflake target must not need credentials to build on DuckDB.
    for name in [n for n in os.environ if n.startswith("WFA_SNOWFLAKE_")]:
        monkeypatch.delenv(name)

    # dbt never creates the raw schema (weather_forecast_audit.warehouse does),
    # so an empty CI-like database needs it initialized first, same as
    # `wfa init-db` in the tracer script and CI gate.
    with duckdb.connect(str(database)) as connection:
        warehouse.init_db(connection)

    result = dbtRunner().invoke(
        ["build", "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    )

    assert result.success, result.exception
    built = {node.node.resource_type for node in result.result}
    assert {"model", "seed", "test"} <= built
    assert get_flags().SEND_ANONYMOUS_USAGE_STATS is False
    with (DBT_DIR / "seeds" / "station_registry.csv").open(newline="") as handle:
        expected_row_count = sum(1 for _ in csv.DictReader(handle))
    with duckdb.connect(str(database)) as connection:
        rows = connection.execute("select count(*) from station_registry").fetchone()
    assert rows == (expected_row_count,)
