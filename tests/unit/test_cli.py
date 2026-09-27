"""wfa CLI: env var handling and command wiring (no network)."""

from pathlib import Path

import duckdb
import pytest

from weather_forecast_audit import cli

pytestmark = pytest.mark.unit


def test_main_requires_wfa_duckdb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DUCKDB_PATH", raising=False)
    with pytest.raises(SystemExit, match="WFA_DUCKDB_PATH"):
        cli.main(["init-db"])


def test_init_db_creates_raw_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "wfa.duckdb"
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    cli.main(["init-db"])

    with duckdb.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "select table_name from information_schema.tables "
                "where table_schema = 'raw'"
            ).fetchall()
        }
    assert "nbs_guidance" in tables


def test_export_requires_wfa_duckdb_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WFA_DUCKDB_PATH", raising=False)
    with pytest.raises(SystemExit, match="WFA_DUCKDB_PATH"):
        cli.main(["export", "--out", "/tmp/wfa-export-does-not-run"])


def test_export_parser_requires_out_flag() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["export"])


def test_ingest_unknown_station_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(tmp_path / "wfa.duckdb"))

    with pytest.raises(SystemExit, match="unknown station"):
        cli.main(
            [
                "ingest",
                "--station",
                "ZZZZ",
                "--start",
                "2023-07-14",
                "--end",
                "2023-07-14",
            ]
        )
