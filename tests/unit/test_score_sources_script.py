"""`scripts/score_sources.py --reference NAME` (issue #19).

The script is not package code, so it is loaded by path. Its default output
must not change when the flag is absent; with the flag, skill is computed
against the named source and the reference is printed.
"""

import importlib.util
from datetime import date, timedelta
from pathlib import Path
from types import ModuleType

import duckdb
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.io]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_sources.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("score_sources_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_db(path: Path) -> None:
    """raw_nbm, baseline and challenger over 40 run dates, one station."""
    errors = {"raw_nbm": 2.0, "baseline": 1.0, "challenger": 0.5}
    with duckdb.connect(str(path)) as conn:
        conn.execute(
            """
            create table fct_forecast_verification (
                station varchar, run_date date, lead_day integer,
                variable varchar, source varchar, target_date date,
                scorable boolean, error_f double
            )
            """
        )
        for offset in range(40):
            run_date_ = date(2025, 1, 1) + timedelta(days=offset)
            for source, magnitude in errors.items():
                sign = 1.0 if offset % 2 == 0 else -1.0
                conn.execute(
                    "insert into fct_forecast_verification values "
                    "('KPHX', ?, 1, 'max', ?, ?, true, ?)",
                    [
                        run_date_,
                        source,
                        run_date_ + timedelta(days=1),
                        sign * magnitude,
                    ],
                )


def _skill_by_source(output: str) -> dict[str, str]:
    """`source -> skill cell` from the first (per-lead) table."""
    skills: dict[str, str] = {}
    for line in output.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and cells[2] in {"raw_nbm", "baseline", "challenger"}:
            skills.setdefault(cells[2], cells[6])
    return skills


def _pooled_skill_by_source(output: str) -> dict[str, str]:
    """`source -> skill cell` from the pooled-over-leads table (6 columns)."""
    _, _, pooled = output.partition("## Pooled over leads")
    skills: dict[str, str] = {}
    for line in pooled.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 6 and cells[1] in {"raw_nbm", "baseline", "challenger"}:
            skills[cells[1]] = cells[5]
    return skills


def test_flag_absent_output_is_the_raw_nbm_reference_and_unannotated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "fct.duckdb"
    _build_db(db_path)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    _load_script().main([])
    default_output = capsys.readouterr().out
    _load_script().main(["--reference", "raw_nbm"])
    explicit_output = capsys.readouterr().out

    assert "eference" not in default_output
    assert default_output.startswith("## Per (variable, lead_day)")
    skills = _skill_by_source(default_output)
    assert skills["raw_nbm"].startswith("0.000")
    # challenger |e| 0.5 vs raw 2.0 -> 1 - 0.5/2.0
    assert skills["challenger"].startswith("0.750")
    # The explicit default scores identically (the reference is only labelled).
    assert _skill_by_source(explicit_output) == skills


def test_reference_flag_scores_against_the_named_source_and_prints_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "fct.duckdb"
    _build_db(db_path)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    _load_script().main(["--reference", "baseline"])
    output = capsys.readouterr().out

    assert output.startswith("Reference source: `baseline`\n")
    skills = _skill_by_source(output)
    assert skills["baseline"].startswith("0.000")
    # challenger |e| 0.5 vs baseline 1.0 -> 0.5; raw_nbm 2.0 vs 1.0 -> -1.0
    assert skills["challenger"].startswith("0.500")
    assert skills["raw_nbm"].startswith("-1.000")


def test_unknown_reference_exits_with_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "fct.duckdb"
    _build_db(db_path)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    with pytest.raises(SystemExit, match="no_such_source"):
        _load_script().main(["--reference", "no_such_source"])


def test_reference_flag_applies_to_the_pooled_table_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "fct.duckdb"
    _build_db(db_path)
    monkeypatch.setenv("WFA_DUCKDB_PATH", str(db_path))

    _load_script().main([])
    default_pooled = _pooled_skill_by_source(capsys.readouterr().out)
    _load_script().main(["--reference", "baseline"])
    baseline_pooled = _pooled_skill_by_source(capsys.readouterr().out)

    assert default_pooled["raw_nbm"].startswith("0.000")
    assert default_pooled["challenger"].startswith("0.750")
    assert baseline_pooled["baseline"].startswith("0.000")
    assert baseline_pooled["challenger"].startswith("0.500")
    assert baseline_pooled["raw_nbm"].startswith("-1.000")
