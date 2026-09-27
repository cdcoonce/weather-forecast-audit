"""Repo-root resolution that survives an installed (non-editable) wheel.

Found while proving D4's Docker requirement: `regimes.py`/`registry.py`/
`dbt_assets.py` each computed their repo root as
`Path(__file__).resolve().parents[2]`, which is only two levels above
`src/weather_forecast_audit/*.py` in an editable checkout. Installed
non-editably (as the Docker image's runtime `.venv` is, via
`uv sync --no-editable`), `__file__` resolves under `site-packages`, so
that traversal lands nowhere near `dbt/seeds/` -- `Definitions` failed to
import in the built image with a `FileNotFoundError` from
`regimes.load_archive_start()` (assets.py now calls it at module import
time, per D1: "the archive start comes from
regimes.load_archive_start()... do not hard-code it").
"""

from pathlib import Path

import pytest

from weather_forecast_audit._paths import find_repo_root

pytestmark = [pytest.mark.unit, pytest.mark.io]


def test_find_repo_root_prefers_the_file_relative_path_when_it_has_dbt_seeds() -> None:
    # regimes.py's own __file__, a real editable-checkout module: two levels
    # up from src/weather_forecast_audit/ is this repo's actual root.
    repo_root = Path(__file__).resolve().parents[2]
    real_module_file = str(repo_root / "src" / "weather_forecast_audit" / "regimes.py")

    root = find_repo_root(real_module_file)

    assert (root / "dbt" / "seeds").is_dir()
    assert root == Path(__file__).resolve().parents[2]


def test_find_repo_root_falls_back_to_cwd_when_file_relative_path_lacks_dbt_seeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Simulate an installed wheel: a module file three levels under some
    # unrelated site-packages tree, with no dbt/ alongside it at all.
    fake_site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    fake_module_file = fake_site_packages / "weather_forecast_audit" / "regimes.py"
    fake_module_file.parent.mkdir(parents=True)
    fake_module_file.write_text("# stand-in for the installed module\n")

    fake_cwd = tmp_path / "container_workdir"
    (fake_cwd / "dbt" / "seeds").mkdir(parents=True)
    monkeypatch.chdir(fake_cwd)

    root = find_repo_root(str(fake_module_file))

    assert root == fake_cwd
