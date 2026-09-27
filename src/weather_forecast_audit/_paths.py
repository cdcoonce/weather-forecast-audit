"""Shared repo-root resolution, robust to an installed (non-editable) wheel.

In an editable/source checkout, the repo root sits two levels above any
module under `src/weather_forecast_audit/` (e.g. `regimes.py` ->
`src/weather_forecast_audit/regimes.py` -> repo root). Once installed as a
wheel with no source tree alongside it -- the Docker image's `--no-editable`
`.venv`, built by `uv sync --no-editable` -- `__file__` resolves under
`site-packages`, where that two-level traversal lands nowhere near `dbt/`.
`dbt/` is instead copied to the container's `WORKDIR` (`COPY dbt ./dbt` in
the Dockerfile), so the current working directory is the fallback there --
which also matches how every `wfa`/`dbt` command in this project is already
invoked, from the repo root (see README.md).
"""

from pathlib import Path


def find_repo_root(module_file: str) -> Path:
    """The repo root containing `dbt/seeds/`, found from a module's `__file__`."""
    from_file = Path(module_file).resolve().parents[2]
    if (from_file / "dbt" / "seeds").is_dir():
        return from_file
    return Path.cwd()
