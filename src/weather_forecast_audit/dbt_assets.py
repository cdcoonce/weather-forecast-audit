"""dbt models as Dagster assets (build spec #10 D4).

Unpartitioned: `transform_job` runs a full `dbt build` over all raw data on
every materialization. The default `DagsterDbtTranslator` maps a dbt source
`raw.<table>` to `AssetKey(["raw", "<table>"])`, which is exactly the key
each raw Python asset in `weather_forecast_audit.assets` uses, so the dbt
staging models connect to them by construction (verified by
`tests/dagster/test_definitions.py`).

The manifest is `dbt/target/manifest.json`. `prepare_if_dev()` regenerates it
during `dagster dev`/`dg dev`; in the built Docker image it is instead baked
at build time (`RUN ... dbt parse`, see the Dockerfile) so the container
never needs a dev-time recompile. Outside both of those -- a plain
`pytest` run -- the gate runs `dbt parse` before pytest for the same reason
(see README/CI): `@dbt_assets` needs a loadable manifest file at import
time, and `prepare_if_dev()` is a no-op outside the `dagster dev` CLI.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dagster as dg
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

REPO_ROOT = Path(__file__).resolve().parents[2]
DBT_PROJECT_DIR = REPO_ROOT / "dbt"

dbt_project = DbtProject(project_dir=DBT_PROJECT_DIR, profiles_dir=DBT_PROJECT_DIR)
dbt_project.prepare_if_dev()


@dbt_assets(manifest=dbt_project.manifest_path)
def dbt_transform_assets(
    context: dg.AssetExecutionContext, dbt: DbtCliResource
) -> Iterator[Any]:
    yield from dbt.cli(["build"], context=context).stream()


dbt_resource = DbtCliResource(project_dir=dbt_project)
