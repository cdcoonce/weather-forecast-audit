"""The CI pytest invocation must deselect `network` tests.

The marker expression is read from the workflow file itself, so dropping or
weakening `-m "not network"` in CI fails this test rather than drifting from it.
"""

import shlex
import textwrap
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
REQUIRED_MARKERS = {"unit", "integration", "io", "dagster", "dbt", "network"}


def _registered_markers() -> list[str]:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    return pyproject["tool"]["pytest"]["ini_options"]["markers"]


def _ci_pytest_marker_expression() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    commands = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "pytest" in step.get("run", "")
    ]
    assert len(commands) == 1, f"expected one pytest step in CI, found {commands}"
    args = shlex.split(commands[0])
    pytest_args = args[args.index("pytest") + 1 :]
    assert "-m" in pytest_args, f"CI pytest step selects no markers: {commands[0]}"
    return pytest_args[pytest_args.index("-m") + 1]


def test_required_markers_are_registered() -> None:
    names = {marker.split(":", 1)[0].strip() for marker in _registered_markers()}
    assert names >= REQUIRED_MARKERS


def test_ci_invocation_deselects_network_tests(pytester: pytest.Pytester) -> None:
    markers = "\n".join(f"    {m!r}," for m in _registered_markers())
    pytester.makepyprojecttoml(
        f"[tool.pytest.ini_options]\naddopts = '--strict-markers'\n"
        f"markers = [\n{markers}\n]\n"
    )
    pytester.makepyfile(
        test_sample=textwrap.dedent(
            """
            import pytest

            def test_offline():
                pass

            @pytest.mark.network
            def test_online():
                raise AssertionError("network test ran under the CI invocation")

            @pytest.mark.integration
            @pytest.mark.network
            def test_online_integration():
                raise AssertionError("network test ran under the CI invocation")
            """
        )
    )

    result = pytester.runpytest("-m", _ci_pytest_marker_expression())

    result.assert_outcomes(passed=1, deselected=2)
