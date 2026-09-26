from importlib.metadata import version

import pytest

import weather_forecast_audit

pytestmark = pytest.mark.unit


def test_package_version_matches_distribution_metadata() -> None:
    assert weather_forecast_audit.__version__ == version("weather-forecast-audit")
