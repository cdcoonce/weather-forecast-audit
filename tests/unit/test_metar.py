"""Pure METAR 6-hour max/min remark-group parsing tests.

The fixture expectations below were computed by hand from
`tests/fixtures/iem/asos_kphx_2023-07-13_2023-07-17.csv` against the build
spec's established facts (1sTTT/2sTTT groups at the :51 synoptic hours
05/11/17/23 UTC), then cross-checked against the spec's known-answer table
(07-15 max window uses 17:51->42.8, 23:51->47.8, 07-16 05:51->47.8; min
window uses 05:51->40.6, 11:51->34.4, 17:51->33.3).
"""

import csv
from pathlib import Path

import pytest

from weather_forecast_audit.iem.metar import SixHourGroups, parse_six_hour_groups

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
ASOS_FIXTURE = (
    REPO / "tests" / "fixtures" / "iem" / "asos_kphx_2023-07-13_2023-07-17.csv"
)


def _load_metars() -> dict[str, str]:
    with ASOS_FIXTURE.open(newline="") as handle:
        return {row["valid"]: row["metar"] for row in csv.DictReader(handle)}


METARS = _load_metars()


# -- fixture rows: synoptic (groups present) and non-synoptic (None) --------


@pytest.mark.parametrize(
    ("valid", "expected_max_c", "expected_min_c"),
    [
        # 07-15 synoptic hours
        ("2023-07-15 05:51", 46.1, 40.6),
        ("2023-07-15 11:51", 41.1, 34.4),
        ("2023-07-15 17:51", 42.8, 33.3),
        ("2023-07-15 23:51", 47.8, 41.7),
        # 07-15 non-synoptic hours (including the 06:51 24h-group hour)
        ("2023-07-15 00:51", None, None),
        ("2023-07-15 06:51", None, None),
        ("2023-07-15 12:51", None, None),
        ("2023-07-15 18:51", None, None),
        # 07-16 synoptic hours
        ("2023-07-16 05:51", 47.8, 40.6),
        ("2023-07-16 11:51", 41.1, 35.6),
        ("2023-07-16 17:51", 41.1, 34.4),
        ("2023-07-16 23:51", 45.6, 40.0),
        # 07-16 non-synoptic hours
        ("2023-07-16 00:51", None, None),
        ("2023-07-16 06:51", None, None),
        ("2023-07-16 12:51", None, None),
        ("2023-07-16 18:51", None, None),
    ],
)
def test_fixture_rows_give_expected_groups(
    valid: str, expected_max_c: float | None, expected_min_c: float | None
) -> None:
    result = parse_six_hour_groups(METARS[valid])
    assert result == SixHourGroups(max_c=expected_max_c, min_c=expected_min_c)


# -- sign handling ------------------------------------------------------------


def test_negative_max_group() -> None:
    metar = "KPHX 150551Z AO2 RMK AO2 T04110106 11006 20344"
    result = parse_six_hour_groups(metar)
    assert result.max_c == -0.6


def test_negative_min_group() -> None:
    metar = "KPHX 150551Z AO2 RMK AO2 T04110106 10461 21123"
    result = parse_six_hour_groups(metar)
    assert result.min_c == -12.3


# -- structural rules -----------------------------------------------------


def test_tokens_before_rmk_are_ignored() -> None:
    metar = "KPHX 150051Z 10478 20344 RMK AO2 T04560078"
    result = parse_six_hour_groups(metar)
    assert result == SixHourGroups(max_c=None, min_c=None)


def test_hourly_temp_group_not_matched() -> None:
    metar = "KPHX 150651Z 39/12 A2972 RMK AO2 SLP040 T03940122"
    result = parse_six_hour_groups(metar)
    assert result == SixHourGroups(max_c=None, min_c=None)


def test_24_hour_group_not_matched() -> None:
    metar = "KPHX 150651Z 39/12 A2972 RMK AO2 SLP040 T03940122 404670339"
    result = parse_six_hour_groups(metar)
    assert result == SixHourGroups(max_c=None, min_c=None)


def test_duplicate_max_group_raises() -> None:
    metar = "KPHX 150551Z RMK AO2 T04110106 10461 10462 20344"
    with pytest.raises(ValueError, match="max"):
        parse_six_hour_groups(metar)


def test_duplicate_min_group_raises() -> None:
    metar = "KPHX 150551Z RMK AO2 T04110106 10461 20344 20345"
    with pytest.raises(ValueError, match="min"):
        parse_six_hour_groups(metar)


# -- empty / missing --------------------------------------------------------


def test_empty_metar_gives_none() -> None:
    assert parse_six_hour_groups("") == SixHourGroups(max_c=None, min_c=None)


def test_missing_marker_metar_gives_none() -> None:
    assert parse_six_hour_groups("M") == SixHourGroups(max_c=None, min_c=None)


def test_no_rmk_section_gives_none() -> None:
    metar = "KPHX 150051Z 30006G15KT 10SM FEW120 42/14 A2976"
    assert parse_six_hour_groups(metar) == SixHourGroups(max_c=None, min_c=None)
