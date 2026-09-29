"""Pure METAR remark parsing for the 6-hour max/min temperature groups.

Parses only the fixed-width `1snTTT` (6-hour maximum) and `2snTTT` (6-hour
minimum) remark groups (build spec fact 1), from the remarks section (after
the first `RMK` token) of a raw METAR report. Every other remark group --
the hourly `TsTTTsTTT` temp/dewpoint group and the 24-hour `4snTTTsnTTT`
group -- is excluded because it does not fit the anchored 5-character
pattern, not because it is matched by name.

`s` (the sign digit) is 0 for positive and 1 for negative; `TTT` is tenths of
degrees Celsius. A group never guesses: if a group (max or min) appears
more than once in one report's remarks, identical repeats collapse to the one
value they share, but differing tokens raise rather than picking one of them.
"""

import re
from dataclasses import dataclass

_MAX_GROUP = re.compile(r"^1[01]\d{3}$")
_MIN_GROUP = re.compile(r"^2[01]\d{3}$")


@dataclass(frozen=True)
class SixHourGroups:
    """The parsed 6-hour max/min temperature remark groups, in Celsius."""

    max_c: float | None
    min_c: float | None


def _group_value_c(token: str) -> float:
    sign = -1.0 if token[1] == "1" else 1.0
    tenths = int(token[2:5])
    return sign * tenths / 10.0


def parse_six_hour_groups(metar: str) -> SixHourGroups:
    """Parse the 6-hour max/min remark groups from a raw METAR report.

    Only whitespace-delimited tokens after the first `RMK` token are
    considered; a report with no `RMK` section (including an empty or `M`
    report) yields `None` for both fields, as does one with no matching
    group. Identical repeated tokens collapse to one, since they leave nothing
    to choose between. If more than one distinct token matches a group's
    pattern, the report is ambiguous and this raises `ValueError` rather than
    guessing which one is authoritative.
    """
    tokens = metar.split()
    try:
        rmk_index = tokens.index("RMK")
    except ValueError:
        return SixHourGroups(max_c=None, min_c=None)

    remarks = tokens[rmk_index + 1 :]
    max_matches = list(
        dict.fromkeys(token for token in remarks if _MAX_GROUP.match(token))
    )
    min_matches = list(
        dict.fromkeys(token for token in remarks if _MIN_GROUP.match(token))
    )

    if len(max_matches) > 1:
        msg = (
            f"metar has {len(max_matches)} 6-hour max groups, "
            f"expected at most 1: {max_matches}"
        )
        raise ValueError(msg)
    if len(min_matches) > 1:
        msg = (
            f"metar has {len(min_matches)} 6-hour min groups, "
            f"expected at most 1: {min_matches}"
        )
        raise ValueError(msg)

    max_c = _group_value_c(max_matches[0]) if max_matches else None
    min_c = _group_value_c(min_matches[0]) if min_matches else None
    return SixHourGroups(max_c=max_c, min_c=min_c)
