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
value they share, but differing tokens are ambiguous and never picked between.
`parse_six_hour_groups` raises on that; `parse_six_hour_groups_tolerant`
instead withholds the ambiguous kind's value and names the kind, so a caller
that cannot stop for one bad report can record it.
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


@dataclass(frozen=True)
class TolerantSixHourGroups:
    """`SixHourGroups` plus which kinds ("max", "min") were ambiguous.

    An ambiguous kind has a `None` value in `groups`; the other kind is
    unaffected. `ambiguous` is ordered max before min.
    """

    groups: SixHourGroups
    ambiguous: tuple[str, ...]


def _group_value_c(token: str) -> float:
    sign = -1.0 if token[1] == "1" else 1.0
    tenths = int(token[2:5])
    return sign * tenths / 10.0


def _distinct_group_tokens(metar: str) -> tuple[list[str], list[str]]:
    """The distinct 6-hour (max, min) tokens in the remarks, in first-seen order.

    Identical repeats collapse; a report with no `RMK` token yields two empty
    lists.
    """
    tokens = metar.split()
    try:
        rmk_index = tokens.index("RMK")
    except ValueError:
        return [], []

    remarks = tokens[rmk_index + 1 :]
    max_matches = list(
        dict.fromkeys(token for token in remarks if _MAX_GROUP.match(token))
    )
    min_matches = list(
        dict.fromkeys(token for token in remarks if _MIN_GROUP.match(token))
    )
    return max_matches, min_matches


def _value_or_none(matches: list[str]) -> float | None:
    return _group_value_c(matches[0]) if len(matches) == 1 else None


def parse_six_hour_groups_tolerant(metar: str) -> TolerantSixHourGroups:
    """Parse the 6-hour max/min groups, reporting ambiguity instead of raising.

    Tokenising and identical-repeat collapse are the same as
    `parse_six_hour_groups`. A kind with one distinct token yields its value;
    a kind with none yields `None` (not ambiguous); a kind with two or more
    distinct tokens yields `None` and is named in `ambiguous`. Max and min are
    independent: an ambiguous max leaves a clean min intact.
    """
    max_matches, min_matches = _distinct_group_tokens(metar)
    ambiguous = tuple(
        kind
        for kind, matches in (("max", max_matches), ("min", min_matches))
        if len(matches) > 1
    )
    groups = SixHourGroups(
        max_c=_value_or_none(max_matches), min_c=_value_or_none(min_matches)
    )
    return TolerantSixHourGroups(groups=groups, ambiguous=ambiguous)


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
    max_matches, min_matches = _distinct_group_tokens(metar)

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

    return SixHourGroups(
        max_c=_value_or_none(max_matches), min_c=_value_or_none(min_matches)
    )
