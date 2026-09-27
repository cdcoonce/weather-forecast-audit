"""NBS archive cycle regimes: which archived cycle is "nearest 12Z" and when.

The archived NBS cycle nearest 12Z changed on 2026-04-30 (build spec fact 2)
and, earlier, on 2020-02-25/26 (build spec #7 decision 5). The changeover
dates and canonical hours are probed facts, not derived here;
`dbt/seeds/nbs_cycle_regimes.csv` is their single source of truth so no date
is duplicated in Python. The first regime's `valid_from` is the pinned NBS
archive start date (`dbt/seeds/nbs_archive.csv`, `load_archive_start`),
found by `scripts/registry/probe_archive_start.py` (build spec #7
decision 6).
"""

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "nbs_cycle_regimes.csv"
DEFAULT_ARCHIVE_SEED_PATH = REPO_ROOT / "dbt" / "seeds" / "nbs_archive.csv"


@dataclass(frozen=True)
class CycleRegime:
    regime_id: str
    valid_from: date
    valid_to: date | None
    canonical_cycle_hour: int


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def load_cycle_regimes(path: Path = DEFAULT_SEED_PATH) -> list[CycleRegime]:
    """Load cycle regimes from the seed CSV, ordered as they appear on disk."""
    with path.open(newline="") as handle:
        return [
            CycleRegime(
                regime_id=row["regime_id"],
                valid_from=date.fromisoformat(row["valid_from"]),
                valid_to=_parse_date(row["valid_to"]),
                canonical_cycle_hour=int(row["canonical_cycle_hour"]),
            )
            for row in csv.DictReader(handle)
        ]


def canonical_cycle_hour(run_date: date, regimes: list[CycleRegime]) -> int:
    """The canonical archived cycle hour ("nearest 12Z") for a run date."""
    for regime in regimes:
        if regime.valid_from <= run_date and (
            regime.valid_to is None or run_date <= regime.valid_to
        ):
            return regime.canonical_cycle_hour
    msg = f"no regime covers run_date {run_date.isoformat()}"
    raise ValueError(msg)


def load_archive_start(path: Path = DEFAULT_ARCHIVE_SEED_PATH) -> date:
    """Load the pinned NBS archive start date from its seed CSV.

    See `scripts/registry/probe_archive_start.py` for the search that pinned
    this date and `docs/spikes/2026-09-26-station-registry/README.md` for
    the evidence.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        msg = f"expected exactly one row in {path}, found {len(rows)}"
        raise ValueError(msg)
    return date.fromisoformat(rows[0]["archive_start_date"])
