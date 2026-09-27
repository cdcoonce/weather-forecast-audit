"""Shared gap-tracking types for the IEM clients.

Every fetcher returns rows plus the gaps it could not fill, rather than
raising on a missing run or a data-free range: a missing day is a fact about
the archive worth recording (and loading into `raw.ingest_gaps`), not a
reason to abort the whole ingest.
"""

from dataclasses import dataclass
from typing import Literal

Reason = Literal[
    "missing_run", "no_txn", "missing_observations", "missing_report"
]


@dataclass(frozen=True)
class GapRecord:
    station: str
    source: Literal["nbs", "asos", "cli"]
    expected: str
    reason: str


@dataclass(frozen=True)
class FetchResult[T]:
    rows: list[T]
    gaps: list[GapRecord]
