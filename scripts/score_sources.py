#!/usr/bin/env python3
"""score_sources.py

Print `raw_nbm` vs `baseline` scores from `fct_forecast_verification` as a
markdown table (build spec #18 D4): one row per (`variable`, `lead_day`,
`source`) -- `n`, bias with its 95% CI, MAE, and skill with its CI -- plus
one pooled-over-leads row per source. A script, not package code: all the
actual statistics come from `weather_forecast_audit.scoring.score` (issue
#8), unmodified.

`--reference NAME` measures skill against another source in the table (for
example `baseline`, to ask whether a challenger beats the baseline rather
than raw NBM) instead of the default `raw_nbm`; the reference source is
printed above the tables. Without the flag the output is unchanged.

Requires `WFA_DUCKDB_PATH` set to a database that has already had `wfa
predict baseline` and `dbt build` run against it, so
`fct_forecast_verification` carries both sources.
"""

import argparse
import os
from collections.abc import Sequence

import duckdb
import polars as pl

from weather_forecast_audit.scoring import score

_GROUP_COLUMNS = {
    "per_lead": ["variable", "lead_day"],
    "pooled": ["variable"],
}


def _db_path() -> str:
    path = os.environ.get("WFA_DUCKDB_PATH")
    if not path:
        msg = "WFA_DUCKDB_PATH must be set to the DuckDB database file path"
        raise SystemExit(msg)
    return path


def _fmt_ci(value: float | None, lo: float | None, hi: float | None) -> str:
    if value is None:
        return "n/a"
    if lo is None or hi is None:
        return f"{value:.3f} [n/a]"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def _print_table(rows: pl.DataFrame, group_cols: list[str]) -> None:
    header = [*group_cols, "source", "n", "bias (95% CI)", "mae", "skill (95% CI)"]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows.sort([*group_cols, "source"]).iter_rows(named=True):
        cells = [str(row[c]) for c in group_cols]
        cells.append(row["source"])
        cells.append(str(row["n"]))
        cells.append(_fmt_ci(row["bias"], row["bias_lo"], row["bias_hi"]))
        cells.append(f"{row['mae']:.3f}")
        cells.append(_fmt_ci(row["skill"], row["skill_lo"], row["skill_hi"]))
        print("| " + " | ".join(cells) + " |")
    print()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--reference",
        metavar="NAME",
        default=None,
        help="source skill is measured against (default: raw_nbm)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    reference: str | None = args.reference
    score_kwargs = {} if reference is None else {"reference": reference}

    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        rows = conn.execute("select * from fct_forecast_verification").pl()

    if reference is not None:
        print(f"Reference source: `{reference}`\n")

    print("## Per (variable, lead_day)\n")
    try:
        per_lead = score(rows, by=tuple(_GROUP_COLUMNS["per_lead"]), **score_kwargs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _print_table(per_lead, _GROUP_COLUMNS["per_lead"])

    print(
        "## Pooled over leads\n\n"
        "Pooled across `lead_day` within each `variable` -- labeled "
        "pooled-over-leads. See docs/methodology.md limitation 1 (pooled-lead "
        "slices understate correlation across leads): treat this row's CI as "
        "optimistic, not a precise interval.\n"
    )
    pooled = score(rows, by=tuple(_GROUP_COLUMNS["pooled"]), **score_kwargs)
    _print_table(pooled, _GROUP_COLUMNS["pooled"])


if __name__ == "__main__":
    main()
