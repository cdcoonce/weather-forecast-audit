#!/usr/bin/env python3
"""score_sources.py

Print `raw_nbm` vs `baseline` scores from `fct_forecast_verification` as a
markdown table (build spec #18 D4): one row per (`variable`, `lead_day`,
`source`) -- `n`, bias with its 95% CI, MAE, and skill with its CI -- plus
one pooled-over-leads row per source. A script, not package code: all the
actual statistics come from `weather_forecast_audit.scoring.score` (issue
#8), unmodified.

Requires `WFA_DUCKDB_PATH` set to a database that has already had `wfa
predict baseline` and `dbt build` run against it, so
`fct_forecast_verification` carries both sources.
"""

import os

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


def main() -> None:
    db_path = _db_path()
    with duckdb.connect(db_path) as conn:
        rows = conn.execute("select * from fct_forecast_verification").pl()

    print("## Per (variable, lead_day)\n")
    per_lead = score(rows, by=tuple(_GROUP_COLUMNS["per_lead"]))
    _print_table(per_lead, _GROUP_COLUMNS["per_lead"])

    print(
        "## Pooled over leads\n\n"
        "Pooled across `lead_day` within each `variable` -- labeled "
        "pooled-over-leads. See docs/methodology.md limitation 1 (pooled-lead "
        "slices understate correlation across leads): treat this row's CI as "
        "optimistic, not a precise interval.\n"
    )
    pooled = score(rows, by=tuple(_GROUP_COLUMNS["pooled"]))
    _print_table(pooled, _GROUP_COLUMNS["pooled"])


if __name__ == "__main__":
    main()
