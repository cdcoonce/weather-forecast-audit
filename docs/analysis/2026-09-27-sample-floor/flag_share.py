"""Reported, not binding (PREREG.md): how many published cells each floor flags.

Scores the committed #31 extract (12 stations, 2025 issuance dates) at the
export's grain, `(station, variable, lead_day, season)`, and counts the cells
that the old 30-date floor and the new 21-block floor would let carry a bias
claim (`no_detectable_bias` false and not sample-flagged).

Run: `uv run python docs/analysis/2026-09-27-sample-floor/flag_share.py`
Writes `results/flag_share.csv`.
"""

from pathlib import Path

import polars as pl

from weather_forecast_audit.scoring import score

HERE = Path(__file__).resolve().parent
EXTRACT = HERE.parent / "2026-09-27-block-length" / "data" / "verification_rows.parquet"
OLD_MIN_SAMPLE_DATES = 30


def main() -> None:
    rows = pl.read_parquet(EXTRACT)
    cells = score(rows, by=("station", "variable", "lead_day", "season"))
    claims_old = ~pl.col("no_detectable_bias") & (
        pl.col("n_dates") >= OLD_MIN_SAMPLE_DATES
    )
    claims_new = ~pl.col("no_detectable_bias") & ~pl.col("min_sample_flag")
    summary = (
        cells.with_columns(lead_scope=pl.lit("all"))
        .vstack(
            cells.filter(pl.col("lead_day") == 1).with_columns(
                lead_scope=pl.lit("lead_1")
            )
        )
        .group_by("lead_scope", maintain_order=True)
        .agg(
            cells=pl.len(),
            n_blocks_min=pl.col("n_blocks").min(),
            n_blocks_max=pl.col("n_blocks").max(),
            claims_old_floor=claims_old.sum(),
            claims_new_floor=claims_new.sum(),
        )
    )
    (HERE / "results").mkdir(exist_ok=True)
    summary.write_csv(HERE / "results" / "flag_share.csv")
    print(summary)


if __name__ == "__main__":
    main()
