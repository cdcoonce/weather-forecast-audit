"""POST-HOC, NOT PRE-REGISTERED: interval coverage on short slices.

`PREREG.md`'s replay scores full-year series (n = 365). Published slices are
often much shorter: one season is about 90 issuance dates, and
`scoring.MIN_SAMPLE_DATES` admits a slice at 30. With 14-day blocks a
30-date slice holds about three blocks, and a percentile bootstrap over three
resampled units is close to degenerate. This script sizes that effect with the
same ten fitted AR models (`results/ar_fits.csv`) so a follow-up issue can
set a block-aware minimum sample. It informs nothing in `PREREG.md`'s decision
rule, which was settled before this ran.

Run: `uv run python docs/analysis/2026-09-27-block-length/posthoc_short_slices.py`
Writes `results/posthoc_short_slices.csv`.
"""

from pathlib import Path

import numpy as np
import polars as pl

from weather_forecast_audit.block_length import ArFit, replay_coverage

HERE = Path(__file__).resolve().parent
SEED = 20260928
N_SIMS = 400
N_BOOT = 500
SLICE_LENGTHS = (30, 90, 365)
BLOCK_LENGTHS = (1, 7, 14)


def main() -> None:
    fits = pl.read_csv(HERE / "results" / "ar_fits.csv")
    children = np.random.SeedSequence(SEED).spawn(fits.height)
    rows = []
    for i, fit_row in enumerate(fits.iter_rows(named=True)):
        phi_text = fit_row["phi"] or ""
        phi = np.array([float(v) for v in str(phi_text).split(";") if v != ""])
        fit = ArFit(
            p=int(fit_row["p"]),
            phi=phi,
            sigma2=float(fit_row["sigma2"]),
            mean=float(fit_row["mean"]),
        )
        for n in SLICE_LENGTHS:
            for block_days in BLOCK_LENGTHS:
                rng = np.random.default_rng(children[i])
                coverage = replay_coverage(
                    fit, n, block_days, n_sims=N_SIMS, rng=rng, n_boot=N_BOOT
                )
                rows.append(
                    {
                        "series": fit_row["series"],
                        "variable": fit_row["variable"],
                        "lead_day": fit_row["lead_day"],
                        "n_dates": n,
                        "block_days": block_days,
                        "n_blocks": -(-n // block_days),
                        "coverage": coverage,
                    }
                )
                print(fit_row["series"], fit_row["variable"], n, block_days, coverage)
    pl.DataFrame(rows).write_csv(HERE / "results" / "posthoc_short_slices.csv")


if __name__ == "__main__":
    main()
