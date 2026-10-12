"""Pre-registered secondary outputs for the challenger backfill evaluation.

Warm-up split (first 12 months after 2020-09-29 versus the rest), monthly
fallback share, and a by-calendar-year table labelled exploratory. Read-only:

    WFA_DUCKDB_PATH=... uv run python \\
        docs/analysis/2026-10-12-challenger-backfill-skill/extras.py

after `wfa predict baseline`, `wfa predict challenger` and `dbt build`.
"""

import os
from datetime import date

import duckdb
import polars as pl

from weather_forecast_audit.scoring import score

WARM_UP_END = date(2021, 9, 29)

FALLBACK_BY_MONTH = """
select
    strftime(run_date, '%Y-%m') as m,
    count(*) as n,
    avg(case when fallback then 1.0 else 0.0 end) as fb_share
from raw.model_predictions
where source = 'challenger'
group by 1
order by 1
"""

FALLBACK_OVERALL = """
select avg(case when fallback then 1.0 else 0.0 end)
from raw.model_predictions
where source = 'challenger'
"""

KEY_COUNT = """
select count(*) from (
    select station, run_date, lead_day, variable
    from fct_forecast_verification
    where source = ?
    group by 1, 2, 3, 4
)
"""


def show(title: str, frame: pl.DataFrame) -> None:
    result = (
        score(frame, by=("variable", "lead_day"))
        .filter(pl.col("source") == "challenger")
        .sort("variable", "lead_day")
    )
    print(f"## {title}")
    for r in result.iter_rows(named=True):
        flag = " MIN_SAMPLE_FLAG" if r.get("min_sample_flag") else ""
        print(
            f"  {r['variable']} lead{r['lead_day']} n={r['n']} "
            f"skill={r['skill']:.4f} [{r['skill_lo']:.4f}, {r['skill_hi']:.4f}]"
            f"{flag}"
        )


def main() -> None:
    conn = duckdb.connect(os.environ["WFA_DUCKDB_PATH"], read_only=True)
    rows = conn.execute(
        "select * from fct_forecast_verification "
        "where source in ('raw_nbm', 'challenger')"
    ).pl()

    show(
        "warm-up: run_date < 2021-09-29 (first 12 months)",
        rows.filter(pl.col("run_date") < WARM_UP_END),
    )
    show(
        "after warm-up: run_date >= 2021-09-29",
        rows.filter(pl.col("run_date") >= WARM_UP_END),
    )
    for year in range(2021, 2027):
        show(
            f"EXPLORATORY by calendar year {year}",
            rows.filter(pl.col("run_date").dt.year() == year),
        )

    print("## fallback share by month (challenger), months with fallback > 0")
    by_month = conn.execute(FALLBACK_BY_MONTH).fetchall()
    for month, n, share in by_month:
        if share > 0:
            print(f"  {month} n={n} fallback_share={share:.3f}")
    overall = conn.execute(FALLBACK_OVERALL).fetchone()
    print("months total", len(by_month), "overall", overall)

    print("## challenger vs raw: row-set parity")
    challenger_keys = conn.execute(KEY_COUNT, ["challenger"]).fetchone()
    raw_keys = conn.execute(KEY_COUNT, ["raw_nbm"]).fetchone()
    print((challenger_keys[0], raw_keys[0]))


if __name__ == "__main__":
    main()
