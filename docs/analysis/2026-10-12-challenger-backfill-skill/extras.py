"""Pre-registered secondary outputs for the challenger backfill evaluation.

Warm-up split (first 12 months after 2020-09-29 versus the rest), monthly
fallback share, and a by-calendar-year table labelled exploratory. Read-only:

after ,  and .
"""
import duckdb, polars as pl
from datetime import date
from weather_forecast_audit.scoring import score
import os
c = duckdb.connect(os.environ["WFA_DUCKDB_PATH"], read_only=True)
rows = c.execute("select * from fct_forecast_verification where source in ('raw_nbm','challenger')").pl()
cut = date(2021, 9, 29)
def show(title, frame):
    s = score(frame, by=("variable", "lead_day")).filter(pl.col("source") == "challenger").sort("variable", "lead_day")
    print(f"## {title}")
    for r in s.iter_rows(named=True):
        flag = " MIN_SAMPLE_FLAG" if r.get("min_sample_flag") else ""
        print(f"  {r['variable']} lead{r['lead_day']} n={r['n']} skill={r['skill']:.4f} [{r['skill_lo']:.4f}, {r['skill_hi']:.4f}]{flag}")
show("warm-up: run_date < 2021-09-29 (first 12 months)", rows.filter(pl.col("run_date") < cut))
show("after warm-up: run_date >= 2021-09-29", rows.filter(pl.col("run_date") >= cut))
for y in range(2021, 2027):
    show(f"EXPLORATORY by calendar year {y}", rows.filter(pl.col("run_date").dt.year() == y))
print("## fallback share by month (challenger), months with fallback > 0")
fb = c.execute("""select strftime(run_date, '%Y-%m') as m, count(*) as n, avg(case when fallback then 1.0 else 0.0 end) as fb_share
                  from raw.model_predictions where source='challenger' group by 1 order by 1""").fetchall()
for m, n, sh in fb:
    if sh > 0: print(f"  {m} n={n} fallback_share={sh:.3f}")
print("months total", len(fb), "overall", c.execute("select avg(case when fallback then 1.0 else 0 end) from raw.model_predictions where source='challenger'").fetchone())
print("## challenger vs raw: row-set parity")
print(c.execute("""select (select count(*) from (select station,run_date,lead_day,variable from fct_forecast_verification where source='challenger' group by 1,2,3,4)),
                          (select count(*) from (select station,run_date,lead_day,variable from fct_forecast_verification where source='raw_nbm' group by 1,2,3,4))""").fetchone())
