-- Cycle-hour counts by run-date month, around the 2026-04-30 changeover:
-- confirms 13Z is canonical through April 2026 and 12Z from May 2026.
select
    date_trunc('month', run_date) as run_month,
    cycle_hour,
    count(*) as n_runs
from fct_forecast_verification
where station = 'KPHX'
group by date_trunc('month', run_date), cycle_hour
order by run_month, cycle_hour;
