-- Headline verification query: KPHX raw NBM signed/absolute error and
-- scorability, by lead day and variable.
select
    lead_day,
    variable,
    avg(error_f) as mean_signed_error_f,
    avg(abs(error_f)) as mae_f,
    count(*) filter (where scorable) as n_scorable,
    count(*) filter (where not scorable) as n_unscorable
from fct_forecast_verification
where station = 'KPHX'
group by lead_day, variable
order by lead_day, variable;
