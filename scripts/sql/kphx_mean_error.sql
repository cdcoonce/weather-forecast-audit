-- Headline verification query: KPHX raw NBM signed/absolute error and
-- scorability, by lead day and variable. mean_hourly_diagnostic_error_f is
-- not a scored metric: it is mean(forecast_f - hourly_observed_f) on
-- scorable (metar_6h) rows, kept only to show the sampling-bias artifact
-- issue #6 removed from the official error_f (docs/methodology.md,
-- "Observed extremes").
select
    lead_day,
    variable,
    avg(error_f) as mean_signed_error_f,
    avg(abs(error_f)) as mae_f,
    avg(
        case
            when scorable then forecast_f - hourly_observed_f
        end
    ) as mean_hourly_diagnostic_error_f,
    count(*) filter (where scorable) as n_scorable,
    count(*) filter (where not scorable) as n_unscorable
from fct_forecast_verification
where station = 'KPHX'
group by lead_day, variable
order by lead_day, variable;
