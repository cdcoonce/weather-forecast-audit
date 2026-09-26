-- error_f must equal forecast_f - observed_f for every scorable row.
select *
from {{ ref('fct_forecast_verification') }}
where
    scorable
    and abs(error_f - (forecast_f - observed_f)) > 0.0001
