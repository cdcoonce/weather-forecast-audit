-- Every raw.model_predictions row must join to its raw verification pair on
-- (station, run_date, lead_day, variable); an orphaned prediction would
-- otherwise be silently dropped by fct_forecast_verification's inner join
-- instead of failing loudly here (build spec #18 D3.4).
select predictions.*
from {{ ref('stg_model_predictions') }} as predictions
left join {{ ref('int_raw_verification_pairs') }} as raw_pairs
    on
        predictions.station = raw_pairs.station
        and predictions.run_date = raw_pairs.run_date
        and predictions.lead_day = raw_pairs.lead_day
        and predictions.variable = raw_pairs.variable
where raw_pairs.station is null
