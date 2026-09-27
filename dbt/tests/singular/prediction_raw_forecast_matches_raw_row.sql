-- raw_forecast_f in stg_model_predictions must equal the raw row's own
-- forecast_f for the same (station, run_date, lead_day, variable): catches
-- a misaligned join key (e.g. a prediction join that drops lead_day and
-- fans out onto every lead of that station/run_date/variable) that would
-- otherwise silently pair a prediction with the wrong raw row (build spec
-- #18 D3.4, and the required teeth for the "join drops lead_day" mutation).
select
    predictions.station,
    predictions.run_date,
    predictions.lead_day,
    predictions.variable,
    predictions.raw_forecast_f,
    raw_pairs.forecast_f
from {{ ref('stg_model_predictions') }} as predictions
inner join {{ ref('int_raw_verification_pairs') }} as raw_pairs
    on
        predictions.station = raw_pairs.station
        and predictions.run_date = raw_pairs.run_date
        and predictions.lead_day = raw_pairs.lead_day
        and predictions.variable = raw_pairs.variable
where abs(predictions.raw_forecast_f - raw_pairs.forecast_f) > 0.0001
