select
    station,
    run_date,
    runtime_utc,
    lead_day,
    variable,
    target_date,
    source,
    forecast_f,
    raw_forecast_f,
    retrained_on,
    trained_through,
    fallback,
    params,
    generated_at
from {{ source('raw', 'model_predictions') }}
