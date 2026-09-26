-- Grain: station x run_date x lead_day x variable x source. v1 only
-- populates source = 'raw_nbm' and lead_day 1-3 (lead 0 and non-canonical
-- rows are dropped here; they still exist upstream in raw/staging).
{{ config(materialized='table') }}

select
    station,
    cast(runtime_utc as date) as run_date,
    lead_day,
    variable,
    'raw_nbm' as source,
    cycle_hour,
    runtime_utc,
    target_date,
    forecast_f,
    spread_f,
    observed_f,
    window_start_utc,
    window_end_utc,
    n_obs,
    hours_covered,
    scorable,
    cli_f,
    case when scorable then forecast_f - observed_f end as error_f
from {{ ref('int_window_matched_pairs') }}
where lead_day between 1 and 3
