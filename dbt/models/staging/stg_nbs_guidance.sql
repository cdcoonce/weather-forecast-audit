-- One row per (station, runtime, ftime) guidance point; txn/xnd/tmp renamed
-- to forecast_f/spread_f/tmp_f to match the mart's naming.
select
    station,
    runtime_utc,
    ftime_utc,
    cycle_hour,
    txn as forecast_f,
    xnd as spread_f,
    tmp as tmp_f
from {{ source('raw', 'nbs_guidance') }}
