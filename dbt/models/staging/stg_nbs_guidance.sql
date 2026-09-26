-- One row per (station, runtime, ftime) guidance point; txn/xnd renamed to
-- forecast_f/spread_f to match the mart's naming.
select
    station,
    runtime_utc,
    ftime_utc,
    cycle_hour,
    txn as forecast_f,
    xnd as spread_f,
    tmp
from {{ source('raw', 'nbs_guidance') }}
