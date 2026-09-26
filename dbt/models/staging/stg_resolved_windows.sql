select
    station,
    runtime_utc,
    ftime_utc,
    variable,
    target_date,
    lead_day,
    window_start_utc,
    window_end_utc,
    observed_f,
    n_obs,
    hours_covered,
    hours_expected,
    scorable
from {{ source('raw', 'resolved_windows') }}
