select
    station,
    local_date,
    high_f,
    low_f
from {{ source('raw', 'cli_daily') }}
