select
    station,
    valid_utc,
    tmpf,
    max_6h_f,
    min_6h_f
from {{ source('raw', 'asos_hourly') }}
