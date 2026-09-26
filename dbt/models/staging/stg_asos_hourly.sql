select
    station,
    valid_utc,
    tmpf
from {{ source('raw', 'asos_hourly') }}
