select
    station,
    source,
    expected,
    reason,
    first_seen
from {{ source('raw', 'ingest_gaps') }}
