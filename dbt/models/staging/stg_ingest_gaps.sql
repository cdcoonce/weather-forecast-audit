select
    station,
    source,
    expected,
    reason
from {{ source('raw', 'ingest_gaps') }}
