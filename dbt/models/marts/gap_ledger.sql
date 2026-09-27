-- One row per (station, kind, gap_date, reason): the gap's expected date
-- (the same parse `warehouse.load_gaps` uses to scope its delete/insert),
-- plus first_seen, the wall-clock time this gap was first observed on file
-- (build spec D3). `date` is a reserved word under the Snowflake dialect
-- (sqlfluff RF04), so the column is named `gap_date`.
{{ config(materialized='table') }}

select
    cast(substr(expected, 1, 10) as date) as gap_date,
    station,
    source as kind,
    reason,
    first_seen
from {{ ref('stg_ingest_gaps') }}
