-- One row per regime boundary, unioning the two seeds that tag
-- fct_forecast_verification (issue #12): NBM operational versions
-- (dbt/seeds/nbm_versions.csv) and archived NBS cycle regimes
-- (dbt/seeds/nbs_cycle_regimes.csv). This is annotation data for a
-- consumer to overlay on a time series; it does not itself build an
-- export or bump any export schema version -- that is issue #9's job.
{{ config(materialized='table') }}

with nbm_version_boundaries as (
    select
        'nbm_version' as kind,
        nbm_version as id,
        valid_from_utc as starts_at,
        verified,
        source_url as citation
    from {{ ref('nbm_versions') }}
),

cycle_regime_boundaries as (
    select
        'cycle_regime' as kind,
        regime_id as id,
        cast(valid_from as timestamp) as starts_at,
        cast(null as boolean) as verified,
        note as citation
    from {{ ref('nbs_cycle_regimes') }}
)

select * from nbm_version_boundaries
union all
select * from cycle_regime_boundaries
order by starts_at
