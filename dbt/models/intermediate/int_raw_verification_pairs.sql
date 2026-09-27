-- The raw_nbm half of fct_forecast_verification's grain (station x run_date
-- x lead_day x variable x source), split out (issue #18 D3) so a model
-- prediction (source != 'raw_nbm') can be joined against it without
-- fct_forecast_verification depending on itself: fct selects from this
-- model, union all a predictions source, keeping the lineage int ->
-- predictions -> fct acyclic. Lead 0 and non-canonical rows are dropped
-- here; they still exist upstream in raw/staging.
--
-- Each row is tagged with the NBM operational version
-- (dbt/seeds/nbm_versions.csv) covering its runtime_utc and the archived
-- cycle regime (dbt/seeds/nbs_cycle_regimes.csv) covering its run_date
-- (issue #12). Both seeds are non-overlapping, gap-free ranges over the
-- archive's history, so these left joins add columns without changing the
-- grain -- fct_forecast_verification's unique_combination_of_columns test
-- on (station, run_date, lead_day, variable, source) guards that.
{{ config(materialized='table') }}

with base as (
    select
        station,
        cast(runtime_utc as date) as run_date,
        lead_day,
        variable,
        'raw_nbm' as source,
        cycle_hour,
        runtime_utc,
        target_date,
        forecast_f,
        spread_f,
        observed_f,
        window_start_utc,
        window_end_utc,
        n_obs,
        hours_covered,
        scorable,
        extreme_source,
        periods_found,
        hourly_observed_f,
        cli_f,
        case when scorable then forecast_f - observed_f end as error_f
    from {{ ref('int_window_matched_pairs') }}
    where lead_day between 1 and 3
)

select
    base.*,
    nbm_versions.nbm_version,
    nbs_cycle_regimes.regime_id as cycle_regime
from base
left join {{ ref('nbm_versions') }} as nbm_versions
    on
        base.runtime_utc >= nbm_versions.valid_from_utc
        and (
            nbm_versions.valid_to_utc is null
            or base.runtime_utc < nbm_versions.valid_to_utc
        )
left join {{ ref('nbs_cycle_regimes') }} as nbs_cycle_regimes
    on
        base.run_date >= nbs_cycle_regimes.valid_from
        and (
            nbs_cycle_regimes.valid_to is null
            or base.run_date <= nbs_cycle_regimes.valid_to
        )
