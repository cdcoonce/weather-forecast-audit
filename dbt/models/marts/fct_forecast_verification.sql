-- Grain: station x run_date x lead_day x variable x source. raw_nbm rows
-- come from int_raw_verification_pairs verbatim; every other source's rows
-- (issue #18: baseline, and later challenger) are a model prediction joined
-- back to its raw verification pair on (station, run_date, lead_day,
-- variable), which is where every observation-side column (observed_f, the
-- windows, scorable, extreme_source, ...) plus nbm_version and
-- cycle_regime come from. This keeps the lineage int_raw_verification_pairs
-- -> raw.model_predictions -> fct acyclic: the Python side that produces
-- predictions reads int_raw_verification_pairs, never this model.
{{ config(materialized='table') }}

with raw_pairs as (
    select * from {{ ref('int_raw_verification_pairs') }}
),

predictions as (
    select * from {{ ref('stg_model_predictions') }}
),

prediction_rows as (
    select
        raw_pairs.station,
        raw_pairs.run_date,
        raw_pairs.lead_day,
        raw_pairs.variable,
        predictions.source,
        raw_pairs.cycle_hour,
        raw_pairs.runtime_utc,
        raw_pairs.target_date,
        predictions.forecast_f,
        raw_pairs.spread_f,
        raw_pairs.observed_f,
        raw_pairs.window_start_utc,
        raw_pairs.window_end_utc,
        raw_pairs.n_obs,
        raw_pairs.hours_covered,
        raw_pairs.scorable,
        raw_pairs.extreme_source,
        raw_pairs.periods_found,
        raw_pairs.hourly_observed_f,
        raw_pairs.cli_f,
        case
            when raw_pairs.scorable
                then predictions.forecast_f - raw_pairs.observed_f
        end as error_f,
        raw_pairs.nbm_version,
        raw_pairs.cycle_regime
    from predictions
    inner join raw_pairs
        on
            predictions.station = raw_pairs.station
            and predictions.run_date = raw_pairs.run_date
            and predictions.lead_day = raw_pairs.lead_day
            and predictions.variable = raw_pairs.variable
)

select * from raw_pairs
union all
select * from prediction_rows
