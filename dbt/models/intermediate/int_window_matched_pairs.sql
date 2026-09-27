-- Guidance txn rows joined to their resolved verification window, with a
-- CLI cross-check: the published daily high (for max) or low (for min) on
-- the target date, independent of the hourly-ASOS-derived observed_f.
with guidance as (
    select *
    from {{ ref('stg_nbs_guidance') }}
    where forecast_f is not null
),

resolved as (
    select *
    from {{ ref('stg_resolved_windows') }}
),

matched as (
    select
        resolved.station,
        resolved.runtime_utc,
        resolved.ftime_utc,
        resolved.variable,
        resolved.target_date,
        resolved.lead_day,
        resolved.window_start_utc,
        resolved.window_end_utc,
        resolved.observed_f,
        resolved.n_obs,
        resolved.hours_covered,
        resolved.hours_expected,
        resolved.scorable,
        resolved.extreme_source,
        resolved.periods_found,
        resolved.hourly_observed_f,
        guidance.cycle_hour,
        guidance.forecast_f,
        guidance.spread_f
    from resolved
    inner join guidance
        on
            resolved.station = guidance.station
            and resolved.runtime_utc = guidance.runtime_utc
            and resolved.ftime_utc = guidance.ftime_utc
),

cli as (
    select
        station,
        local_date,
        high_f,
        low_f
    from {{ ref('stg_cli_daily') }}
)

select
    matched.*,
    case
        when matched.variable = 'max' then cli.high_f
        else cli.low_f
    end as cli_f
from matched
left join cli
    on
        matched.station = cli.station
        and matched.target_date = cli.local_date
