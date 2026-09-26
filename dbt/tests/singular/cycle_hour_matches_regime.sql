-- Every row's cycle_hour must equal the seed's canonical cycle hour for its
-- run_date; if it does not, canonical-cycle filtering upstream is broken.
select
    fct.station,
    fct.run_date,
    fct.cycle_hour,
    regimes.canonical_cycle_hour
from {{ ref('fct_forecast_verification') }} as fct
inner join {{ ref('nbs_cycle_regimes') }} as regimes
    on
        fct.run_date >= regimes.valid_from
        and (regimes.valid_to is null or fct.run_date <= regimes.valid_to)
where fct.cycle_hour != regimes.canonical_cycle_hour
