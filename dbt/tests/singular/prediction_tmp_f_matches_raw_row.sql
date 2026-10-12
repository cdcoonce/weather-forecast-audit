-- tmp_f is a guidance-side property of the run (issue #19): every
-- prediction row in fct_forecast_verification must carry exactly the tmp_f
-- of the raw_nbm row it was joined back to, for every source. A join that
-- re-derives tmp_f from somewhere else, or a prediction branch that drops
-- the column (null where the raw row has a value), would let a model's
-- feature differ between the raw and corrected rows of one pair.
select
    fct.station,
    fct.run_date,
    fct.lead_day,
    fct.variable,
    fct.source,
    fct.tmp_f,
    raw_pairs.tmp_f as raw_tmp_f
from {{ ref('fct_forecast_verification') }} as fct
inner join {{ ref('int_raw_verification_pairs') }} as raw_pairs
    on
        fct.station = raw_pairs.station
        and fct.run_date = raw_pairs.run_date
        and fct.lead_day = raw_pairs.lead_day
        and fct.variable = raw_pairs.variable
where fct.tmp_f is distinct from raw_pairs.tmp_f
