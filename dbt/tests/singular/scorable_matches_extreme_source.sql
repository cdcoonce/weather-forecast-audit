-- scorable must be true exactly when extreme_source = 'metar_6h', and false
-- exactly when extreme_source = 'none' (issue #6: no other source values
-- exist, so this is an exhaustive check, not just a spot check).
select *
from {{ ref('fct_forecast_verification') }}
where
    (scorable and extreme_source != 'metar_6h')
    or (not scorable and extreme_source != 'none')
