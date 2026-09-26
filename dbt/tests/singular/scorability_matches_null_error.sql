-- A scorable row must have both observed_f and error_f; an unscorable row
-- must have neither. Fails (returns rows) on any mismatch.
select *
from {{ ref('fct_forecast_verification') }}
where
    (scorable and (observed_f is null or error_f is null))
    or (not scorable and error_f is not null)
