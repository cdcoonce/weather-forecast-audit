{#
    Multi-column uniqueness, written in-project (no dbt packages; CI has no
    network to fetch dbt_utils). Fails if any combination of the given
    columns appears more than once in the model.
#}
{% test unique_combination_of_columns(model, combination_of_columns) %}

with validation_errors as (
    select
        {{ combination_of_columns | join(', ') }}
    from {{ model }}
    group by {{ combination_of_columns | join(', ') }}
    having count(*) > 1
)

select *
from validation_errors

{% endtest %}
