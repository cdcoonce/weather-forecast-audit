import pytest
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.unit


def test_bare_postgresql_url_resolves_to_psycopg2() -> None:
    # dagster-postgres builds a bare `postgresql://` URL and ships psycopg2.
    # SQLAlchemy 2.1 made psycopg (v3) the default driver for that URL, which
    # broke every run worker on rammingspeed with "No module named 'psycopg'".
    assert make_url("postgresql://user@host/db").get_dialect().driver == "psycopg2"
