# weather-forecast-audit

A living, public calibration audit of NWS forecast guidance (National Blend of Models): where, when, and by how much daily max/min temperature forecasts are systematically wrong — with a walk-forward bias-correction experiment scored daily.

The National Blend of Models (NBM) is the machine forecast NWS forecasters start from. This project grades its daily max/min temperature guidance at the NWS climate-report stations across the continental US, against what was actually observed, and publishes where it runs warm or cold, by season and lead time. Every published number carries a confidence interval. Inside the audit, a rolling-bias baseline and a gradient-boosted challenger issue corrected forecasts, evaluated walk-forward so they never see the future, and a live scorecard grades raw NBM against both. "The ML can't beat the NBM" is a reportable finding.

The plan is [PRD #1](https://github.com/cdcoonce/weather-forecast-audit/issues/1).

## Background

- [`docs/brainstorms/`](docs/brainstorms/) — why this project, and the alternatives that were killed.
- [`docs/spikes/`](docs/spikes/) — the archive-parity spike: the Iowa Environmental Mesonet's NBM station-guidance endpoint (`model=NBS`) serves both the historical archive and the live runs, so backfill and live feed grade one product.

## Local setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if needed.

On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`. Linux needs `libgomp1`, which the Docker image installs.

```bash
uv sync
```

The same gate CI runs:

```bash
uv run ruff check .
uv run pytest -m "not network"
export WFA_DUCKDB_PATH="$PWD/.ci/warehouse.duckdb" && mkdir -p .ci
uv run dbt build --project-dir dbt --profiles-dir dbt
uv run sqlfluff lint dbt/models
```

### Layout

| Path | What |
| --- | --- |
| `src/weather_forecast_audit/` | Python package |
| `dbt/` | dbt project on `dbt-duckdb`; the database file is `$WFA_DUCKDB_PATH` |
| `dbt/profiles.yml` | Checked-in profile, env vars only; also carries an unused `snowflake` target |
| `dbt/.sqlfluff` | Snowflake-dialect lint config (the portability guard) |
| `tests/` | pytest suite |
| `src/weather_forecast_audit/definitions.py` | Dagster code location, served on gRPC 4002 on rammingspeed |
| `Dockerfile` | Code-location image; also the image every run container starts from |
| `scripts/offline.sh` | Runs a command with no network access (Linux CI only) |

### Conventions

- **Test markers:** `unit`, `integration`, `io`, `dagster`, `dbt`, `network`. Mark any test that reaches the internet `network`; CI deselects those.
- **Snowflake portability:** models run on DuckDB but must parse as Snowflake SQL. `sqlfluff lint` fails on DuckDB-only syntax, such as list comprehensions `[x * 2 for x in xs]`. Write the portable form.
- **Network-free CI:** after `uv sync`, every CI step runs through `scripts/offline.sh`, a network namespace with no route out. A step that needs the network fails there, not silently later.
