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
export WFA_DUCKDB_PATH="$PWD/.ci/warehouse.duckdb" && mkdir -p .ci
uv run dbt parse --project-dir dbt --profiles-dir dbt  # bakes dbt/target/manifest.json
uv run pytest -m "not network"
uv run wfa init-db
uv run dbt build --project-dir dbt --profiles-dir dbt
uv run sqlfluff lint dbt/models
```

The `dbt parse` step has to run before `pytest`: `weather_forecast_audit.definitions`
loads dbt models as Dagster assets via `@dbt_assets(manifest=...)`, which
needs an existing `dbt/target/manifest.json` at import time. `dbt/target/`
is gitignored (rebuilt every time), so a fresh checkout has no manifest
until something creates one; the Docker image does this at build time
(`dbt parse` against a dummy `WFA_DUCKDB_PATH`, since parse opens no
connection but the profile's `env_var()` must still resolve), and local/CI
runs do the same thing as a gate step for the same reason.

### The `wfa` CLI

`wfa` fetches, loads, and resolves NWS archive data into the DuckDB
warehouse. `WFA_DUCKDB_PATH` is required for every command:

```bash
export WFA_DUCKDB_PATH="$PWD/.ci/warehouse.duckdb"
uv run wfa init-db
uv run wfa ingest --station KPHX --start 2023-07-14 --end 2023-07-14
uv run wfa export --out /tmp/wfa-export
```

`wfa export` writes the versioned export contract (`export_schema/v1/`):
`manifest.json`, `city_index.json`, one `cities/{ICAO}.json` per station in
the registry, and a `completeness.json` stub. It reads
`fct_forecast_verification` (`source = raw_nbm` only in v1), so `dbt build`
must have run first.

### Dagster: materializing one partition for a station subset

`weather_forecast_audit.definitions:defs` exposes four daily-partitioned raw
assets (`raw/nbs_guidance`, `raw/asos_hourly`, `raw/cli_daily`,
`raw/resolved_windows`, batched together in `ingest_job`), plus
`transform_job` (the dbt models) and `freshness_check_job` (the two
freshness checks only -- no schedule is wired up yet; that's issue #16).
`StationsResource.only` restricts a run to a station subset without editing
the seed. From a Python shell or a script, with `WFA_DUCKDB_PATH` set:

```python
from dagster import DagsterInstance
from weather_forecast_audit.definitions import defs, ingest_job
from weather_forecast_audit.resources import StationsResource

job = defs.resolve_job_def("ingest_job")
with DagsterInstance.get() as instance:
    result = job.execute_in_process(
        instance=instance,
        partition_key="2023-07-14",
        run_config={"resources": {"stations": {"config": {"only": ["KPHX"]}}}},
    )
```

Materializing a recent `raw/resolved_windows` partition before its
dependent `raw/asos_hourly` partitions exist yet (out to `D+4`) is expected
to leave some windows unscorable; re-materialize once they land.

### Tracer bullet

`scripts/tracer_kphx.sh` ingests ~5 months of KPHX guidance and observations
(straddling the 2026-04-30 NBS cycle changeover), builds dbt, and prints the
headline verification queries. It makes real, politely rate-limited network
requests to IEM:

```bash
export WFA_DUCKDB_PATH="$PWD/.ci/warehouse.duckdb"
./scripts/tracer_kphx.sh
```

## Site

`site/` is a static React + Vite + TypeScript app (Poppins, grayscale
chrome, light and dark themes, no UI framework, no router library) that
reads the export contract as static JSON files under `data/`.

```bash
cd site
npm install
npm run dev
```

Dev data lives at `site/public/data/` (gitignored: never hand-write it).
Regenerate it from a warehouse you've already built (`wfa init-db` +
ingest + `dbt build`, or `./scripts/tracer_kphx.sh` -- see "Local setup"
above and "Tracer bullet" below):

```bash
export WFA_DUCKDB_PATH="$PWD/../.ci/warehouse.duckdb"  # from site/
npm run generate:data
```

That script is a thin wrapper around `uv run wfa export --out public/data`
run from the repo root; it never writes the JSON itself.

Site gate, mirroring CI's `site` job:

```bash
cd site
npm ci
npm run lint     # tsc --noEmit, then ESLint
npm test -- --run
npm run build
```

### Layout

| Path | What |
| --- | --- |
| `src/weather_forecast_audit/` | Python package |
| `src/weather_forecast_audit/iem/` | IEM HTTP client (`http.py`) and per-product parsers (`guidance.py`, `observations.py`) |
| `src/weather_forecast_audit/resolver.py` | Pure NBM verification-window resolver (no I/O, no time zones) |
| `src/weather_forecast_audit/warehouse.py` | Owns all `raw.*` DDL and the idempotent DuckDB loaders |
| `src/weather_forecast_audit/pipeline.py` | `ingest_station`/`resolve_station`: fetch, load, resolve one station |
| `src/weather_forecast_audit/export.py` | The v1 export contract: manifest, city index, per-city stats, completeness stub |
| `src/weather_forecast_audit/cli.py` | The `wfa` CLI (`init-db`, `ingest`, `export`) |
| `export_schema/v1/` | The four JSON Schemas (draft 2020-12) for the export contract |
| `export_schema/lock.json` | SHA-256 lock of every schema version, append-only |
| `dbt/` | dbt project on `dbt-duckdb`; the database file is `$WFA_DUCKDB_PATH` |
| `dbt/profiles.yml` | Checked-in profile, env vars only; also carries an unused `snowflake` target |
| `dbt/.sqlfluff` | Snowflake-dialect lint config (the portability guard) |
| `dbt/seeds/` | `station_registry.csv`/`station_exclusions.csv` (built by `scripts/registry/build_registry.py`), `nbs_cycle_regimes.csv`/`nbs_archive.csv` (built by `scripts/registry/probe_archive_start.py`) — the only source of station metadata, cycle-changeover dates, and the pinned archive start |
| `dbt/models/marts/fct_forecast_verification.sql` | The verification fact table |
| `dbt/models/marts/gap_ledger.sql` | One row per open ingest gap, with `first_seen` |
| `docs/methodology.md` | Verification windows, scope, completeness threshold, orchestration/partitions, and known caveats |
| `tests/` | pytest suite |
| `tests/support/fixture_fetcher.py` | Shared IEM fixture fetcher for DB-backed tests |
| `src/weather_forecast_audit/definitions.py` | Dagster code location, served on gRPC 4002 on rammingspeed |
| `src/weather_forecast_audit/assets.py` | The four daily-partitioned raw ingest assets, `raw/ingest_gaps`, and the freshness/gap-rate checks |
| `src/weather_forecast_audit/dbt_assets.py` | dbt models loaded as Dagster assets (`@dbt_assets`) |
| `src/weather_forecast_audit/resources.py` | `WarehouseResource`/`IemResource`/`StationsResource`/`ClockResource` |
| `src/weather_forecast_audit/checks.py` | Pure freshness/gap-rate evaluators the asset checks wrap |
| `Dockerfile` | Code-location image; also the image every run container starts from |
| `scripts/offline.sh` | Runs a command with no network access (Linux CI only) |
| `scripts/tracer_kphx.sh` | Ingests, builds, and prints the KPHX verification tracer bullet |
| `scripts/record_fixtures.py` | One-off: records IEM fixtures for the Dagster asset tests (KORD, extended KPHX asos) |
| `site/` | The React + Vite + TypeScript site (see "Site" above) |
| `site/scripts/generate-dev-data.mjs` | `npm run generate:data`: runs `wfa export` into `site/public/data/` |
| `site/src/test/fixtures/export/` | Committed export of the tracer fixture DB, drift-guarded against a fresh export |

### Conventions

- **Test markers:** `unit`, `integration`, `io`, `dagster`, `dbt`, `network`. Mark any test that reaches the internet `network`; CI deselects those.
- **Snowflake portability:** models run on DuckDB but must parse as Snowflake SQL. `sqlfluff lint` fails on DuckDB-only syntax, such as list comprehensions `[x * 2 for x in xs]`. Write the portable form.
- **Network-free CI:** after `uv sync`, every CI step runs through `scripts/offline.sh`, a network namespace with no route out. A step that needs the network fails there, not silently later.
