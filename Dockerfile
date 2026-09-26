# weather-forecast-audit Dagster gRPC code-location image.
#
# Runs on rammingspeed (Linux x86_64, Intel i5-3230M, NO AVX2) behind the
# shared Dagster OSS webserver/daemon, whose DockerRunLauncher starts every run
# as a fresh container from THIS image. Nothing persists between runs.
#
# Two stages: uv resolves and installs into a self-contained (non-editable)
# .venv in `build`; the runtime stage carries only that .venv, the dbt project,
# and the one system library the wheels need.

FROM ghcr.io/astral-sh/uv:0.12.13 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /opt/dagster/app
# Dependency layer caches independently of source changes. README.md is here
# because hatchling reads it for the package's `readme` field.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra deploy --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra deploy --no-editable

FROM python:3.12-slim
# LightGBM's manylinux wheel links the system OpenMP runtime (libgomp).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/dagster/app
COPY --from=build /opt/dagster/app/.venv ./.venv
ENV PATH="/opt/dagster/app/.venv/bin:${PATH}"

# Bake the dbt manifest at build time. `parse` opens no connection, but the
# profile's env_var() must resolve, so the DuckDB path is a build-only dummy.
COPY dbt ./dbt
RUN WFA_DUCKDB_PATH=/tmp/parse.duckdb dbt parse --project-dir dbt --profiles-dir dbt

# In-container instance dir for the gRPC server process; run and event storage
# live in the platform's Postgres (dagster-postgres, `deploy` extra).
ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p "${DAGSTER_HOME}" /opt/dagster/compute_logs

EXPOSE 4002
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4002", "-m", "weather_forecast_audit.definitions"]
