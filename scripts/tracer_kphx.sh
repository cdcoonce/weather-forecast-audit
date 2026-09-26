#!/usr/bin/env bash
# Tracer bullet: ingest ~5 months of KPHX guidance and observations (one run
# either side of the 2026-04-30 NBS cycle changeover), build dbt, and print
# the headline verification queries.
#
# Makes real network requests to IEM: a handful, politely rate-limited by
# UrllibFetcher (default 1s between requests). Requires WFA_DUCKDB_PATH.
set -euo pipefail

if [[ -z "${WFA_DUCKDB_PATH:-}" ]]; then
  echo "tracer_kphx.sh: WFA_DUCKDB_PATH must be set" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run wfa init-db
uv run wfa ingest --station KPHX --start 2023-06-01 --end 2023-08-31
uv run wfa ingest --station KPHX --start 2026-04-15 --end 2026-05-20
uv run dbt build --project-dir "$REPO_ROOT/dbt" --profiles-dir "$REPO_ROOT/dbt"

echo
echo "== KPHX mean error by lead_day x variable =="
uv run python "$REPO_ROOT/scripts/run_sql.py" "$REPO_ROOT/scripts/sql/kphx_mean_error.sql"

echo
echo "== KPHX cycle_hour counts by run-date month (changeover check) =="
uv run python "$REPO_ROOT/scripts/run_sql.py" \
  "$REPO_ROOT/scripts/sql/kphx_cycle_hour_by_month.sql"
