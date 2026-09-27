"""One-off script: record IEM fixtures needed by the Dagster asset tests (issue #10).

Run once, with network, from the repo root:

    uv run python scripts/record_fixtures.py

Writes to `tests/fixtures/iem/`, reusing the exact URL-building helpers the
production clients use (`iem.guidance._guidance_url`, `iem.observations._asos_url`)
so a recorded fixture is byte-for-byte what the real client would request.
Uses `UrllibFetcher` directly, which throttles to one request per second, so
the whole run makes about 4 polite requests.

Recorded fixtures (build spec #10, "second station" and "KPHX ASOS on 07-18"):
- `nbs_kord_2023-07-14.csv`: KORD NBS run for the 2023-07-14 run window.
- `asos_kord_2023-07-13_2023-07-18.csv`: KORD hourly obs, 2023-07-13..07-18.
- `asos_kphx_2023-07-13_2023-07-18.csv`: KPHX hourly obs extended one day past
  the existing tracer fixture (which stops 07-17), so the asos partition
  covering 07-18 (needed by resolved_windows' D+4 upstream window for the
  2023-07-14 run) is not a manufactured gap. Left alongside, not replacing,
  `asos_kphx_2023-07-13_2023-07-17.csv`, which the tracer integration test
  still reads unchanged.
- `cli_kord_2023.json`: KORD CLI daily reports, clipped to July 2023 (the
  client fetches a full year and filters locally, so a clipped payload for
  the year is a faithful stand-in as long as it covers the needed window).
"""

import json
from datetime import date
from pathlib import Path

from weather_forecast_audit.iem.guidance import _guidance_url
from weather_forecast_audit.iem.http import UrllibFetcher
from weather_forecast_audit.iem.observations import CLI_URL, _asos_url
from weather_forecast_audit.registry import load_registry

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "iem"


def _write(name: str, body: bytes) -> None:
    path = FIXTURES / name
    path.write_bytes(body)
    print(f"wrote {path.relative_to(FIXTURES.parents[2])} ({len(body)} bytes)")


def _clip_cli_json(body: bytes, start: date, end: date) -> bytes:
    payload = json.loads(body)
    kept = [
        record
        for record in payload.get("results", [])
        if start <= date.fromisoformat(record["valid"]) <= end
    ]
    return json.dumps({"results": kept}).encode("utf-8")


def main() -> None:
    fetcher = UrllibFetcher()
    registry = load_registry()
    kord = registry["KORD"]

    nbs_url = _guidance_url(kord.nbm_station_id, date(2023, 7, 14), date(2023, 7, 14))
    _write("nbs_kord_2023-07-14.csv", fetcher.get(nbs_url).body)

    asos_kord_url = _asos_url(kord.icao, date(2023, 7, 13), date(2023, 7, 18))
    _write("asos_kord_2023-07-13_2023-07-18.csv", fetcher.get(asos_kord_url).body)

    asos_kphx_url = _asos_url("KPHX", date(2023, 7, 13), date(2023, 7, 18))
    _write("asos_kphx_2023-07-13_2023-07-18.csv", fetcher.get(asos_kphx_url).body)

    cli_url = f"{CLI_URL}?station={kord.cli_station}&year=2023"
    cli_body = fetcher.get(cli_url).body
    clipped = _clip_cli_json(cli_body, date(2023, 7, 1), date(2023, 7, 31))
    _write("cli_kord_2023.json", clipped)


if __name__ == "__main__":
    main()
