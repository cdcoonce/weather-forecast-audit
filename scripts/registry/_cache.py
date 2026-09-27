"""A URL-keyed, on-disk response cache for the registry-building scripts.

Every IEM (and Natural Earth) request `build_registry.py` and
`probe_archive_start.py` make goes through this wrapper so a rerun is free
and offline: the response body is cached at
`.cache/registry/<sha256(url)>.bin`, alongside a `.url` sibling holding the
literal URL (for auditing which hash is which) and a `.error` sibling when
the request failed (IEM's `/api/1/mos.json` 404s, rather than returning
200-with-empty-data, for a runtime with zero archived stations -- a
meaningful, cacheable answer, not an infrastructure failure). `--offline`
mode raises instead of making a live request on a cache miss, so a
byte-identical rerun proves the cache truly covers everything the first run
touched, including the requests that 404'd.
"""

import hashlib
from pathlib import Path

from weather_forecast_audit.iem.http import Fetcher, FetchError, HttpResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "registry"


class OfflineCacheMissError(Exception):
    """Raised in --offline mode when a URL has no cached response."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(f"--offline but no cached response for: {url}")


class CachingFetcher:
    """Wraps a `Fetcher`, caching every response (success or FetchError)."""

    def __init__(
        self,
        inner: Fetcher,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        offline: bool = False,
    ) -> None:
        self.inner = inner
        self.cache_dir = cache_dir
        self.offline = offline
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.requests_made = 0  # live requests only, not cache hits

    def _paths(self, url: str) -> tuple[Path, Path, Path]:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return (
            self.cache_dir / f"{key}.bin",
            self.cache_dir / f"{key}.url",
            self.cache_dir / f"{key}.error",
        )

    def get(self, url: str) -> HttpResponse:
        body_path, url_path, error_path = self._paths(url)
        if error_path.exists():
            status_text, _, reason = error_path.read_text(
                encoding="utf-8"
            ).partition("\n")
            status = int(status_text) if status_text != "None" else None
            raise FetchError(status=status, reason=reason)
        if body_path.exists():
            return HttpResponse(status=200, body=body_path.read_bytes())
        if self.offline:
            raise OfflineCacheMissError(url)

        try:
            response = self.inner.get(url)
        except FetchError as exc:
            self.requests_made += 1
            error_path.write_text(f"{exc.status}\n{exc.reason}", encoding="utf-8")
            url_path.write_text(url, encoding="utf-8")
            raise
        self.requests_made += 1
        body_path.write_bytes(response.body)
        url_path.write_text(url, encoding="utf-8")
        return response
