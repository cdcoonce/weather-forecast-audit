"""Polite HTTP fetching for IEM endpoints.

IEM's cgi-bin services are a shared community resource with no published
rate limit; `UrllibFetcher` enforces a minimum interval between requests and
retries transient failures with exponential backoff, so a multi-month
backfill behaves like a careful human, not a scraper.
"""

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from weather_forecast_audit import __version__

DEFAULT_USER_AGENT = (
    f"weather-forecast-audit/{__version__} "
    "(https://github.com/cdcoonce/weather-forecast-audit)"
)

# 429 (rate limited) and 5xx (server-side): worth a retry. Other 4xx are the
# caller's fault (bad params, missing resource) and retrying won't help.
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class Fetcher(Protocol):
    def get(self, url: str) -> HttpResponse: ...


class FetchError(Exception):
    """Raised when a fetch fails after exhausting retries (or is not retried)."""

    def __init__(self, status: int | None, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"{reason} (status={status})")


class UrllibFetcher:
    """A `Fetcher` backed by `urllib.request`, with politeness and retries."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        min_interval_s: float = 1.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
        timeout_s: float = 60,
    ) -> None:
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep if sleep is not None else time.sleep
        self._clock = clock if clock is not None else time.monotonic
        self.timeout_s = timeout_s
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def get(self, url: str) -> HttpResponse:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            request = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent}
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    return HttpResponse(status=response.status, body=response.read())
            except urllib.error.HTTPError as exc:
                status = exc.code
                error = FetchError(status=status, reason=f"http_error:{status}")
                if status not in RETRYABLE_STATUSES or attempt == self.max_retries:
                    raise error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                reason = getattr(exc, "reason", exc)
                error = FetchError(status=None, reason=f"http_error:{reason}")
                if attempt == self.max_retries:
                    raise error from exc
            self._sleep(self.backoff_s * (2**attempt))
        msg = "unreachable: loop always returns or raises"
        raise AssertionError(msg)
