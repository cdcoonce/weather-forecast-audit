"""Polite HTTP fetching for IEM endpoints.

IEM's cgi-bin services are a shared community resource with no published
rate limit; `UrllibFetcher` enforces a minimum interval between requests and
retries transient failures (retryable HTTP statuses, timeouts, and network
errors raised while connecting or reading the body, such as a truncated
response) with exponential backoff, so a multi-month backfill behaves like a
careful human, not a scraper.
"""

import http.client
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


@dataclass
class FetchStats:
    """Measured request statistics for one `UrllibFetcher`.

    Pure diagnostics: nothing reads these to decide behaviour. They exist so a
    slow run can be explained from its materialization metadata (timeouts,
    rate limiting, one slow URL) instead of guessed at.
    """

    attempts: int = 0  # every try, including retries
    retries: int = 0  # attempts after the first for a given `get`
    timeouts: int = 0  # TimeoutError (connect/read, or the body deadline)
    read_errors: int = 0  # HTTPException/OSError other than URLError/TimeoutError
    url_errors: int = 0  # URLError other than timeouts
    throttled_429: int = 0  # attempts that saw HTTP 429
    server_errors_5xx: int = 0  # attempts that saw HTTP 5xx
    seconds_in_attempts: float = 0.0  # summed wall time of attempts
    seconds_backing_off: float = 0.0  # summed retry backoff sleeps
    seconds_throttling: float = 0.0  # summed min-interval politeness sleeps
    slowest_attempt_seconds: float = 0.0
    slowest_attempt_url: str = ""

    def as_metadata(self) -> dict[str, int | float | str]:
        """Flat `http_`-prefixed keys for Dagster materialization metadata."""
        return {
            "http_attempts": self.attempts,
            "http_retries": self.retries,
            "http_timeouts": self.timeouts,
            "http_read_errors": self.read_errors,
            "http_url_errors": self.url_errors,
            "http_throttled_429": self.throttled_429,
            "http_server_errors_5xx": self.server_errors_5xx,
            "http_seconds_in_attempts": round(self.seconds_in_attempts, 1),
            "http_seconds_backing_off": round(self.seconds_backing_off, 1),
            "http_seconds_throttling": round(self.seconds_throttling, 1),
            "http_slowest_attempt_seconds": round(self.slowest_attempt_seconds, 1),
            "http_slowest_attempt_url": self.slowest_attempt_url,
        }


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
        self.stats = FetchStats()

    def _throttle(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            wait = self.min_interval_s - (now - self._last_request_at)
            if wait > 0:
                self.stats.seconds_throttling += wait
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _read_body(self, response: object) -> bytes:
        """Read the whole response body under a total wall-clock deadline.

        `urlopen(..., timeout=self.timeout_s)` only bounds each individual
        socket operation: a server that keeps trickling small chunks never
        triggers that per-call timeout but can still hang the read forever.
        Reading in chunks and checking the injected clock after each one
        catches that case too, and is unit-testable without real sockets.
        """
        chunks: list[bytes] = []
        deadline_start = self._clock()
        while True:
            chunk = response.read(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            if self._clock() - deadline_start > self.timeout_s:
                msg = f"body read exceeded the {self.timeout_s}s total deadline"
                raise TimeoutError(msg)

    def _record_attempt(self, url: str, started_at: float) -> None:
        elapsed = self._clock() - started_at
        stats = self.stats
        stats.seconds_in_attempts += elapsed
        if elapsed > stats.slowest_attempt_seconds:
            stats.slowest_attempt_seconds = elapsed
            stats.slowest_attempt_url = url[:200]

    def get(self, url: str) -> HttpResponse:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats.attempts += 1
            if attempt > 0:
                self.stats.retries += 1
            started_at = self._clock()
            request = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent}
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    body = self._read_body(response)
                    return HttpResponse(status=response.status, body=body)
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status == 429:
                    self.stats.throttled_429 += 1
                elif 500 <= status < 600:
                    self.stats.server_errors_5xx += 1
                error = FetchError(status=status, reason=f"http_error:{status}")
                if status not in RETRYABLE_STATUSES or attempt == self.max_retries:
                    raise error from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if isinstance(exc, TimeoutError):
                    self.stats.timeouts += 1
                else:
                    self.stats.url_errors += 1
                reason = getattr(exc, "reason", exc)
                error = FetchError(status=None, reason=f"http_error:{reason}")
                if attempt == self.max_retries:
                    raise error from exc
            except (http.client.HTTPException, OSError) as exc:
                # Raised while reading the body (IncompleteRead, connection
                # resets, TLS errors) or by a dropped connection
                # (RemoteDisconnected). Named by class, not str(exc): the text
                # carries varying byte counts and the reason keys the
                # ingest-gaps ledger. Programming errors are not caught.
                self.stats.read_errors += 1
                error = FetchError(
                    status=None, reason=f"http_error:{type(exc).__name__}"
                )
                if attempt == self.max_retries:
                    raise error from exc
            finally:
                self._record_attempt(url, started_at)
            backoff = self.backoff_s * (2**attempt)
            self.stats.seconds_backing_off += backoff
            self._sleep(backoff)
        msg = "unreachable: loop always returns or raises"
        raise AssertionError(msg)
