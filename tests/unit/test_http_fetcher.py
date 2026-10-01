"""UrllibFetcher: politeness (min interval) and retry policy.

No real sleeping, no real sockets: `time.sleep`/`time.monotonic` are injected
as fakes and `urllib.request.urlopen` is monkeypatched.
"""

import http.client
import ssl
import urllib.error
from typing import BinaryIO

import pytest

from weather_forecast_audit.iem.http import FetchError, HttpResponse, UrllibFetcher

pytestmark = pytest.mark.unit


class FakeClock:
    """A clock that advances by a fixed step each time it is read."""

    def __init__(self, step: float = 0.0) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class FakeSleep:
    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.now += seconds


class _FakeHttpResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body
        self._sent = False

    def read(self, size: int | None = None) -> bytes:
        # Real sockets return the whole available chunk then b"" at EOF;
        # these test bodies are small enough to fit in one chunk.
        if self._sent:
            return b""
        self._sent = True
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeTricklingResponse:
    """A response whose body arrives as slow chunks, advancing a fake clock.

    Models a server that never blocks on a single `read()` past the socket
    timeout, but takes far longer than `timeout_s` in total -- the hang
    `UrllibFetcher` must catch via a wall-clock deadline across the whole
    read, not a per-`read()`-call timeout.
    """

    def __init__(
        self,
        status: int,
        chunks: list[bytes],
        clock: FakeClock,
        seconds_per_chunk: float,
    ) -> None:
        self.status = status
        self._chunks = list(chunks)
        self._clock = clock
        self._seconds_per_chunk = seconds_per_chunk

    def read(self, size: int | None = None) -> bytes:
        if not self._chunks:
            return b""
        self._clock.now += self._seconds_per_chunk
        return self._chunks.pop(0)

    def __enter__(self) -> "_FakeTricklingResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeFailingReadResponse:
    """A response whose `read()` raises `exc`: the connection dies mid-body."""

    def __init__(self, exc: BaseException) -> None:
        self.status = 200
        self._exc = exc

    def read(self, size: int | None = None) -> bytes:
        raise self._exc

    def __enter__(self) -> "_FakeFailingReadResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def _urlopen_sequence(
    outcomes: list[object],
) -> tuple[object, list[str]]:
    """Build a fake urlopen that pops from `outcomes` (a response, or an exc)."""
    calls: list[str] = []
    remaining = list(outcomes)

    def fake_urlopen(request: object, timeout: float) -> BinaryIO:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    return fake_urlopen, calls


def test_get_returns_response_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_urlopen, calls = _urlopen_sequence([_FakeHttpResponse(200, b"hello")])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fetcher = UrllibFetcher(sleep=lambda _s: None, clock=FakeClock())

    response = fetcher.get("https://example.test/data")

    assert response == HttpResponse(status=200, body=b"hello")
    assert calls == ["https://example.test/data"]


def test_min_interval_enforced_between_requests() -> None:
    clock = FakeClock(step=0.0)
    sleep = FakeSleep(clock)
    fetcher = UrllibFetcher(min_interval_s=1.0, sleep=sleep, clock=clock)

    import urllib.request as urllib_request

    responses = iter([_FakeHttpResponse(200, b"a"), _FakeHttpResponse(200, b"b")])
    urllib_request.urlopen = lambda request, timeout: next(responses)  # type: ignore[method-assign]
    try:
        fetcher.get("https://example.test/1")
        fetcher.get("https://example.test/2")
    finally:
        pass

    # The clock never advances on its own (step=0), so the second request must
    # have slept the full min interval.
    assert sleep.calls and sleep.calls[-1] == pytest.approx(1.0)


def test_retries_503_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    error = urllib.error.HTTPError(
        "https://example.test", 503, "Service Unavailable", {}, None
    )
    fake_urlopen, calls = _urlopen_sequence([error, _FakeHttpResponse(200, b"ok")])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(
        sleep=sleep, clock=FakeClock(), max_retries=3, backoff_s=2.0
    )

    response = fetcher.get("https://example.test/data")

    assert response.body == b"ok"
    assert len(calls) == 2
    assert sleep.calls  # a backoff sleep happened between attempts


def test_503_exhausts_retries_raises_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def always_503(request: object, timeout: float) -> BinaryIO:
        raise urllib.error.HTTPError(
            "https://example.test", 503, "Service Unavailable", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", always_503)
    fetcher = UrllibFetcher(sleep=lambda _s: None, clock=FakeClock(), max_retries=3)

    with pytest.raises(FetchError) as excinfo:
        fetcher.get("https://example.test/data")
    assert excinfo.value.status == 503
    assert "503" in excinfo.value.reason


def test_404_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    calls: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> BinaryIO:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        raise urllib.error.HTTPError("https://example.test", 404, "Not Found", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fetcher = UrllibFetcher(sleep=lambda _s: None, clock=FakeClock(), max_retries=3)

    with pytest.raises(FetchError) as excinfo:
        fetcher.get("https://example.test/data")
    assert excinfo.value.status == 404
    assert len(calls) == 1


def test_get_raises_on_body_that_trickles_past_the_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each read() call advances the clock by 20s but returns promptly (no
    # per-call timeout is ever hit); 4 chunks blow well past a 60s deadline.
    clock = FakeClock()
    chunks = [b"x" * 10, b"x" * 10, b"x" * 10, b"x" * 10]
    response = _FakeTricklingResponse(200, chunks, clock, seconds_per_chunk=20.0)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: response
    )
    fetcher = UrllibFetcher(
        sleep=lambda _s: None, clock=clock, timeout_s=60, max_retries=0
    )

    with pytest.raises(FetchError) as excinfo:
        fetcher.get("https://example.test/slow")
    assert "deadline" in excinfo.value.reason


def test_get_retries_after_a_stalled_body_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    slow = _FakeTricklingResponse(
        200, [b"x" * 10, b"x" * 10, b"x" * 10, b"x" * 10], clock, seconds_per_chunk=20.0
    )
    fast = _FakeHttpResponse(200, b"ok")
    responses = iter([slow, fast])
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: next(responses)
    )
    sleep = FakeSleep(clock)
    fetcher = UrllibFetcher(
        sleep=sleep, clock=clock, timeout_s=60, max_retries=1, backoff_s=0.0
    )

    response = fetcher.get("https://example.test/slow")

    assert response.body == b"ok"


def test_default_user_agent_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: dict[str, str] = {}

    def fake_urlopen(request: object, timeout: float) -> BinaryIO:
        seen_headers.update(request.headers)  # type: ignore[attr-defined]
        return _FakeHttpResponse(200, b"x")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fetcher = UrllibFetcher(sleep=lambda _s: None, clock=FakeClock())

    fetcher.get("https://example.test/data")

    assert "weather-forecast-audit" in seen_headers.get("User-agent", "")


def _assert_retries_once_after_read_error(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    fake_urlopen, calls = _urlopen_sequence(
        [_FakeFailingReadResponse(exc), _FakeHttpResponse(200, b"ok")]
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(
        sleep=sleep, clock=FakeClock(), max_retries=3, backoff_s=2.0, min_interval_s=0
    )

    response = fetcher.get("https://example.test/data")

    assert response == HttpResponse(status=200, body=b"ok")
    assert len(calls) == 2
    assert sleep.calls == [2.0 * 1]


def test_get_retries_after_an_incomplete_read_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_retries_once_after_read_error(
        monkeypatch, http.client.IncompleteRead(b"partial", 10)
    )


def test_get_retries_after_a_connection_reset_while_reading_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_retries_once_after_read_error(
        monkeypatch, ConnectionResetError("reset by peer")
    )


def test_get_retries_after_an_ssl_error_while_reading_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_retries_once_after_read_error(monkeypatch, ssl.SSLError("bad record mac"))


def test_get_retries_after_a_remote_disconnect_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_urlopen, calls = _urlopen_sequence(
        [
            http.client.RemoteDisconnected("closed without response"),
            _FakeHttpResponse(200, b"ok"),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(
        sleep=sleep, clock=FakeClock(), max_retries=3, backoff_s=2.0, min_interval_s=0
    )

    response = fetcher.get("https://example.test/data")

    assert response.body == b"ok"
    assert len(calls) == 2
    assert sleep.calls == [2.0 * 1]


def test_incomplete_read_exhausts_retries_raises_fetch_error_named_by_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = http.client.IncompleteRead(b"x" * 55935, 70000)
    calls: list[str] = []

    def always_incomplete(request: object, timeout: float) -> BinaryIO:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        return _FakeFailingReadResponse(incomplete)  # type: ignore[return-value]

    monkeypatch.setattr("urllib.request.urlopen", always_incomplete)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(
        sleep=sleep, clock=FakeClock(), max_retries=3, backoff_s=2.0, min_interval_s=0
    )

    with pytest.raises(FetchError) as excinfo:
        fetcher.get("https://example.test/data")

    assert excinfo.value.status is None
    # Named by class: the exception text carries a varying byte count, and the
    # reason ends up in the ingest-gaps ledger key.
    assert excinfo.value.reason == "http_error:IncompleteRead"
    assert "55935" not in excinfo.value.reason
    assert excinfo.value.__cause__ is incomplete
    assert len(calls) == 4
    assert sleep.calls == [2.0 * 2**0, 2.0 * 2**1, 2.0 * 2**2]


@pytest.mark.parametrize("exc_type", [ValueError, TypeError, AttributeError, KeyError])
def test_a_programming_error_while_reading_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, exc_type: type[Exception]
) -> None:
    calls: list[str] = []
    bug = exc_type("bug")

    def bad_read(request: object, timeout: float) -> BinaryIO:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        return _FakeFailingReadResponse(bug)  # type: ignore[return-value]

    monkeypatch.setattr("urllib.request.urlopen", bad_read)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(sleep=sleep, clock=FakeClock(), max_retries=3)

    with pytest.raises(exc_type) as excinfo:
        fetcher.get("https://example.test/data")

    assert excinfo.value is bug
    assert len(calls) == 1
    assert sleep.calls == []


def test_url_error_exhausts_retries_with_its_reason_text_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = urllib.error.URLError("connection refused")
    calls: list[str] = []

    def always_refused(request: object, timeout: float) -> BinaryIO:
        calls.append(request.full_url)  # type: ignore[attr-defined]
        raise refused

    monkeypatch.setattr("urllib.request.urlopen", always_refused)
    sleep = FakeSleep()
    fetcher = UrllibFetcher(
        sleep=sleep, clock=FakeClock(), max_retries=3, backoff_s=2.0, min_interval_s=0
    )

    with pytest.raises(FetchError) as excinfo:
        fetcher.get("https://example.test/data")

    assert excinfo.value.status is None
    # The URLError clause keeps its own reason (the wrapped reason text), not
    # the class name the read-phase clause uses.
    assert excinfo.value.reason == "http_error:connection refused"
    assert excinfo.value.__cause__ is refused
    assert len(calls) == 4
    assert sleep.calls == [2.0 * 2**0, 2.0 * 2**1, 2.0 * 2**2]


# --- request statistics (`UrllibFetcher.stats`) -----------------------------
#
# Diagnostics only: they record what `get` did without changing it. Time is
# controlled by advancing the fake clock inside the fake urlopen (step=0 so
# the clock only moves when a test says so).


def _timed_urlopen(clock: FakeClock, outcomes: list[tuple[float, object]]) -> object:
    """A fake urlopen whose each call takes `duration` fake seconds."""
    remaining = list(outcomes)

    def fake_urlopen(request: object, timeout: float) -> BinaryIO:
        duration, outcome = remaining.pop(0)
        clock.now += duration
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]

    return fake_urlopen


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.test", status, "x", {}, None)


def _stats_fetcher(clock: FakeClock, **kwargs: float) -> UrllibFetcher:
    return UrllibFetcher(
        sleep=FakeSleep(), clock=clock, min_interval_s=0, backoff_s=2.0, **kwargs
    )


def test_stats_count_a_clean_request(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(clock, [(3.0, _FakeHttpResponse(200, b"ok"))]),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert (stats.attempts, stats.retries) == (1, 0)
    assert (stats.timeouts, stats.read_errors, stats.url_errors) == (0, 0, 0)
    assert (stats.throttled_429, stats.server_errors_5xx) == (0, 0)
    assert stats.seconds_in_attempts == pytest.approx(3.0)
    assert stats.slowest_attempt_seconds == pytest.approx(3.0)
    assert stats.slowest_attempt_url == "https://example.test/data"


def test_stats_count_a_503_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock, [(1.0, _http_error(503)), (1.0, _FakeHttpResponse(200, b"ok"))]
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert (stats.attempts, stats.retries) == (2, 1)
    assert stats.server_errors_5xx == 1
    assert stats.throttled_429 == 0
    assert stats.seconds_backing_off == pytest.approx(2.0 * 1)


def test_stats_count_a_429(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock, [(0.0, _http_error(429)), (0.0, _FakeHttpResponse(200, b"ok"))]
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    assert fetcher.stats.throttled_429 == 1
    assert fetcher.stats.server_errors_5xx == 0
    assert fetcher.stats.retries == 1


def test_stats_count_a_timeout_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock,
            [(60.0, TimeoutError("timed out")), (1.0, _FakeHttpResponse(200, b"ok"))],
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert stats.timeouts == 1
    assert (stats.read_errors, stats.url_errors) == (0, 0)
    assert stats.retries == 1
    assert stats.slowest_attempt_seconds == pytest.approx(60.0)


def test_stats_count_an_incomplete_read_as_a_read_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock,
            [
                (
                    0.0,
                    _FakeFailingReadResponse(http.client.IncompleteRead(b"p", 10)),
                ),
                (0.0, _FakeHttpResponse(200, b"ok")),
            ],
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert stats.read_errors == 1
    assert (stats.timeouts, stats.url_errors) == (0, 0)
    assert stats.retries == 1


def test_stats_count_a_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock,
            [
                (0.0, urllib.error.URLError("connection refused")),
                (0.0, _FakeHttpResponse(200, b"ok")),
            ],
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert stats.url_errors == 1
    assert (stats.timeouts, stats.read_errors) == (0, 0)


def test_stats_track_the_slowest_attempt_and_its_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    long_url = "https://example.test/" + "x" * 300
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock,
            [
                (1.5, _FakeHttpResponse(200, b"a")),
                (7.25, _FakeHttpResponse(200, b"b")),
                (2.0, _FakeHttpResponse(200, b"c")),
            ],
        ),
    )
    fetcher = _stats_fetcher(clock)

    fetcher.get("https://example.test/first")
    fetcher.get(long_url)
    fetcher.get("https://example.test/third")

    stats = fetcher.stats
    assert stats.attempts == 3
    assert stats.seconds_in_attempts == pytest.approx(1.5 + 7.25 + 2.0)
    assert stats.slowest_attempt_seconds == pytest.approx(7.25)
    assert stats.slowest_attempt_url == long_url[:200]
    assert len(stats.slowest_attempt_url) == 200


def test_stats_separate_throttle_sleep_from_backoff_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(
            clock,
            [(0.0, _FakeHttpResponse(200, b"a")), (0.0, _FakeHttpResponse(200, b"b"))],
        ),
    )
    fetcher = UrllibFetcher(
        sleep=FakeSleep(clock), clock=clock, min_interval_s=1.0, backoff_s=2.0
    )

    fetcher.get("https://example.test/1")
    fetcher.get("https://example.test/2")

    assert fetcher.stats.seconds_throttling == pytest.approx(1.0)
    assert fetcher.stats.seconds_backing_off == 0


def test_stats_exhausted_retries_still_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        _timed_urlopen(clock, [(1.0, _http_error(503)) for _ in range(3)]),
    )
    fetcher = _stats_fetcher(clock, max_retries=2)

    with pytest.raises(FetchError):
        fetcher.get("https://example.test/data")

    stats = fetcher.stats
    assert (stats.attempts, stats.retries) == (3, 2)
    assert stats.server_errors_5xx == 3
    assert stats.seconds_in_attempts == pytest.approx(3.0)
    # Backoff sleeps happen only between attempts: 2.0 * (1 + 2).
    assert stats.seconds_backing_off == pytest.approx(2.0 * 1 + 2.0 * 2)


def test_as_metadata_keys_and_rounding() -> None:
    from weather_forecast_audit.iem.http import FetchStats

    stats = FetchStats(
        attempts=5,
        retries=2,
        timeouts=1,
        read_errors=3,
        url_errors=4,
        throttled_429=6,
        server_errors_5xx=7,
        seconds_in_attempts=12.34,
        seconds_backing_off=6.06,
        seconds_throttling=0.04,
        slowest_attempt_seconds=60.05,
        slowest_attempt_url="https://example.test/x",
    )

    metadata = stats.as_metadata()

    assert sorted(metadata) == [
        "http_attempts",
        "http_read_errors",
        "http_retries",
        "http_seconds_backing_off",
        "http_seconds_in_attempts",
        "http_seconds_throttling",
        "http_server_errors_5xx",
        "http_slowest_attempt_seconds",
        "http_slowest_attempt_url",
        "http_throttled_429",
        "http_timeouts",
        "http_url_errors",
    ]
    assert metadata["http_attempts"] == 5
    assert metadata["http_retries"] == 2
    assert metadata["http_timeouts"] == 1
    assert metadata["http_read_errors"] == 3
    assert metadata["http_url_errors"] == 4
    assert metadata["http_throttled_429"] == 6
    assert metadata["http_server_errors_5xx"] == 7
    assert metadata["http_seconds_in_attempts"] == 12.3
    assert metadata["http_seconds_backing_off"] == 6.1
    assert metadata["http_seconds_throttling"] == 0.0
    assert metadata["http_slowest_attempt_seconds"] == 60.0
    assert metadata["http_slowest_attempt_url"] == "https://example.test/x"
    for key in ("attempts", "retries", "timeouts", "read_errors", "url_errors"):
        assert type(metadata[f"http_{key}"]) is int
    for key in (
        "seconds_in_attempts",
        "seconds_backing_off",
        "seconds_throttling",
        "slowest_attempt_seconds",
    ):
        assert type(metadata[f"http_{key}"]) is float
    assert type(metadata["http_slowest_attempt_url"]) is str
