"""UrllibFetcher: politeness (min interval) and retry policy.

No real sleeping, no real sockets: `time.sleep`/`time.monotonic` are injected
as fakes and `urllib.request.urlopen` is monkeypatched.
"""

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

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
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


def test_default_user_agent_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_headers: dict[str, str] = {}

    def fake_urlopen(request: object, timeout: float) -> BinaryIO:
        seen_headers.update(request.headers)  # type: ignore[attr-defined]
        return _FakeHttpResponse(200, b"x")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    fetcher = UrllibFetcher(sleep=lambda _s: None, clock=FakeClock())

    fetcher.get("https://example.test/data")

    assert "weather-forecast-audit" in seen_headers.get("User-agent", "")
