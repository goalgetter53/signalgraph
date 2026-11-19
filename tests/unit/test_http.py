import random
from datetime import UTC, datetime

import httpx
import pytest

from signalgraph.connectors.http import (
    HostRateLimiter,
    HttpError,
    PoliteClient,
    backoff_delay,
    parse_retry_after,
)


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _client(handler, sleeps: Sleeps, **kw) -> PoliteClient:  # type: ignore[no-untyped-def]
    return PoliteClient(
        "ua-test (contact@example.com)",
        transport=httpx.MockTransport(handler),
        sleep=sleeps,
        rng=random.Random(0),
        rate_limiter=HostRateLimiter(0.0),
        **kw,
    )


async def test_retries_429_then_succeeds_and_honours_retry_after() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.headers["User-Agent"].startswith("ua-test")
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": True})

    sleeps = Sleeps()
    async with _client(handler, sleeps) as c:
        assert await c.get_json("https://example.com/x") == {"ok": True}
        assert c.retry_count == 2
    assert calls["n"] == 3
    assert [s for s in sleeps.calls if s > 0]
    assert all(s >= 7 for s in sleeps.calls if s > 0)


async def test_transport_errors_are_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, text="fine")

    async with _client(handler, Sleeps()) as c:
        assert await c.get_text("https://example.com/") == "fine"


async def test_gives_up_after_max_retries() -> None:
    sleeps = Sleeps()
    async with _client(lambda r: httpx.Response(503), sleeps, max_retries=2) as c:
        with pytest.raises(HttpError, match="gave up after 3 attempts"):
            await c.get("https://example.com/")
    assert len(sleeps.calls) == 2


async def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    async with _client(handler, Sleeps()) as c:
        with pytest.raises(HttpError) as info:
            await c.get("https://example.com/missing")
    assert info.value.status == 404
    assert calls["n"] == 1


def test_backoff_is_full_jitter_and_capped() -> None:
    rng = random.Random(1)
    for attempt in range(10):
        d = backoff_delay(attempt, 0.5, 8.0, rng)
        assert 0 <= d <= min(8.0, 0.5 * 2**attempt)


def test_parse_retry_after() -> None:
    now = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("12") == 12
    assert parse_retry_after("Thu, 19 Nov 2026 12:00:30 GMT", now) == pytest.approx(30)
    assert parse_retry_after("garbage") is None
    assert parse_retry_after(None) is None


async def test_rate_limiter_spaces_requests_per_host() -> None:
    limiter = HostRateLimiter(0.0, {"efts.sec.gov": 0.2})
    sleeps = Sleeps()
    for _ in range(3):
        await limiter.acquire("efts.sec.gov", sleeps)
    await limiter.acquire("other.host", sleeps)
    waits = [s for s in sleeps.calls if s > 0]
    assert len(waits) == 2
    assert waits[0] == pytest.approx(0.2, abs=0.05)
    assert waits[1] == pytest.approx(0.4, abs=0.05)
