"""Polite async HTTP: per-host rate limiting, retries with full jitter, Retry-After.

Every connector goes through ``PoliteClient`` so the politeness rules live in
exactly one place:

* a per-host minimum interval (SEC asks for <= 10 req/s; we default lower),
* retries on transport errors, 429 and 5xx with exponential backoff and
  *full jitter* (``sleep = uniform(0, min(cap, base * 2**attempt))``),
* ``Retry-After`` honoured when the server sends it,
* a descriptive User-Agent with a contact address.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

Sleeper = Callable[[float], Awaitable[None]]


class HostRateLimiter:
    """Enforce a minimum interval between requests to the same host."""

    def __init__(self, default_interval_s: float, per_host: Mapping[str, float] | None = None):
        self._default = default_interval_s
        self._per_host = dict(per_host or {})
        self._next_slot: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def interval_for(self, host: str) -> float:
        return self._per_host.get(host, self._default)

    async def acquire(self, host: str, sleep: Sleeper = asyncio.sleep) -> None:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot.get(host, now))
            self._next_slot[host] = slot + self.interval_for(host)
        delay = slot - now
        if delay > 0:
            await sleep(delay)


def backoff_delay(attempt: int, base_s: float, cap_s: float, rng: random.Random) -> float:
    """Full-jitter exponential backoff (AWS architecture blog variant)."""
    return rng.uniform(0.0, min(cap_s, base_s * (2**attempt)))


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    now = now or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - now).total_seconds())


class HttpError(RuntimeError):
    def __init__(self, url: str, status: int | None, message: str):
        super().__init__(f"{url}: {status} {message}")
        self.url = url
        self.status = status


class PoliteClient:
    def __init__(
        self,
        user_agent: str,
        *,
        timeout_s: float = 20.0,
        max_retries: int = 4,
        base_backoff_s: float = 0.5,
        max_backoff_s: float = 30.0,
        rate_limiter: HostRateLimiter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleeper = asyncio.sleep,
        rng: random.Random | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        headers.update(extra_headers or {})
        self._client = httpx.AsyncClient(
            headers=headers, timeout=timeout_s, transport=transport, follow_redirects=True
        )
        self._max_retries = max_retries
        self._base = base_backoff_s
        self._cap = max_backoff_s
        self._limiter = rate_limiter or HostRateLimiter(
            0.25, {"efts.sec.gov": 0.2, "www.sec.gov": 0.2, "data.sec.gov": 0.2}
        )
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.request_count = 0
        self.retry_count = 0

    async def __aenter__(self) -> PoliteClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str, *, params: Mapping[str, Any] | None = None) -> httpx.Response:
        host = urlsplit(url).hostname or ""
        last_error: str = "unknown"
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire(host, self._sleep)
            self.request_count += 1
            try:
                response = await self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc!r}"
                status: int | None = None
                retry_after: float | None = None
            else:
                if response.status_code < 400:
                    return response
                status = response.status_code
                last_error = response.reason_phrase
                if status not in RETRYABLE_STATUS:
                    raise HttpError(url, status, last_error)
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
            if attempt == self._max_retries:
                raise HttpError(url, status, f"gave up after {attempt + 1} attempts: {last_error}")
            delay = backoff_delay(attempt, self._base, self._cap, self._rng)
            if retry_after is not None:
                delay = max(delay, min(retry_after, self._cap))
            self.retry_count += 1
            log.warning("retrying %s in %.2fs (%s)", url, delay, last_error)
            await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def get_json(self, url: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return (await self.get(url, params=params)).json()

    async def get_text(self, url: str, *, params: Mapping[str, Any] | None = None) -> str:
        return (await self.get(url, params=params)).text
