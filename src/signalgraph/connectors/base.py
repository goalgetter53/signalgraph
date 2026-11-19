"""Connector contract plus a fixture-replay transport for offline runs and tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import httpx
import yaml

from signalgraph.connectors.http import PoliteClient
from signalgraph.domain.models import SignalEvent


class Connector(ABC):
    """A source of ``SignalEvent``s.

    ``key`` identifies the watermark stream (one per connector *instance*, e.g.
    ``greenhouse:acme``). ``fetch`` receives the stored watermark and must return
    events whose ``occurred_at`` is >= that watermark. Returning a few items
    *at* the watermark again is fine: storage upserts are idempotent, and the
    inclusive bound avoids losing items that share the watermark timestamp.
    """

    source_name: ClassVar[str]

    def __init__(self, client: PoliteClient) -> None:
        self.client = client

    @property
    @abstractmethod
    def key(self) -> str: ...

    @abstractmethod
    async def fetch(self, since: datetime | None) -> list[SignalEvent]: ...

    @staticmethod
    def keep(events: list[SignalEvent], since: datetime | None) -> list[SignalEvent]:
        if since is None:
            return events
        return [e for e in events if e.occurred_at >= since]


class FixtureTransport(httpx.AsyncBaseTransport):
    """Serve recorded responses from disk, keyed by ``host + path`` (query ignored).

    ``manifest.yaml`` in the fixture root maps ``host/path`` to a file name.
    Anything not in the manifest returns 404, so a test cannot silently hit
    the network.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        raw = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8")) or {}
        self.routes: dict[str, str] = {str(k): str(v) for k, v in raw.items()}
        self.requests: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        parts = urlsplit(str(request.url))
        key = f"{parts.hostname}{parts.path}"
        self.requests.append(str(request.url))
        name = self.routes.get(key)
        if name is None:
            return httpx.Response(404, text=f"no fixture for {key}", request=request)
        path = self.root / name
        ctype = "application/json" if path.suffix == ".json" else "application/xml"
        return httpx.Response(
            200, content=path.read_bytes(), headers={"Content-Type": ctype}, request=request
        )
