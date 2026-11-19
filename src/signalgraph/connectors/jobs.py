"""Public job-board APIs (Greenhouse, Lever, Ashby) -> hiring signals.

All three expose unauthenticated JSON for a company's public board. Each
posting becomes one ``hiring`` signal whose subtype is the role family
(gtm, eng, data, ...) so scoring can weight "hiring 4 AEs" above "hiring an
office manager".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from signalgraph.connectors.base import Connector
from signalgraph.connectors.http import PoliteClient
from signalgraph.domain.models import SignalEvent, SignalType, Source
from signalgraph.domain.taxonomy import classify_role


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return urlsplit(url).hostname


def hiring_event(
    *,
    source: Source,
    native_id: str,
    company: str,
    domain: str | None,
    title: str,
    department: str | None,
    location: str | None,
    country: str | None,
    occurred_at: datetime,
    url: str | None,
    board: str,
) -> SignalEvent:
    family = classify_role(title, department)
    return SignalEvent(
        source=source,
        native_id=native_id,
        signal_type=SignalType.HIRING,
        subtype=family,
        company_name=company,
        company_domain=domain,
        location=location,
        country=country,
        occurred_at=occurred_at,
        title=title,
        url=url,
        magnitude=1.0,
        attributes={"department": department, "board": board, "role_family": family},
    )


class _BoardConnector(Connector):
    def __init__(
        self,
        client: PoliteClient,
        *,
        board: str,
        company: str | None = None,
        domain: str | None = None,
        country: str | None = None,
    ) -> None:
        super().__init__(client)
        self.board = board
        self.company = company
        self.domain = domain
        self.country = country

    @property
    def key(self) -> str:
        return f"{self.source_name}:{self.board}"


class GreenhouseConnector(_BoardConnector):
    source_name = "greenhouse"
    BASE = "https://boards-api.greenhouse.io/v1/boards/{board}"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        base = self.BASE.format(board=self.board)
        company = self.company
        if company is None:
            meta = await self.client.get_json(base)
            company = str(meta.get("name") or self.board)
        payload = await self.client.get_json(f"{base}/jobs")
        events = []
        for job in payload.get("jobs", []):
            ts = job.get("first_published") or job.get("updated_at")
            events.append(
                hiring_event(
                    source=Source.GREENHOUSE,
                    native_id=f"{self.board}:{job['id']}",
                    company=str(job.get("company_name") or company),
                    domain=self.domain or _host(job.get("absolute_url")),
                    title=str(job.get("title", "")),
                    department=None,
                    location=(job.get("location") or {}).get("name"),
                    country=self.country,
                    occurred_at=datetime.fromisoformat(ts),
                    url=job.get("absolute_url"),
                    board=self.board,
                )
            )
        return self.keep(events, since)


class LeverConnector(_BoardConnector):
    source_name = "lever"
    BASE = "https://api.lever.co/v0/postings/{board}"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        payload: list[dict[str, Any]] = await self.client.get_json(
            self.BASE.format(board=self.board), params={"mode": "json"}
        )
        events = []
        for post in payload:
            cats = post.get("categories") or {}
            events.append(
                hiring_event(
                    source=Source.LEVER,
                    native_id=f"{self.board}:{post['id']}",
                    company=self.company or self.board,
                    domain=self.domain or _host(post.get("hostedUrl")),
                    title=str(post.get("text", "")),
                    department=cats.get("department") or cats.get("team"),
                    location=cats.get("location"),
                    country=post.get("country") or self.country,
                    occurred_at=datetime.fromtimestamp(int(post["createdAt"]) / 1000, tz=UTC),
                    url=post.get("hostedUrl"),
                    board=self.board,
                )
            )
        return self.keep(events, since)


class AshbyConnector(_BoardConnector):
    source_name = "ashby"
    BASE = "https://api.ashbyhq.com/posting-api/job-board/{board}"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        payload = await self.client.get_json(
            self.BASE.format(board=self.board), params={"includeCompensation": "false"}
        )
        events = []
        for job in payload.get("jobs", []):
            if job.get("isListed") is False:
                continue
            postal = ((job.get("address") or {}).get("postalAddress")) or {}
            events.append(
                hiring_event(
                    source=Source.ASHBY,
                    native_id=f"{self.board}:{job['id']}",
                    company=self.company or self.board,
                    domain=self.domain or _host(job.get("jobUrl")),
                    title=str(job.get("title", "")),
                    department=job.get("department") or job.get("team"),
                    location=job.get("location"),
                    country=postal.get("addressCountry") or self.country,
                    occurred_at=datetime.fromisoformat(job["publishedAt"]),
                    url=job.get("jobUrl"),
                    board=self.board,
                )
            )
        return self.keep(events, since)
