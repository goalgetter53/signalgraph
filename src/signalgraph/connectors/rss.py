"""RSS 2.0 / Atom news feeds -> news signals.

Feeds are configured per company (newsroom, blog, press-release feed), so the
subject company is known; the headline is classified into funding /
leadership_change / expansion / product_launch / ... by transparent rules.
Parsing uses ``defusedxml`` because feeds are untrusted input.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from defusedxml import ElementTree

from signalgraph.connectors.base import Connector
from signalgraph.connectors.http import PoliteClient
from signalgraph.domain.models import SignalEvent, SignalType, Source, stable_id
from signalgraph.domain.taxonomy import classify_news

ATOM = "{http://www.w3.org/2005/Atom}"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None


def parse_feed(xml_text: str) -> list[dict[str, Any]]:
    """Return ``[{id, title, link, published}]`` for RSS 2.0 or Atom documents."""
    root = ElementTree.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    if root.tag == f"{ATOM}feed":
        for entry in root.findall(f"{ATOM}entry"):
            link_el = entry.find(f"{ATOM}link")
            items.append(
                {
                    "id": (entry.findtext(f"{ATOM}id") or "").strip(),
                    "title": (entry.findtext(f"{ATOM}title") or "").strip(),
                    "link": link_el.get("href") if link_el is not None else None,
                    "published": _parse_date(
                        entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated")
                    ),
                }
            )
    else:
        for item in root.iter("item"):
            items.append(
                {
                    "id": (item.findtext("guid") or item.findtext("link") or "").strip(),
                    "title": (item.findtext("title") or "").strip(),
                    "link": (item.findtext("link") or "").strip() or None,
                    "published": _parse_date(item.findtext("pubDate")),
                }
            )
    return items


class RssConnector(Connector):
    source_name = "rss"

    def __init__(
        self,
        client: PoliteClient,
        *,
        url: str,
        company: str,
        domain: str | None = None,
        country: str | None = None,
    ) -> None:
        super().__init__(client)
        self.url = url
        self.company = company
        self.domain = domain
        self.country = country

    @property
    def key(self) -> str:
        return f"rss:{stable_id(self.url)}"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        text = await self.client.get_text(self.url)
        events = []
        for item in parse_feed(text):
            if not item["title"] or item["published"] is None:
                continue
            label = classify_news(item["title"])
            events.append(
                SignalEvent(
                    source=Source.RSS,
                    native_id=item["id"] or stable_id(self.url, item["title"]),
                    signal_type=SignalType.NEWS,
                    subtype=label,
                    company_name=self.company,
                    company_domain=self.domain,
                    country=self.country,
                    occurred_at=item["published"].astimezone(UTC),
                    title=item["title"],
                    url=item["link"],
                    attributes={"feed": self.url},
                )
            )
        return self.keep(events, since)
