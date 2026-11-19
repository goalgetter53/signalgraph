"""GitHub organisation public events -> engineering-activity signals.

Raw events are noisy (one push = one event), so we aggregate per org, per
repo, per UTC day and per event kind into a single signal whose magnitude is
the event count. A new public repo or a release is a stronger signal than
pushes, hence distinct subtypes.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from signalgraph.connectors.base import Connector
from signalgraph.connectors.http import PoliteClient
from signalgraph.domain.models import SignalEvent, SignalType, Source

_KIND = {
    "PushEvent": "push",
    "PullRequestEvent": "pull_request",
    "ReleaseEvent": "release",
    "CreateEvent": "create",
    "PublicEvent": "new_public_repo",
    "IssuesEvent": "issues",
    "WatchEvent": "stars",
    "ForkEvent": "forks",
}


class GitHubOrgConnector(Connector):
    source_name = "github"
    EVENTS = "https://api.github.com/orgs/{org}/events"
    ORG = "https://api.github.com/orgs/{org}"

    def __init__(
        self,
        client: PoliteClient,
        *,
        org: str,
        company: str | None = None,
        domain: str | None = None,
        country: str | None = None,
    ) -> None:
        super().__init__(client)
        self.org = org
        self.company = company
        self.domain = domain
        self.country = country

    @property
    def key(self) -> str:
        return f"github:{self.org}"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        company, domain, location = self.company, self.domain, None
        if company is None or domain is None:
            meta: dict[str, Any] = await self.client.get_json(self.ORG.format(org=self.org))
            company = company or meta.get("name") or self.org
            domain = domain or meta.get("blog") or None
            location = meta.get("location")
        raw: list[dict[str, Any]] = await self.client.get_json(
            self.EVENTS.format(org=self.org), params={"per_page": 100}
        )
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for ev in raw:
            kind = _KIND.get(str(ev.get("type")))
            if kind is None:
                continue
            day = datetime.fromisoformat(ev["created_at"]).astimezone(UTC).date().isoformat()
            repo = (ev.get("repo") or {}).get("name", self.org)
            buckets[(repo, day, kind)].append(ev)
        events = []
        for (repo, day, kind), items in sorted(buckets.items()):
            latest = max(datetime.fromisoformat(i["created_at"]) for i in items)
            events.append(
                SignalEvent(
                    source=Source.GITHUB,
                    native_id=f"{repo}:{day}:{kind}",
                    signal_type=SignalType.ENG_ACTIVITY,
                    subtype=kind,
                    company_name=str(company),
                    company_domain=domain,
                    location=location,
                    country=self.country,
                    occurred_at=latest,
                    title=f"{len(items)} {kind} event(s) in {repo}",
                    url=f"https://github.com/{repo}",
                    magnitude=float(len(items)),
                    attributes={"repo": repo, "day": day, "event_ids": [i["id"] for i in items]},
                )
            )
        return self.keep(events, since)
