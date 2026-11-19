"""Application services: ingest -> resolve -> score. Orchestration only, no SQL, no HTTP details."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from signalgraph.connectors import (
    AshbyConnector,
    Connector,
    EdgarFormDConnector,
    GitHubOrgConnector,
    GreenhouseConnector,
    LeverConnector,
    PoliteClient,
    RssConnector,
)
from signalgraph.resolution.engine import EntityResolver
from signalgraph.scoring.config import IcpConfig, ScoringConfig
from signalgraph.scoring.engine import score_account
from signalgraph.storage import Repository

log = logging.getLogger(__name__)


def new_run_id(kind: str) -> str:
    return f"{kind}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"


def load_watchlist(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: watchlist must be a mapping")
    return data


def build_connectors(
    watchlist: dict[str, Any], client: PoliteClient, *, today: date | None = None
) -> list[Connector]:
    connectors: list[Connector] = []
    edgar = watchlist.get("edgar") or {}
    if edgar.get("enabled"):
        connectors.append(
            EdgarFormDConnector(
                client,
                lookback_days=int(edgar.get("lookback_days", 7)),
                max_filings=int(edgar.get("max_filings", 50)),
                fetch_details=bool(edgar.get("fetch_details", True)),
                today=today,
            )
        )
    for c in watchlist.get("companies") or []:
        common = {"company": c.get("name"), "domain": c.get("domain"), "country": c.get("country")}
        if c.get("greenhouse"):
            connectors.append(GreenhouseConnector(client, board=c["greenhouse"], **common))
        if c.get("lever"):
            connectors.append(LeverConnector(client, board=c["lever"], **common))
        if c.get("ashby"):
            connectors.append(AshbyConnector(client, board=c["ashby"], **common))
        if c.get("github"):
            connectors.append(GitHubOrgConnector(client, org=c["github"], **common))
        for url in c.get("rss") or []:
            connectors.append(
                RssConnector(
                    client,
                    url=url,
                    company=c["name"],
                    domain=c.get("domain"),
                    country=c.get("country"),
                )
            )
    return connectors


@dataclass
class IngestReport:
    run_id: str
    per_connector: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def inserted(self) -> int:
        return sum(int(v.get("inserted", 0)) for v in self.per_connector.values())

    @property
    def errors(self) -> dict[str, str]:
        return {k: v["error"] for k, v in self.per_connector.items() if "error" in v}


async def ingest(
    repo: Repository, connectors: list[Connector], *, concurrency: int = 4
) -> IngestReport:
    """Fetch every connector from its watermark and upsert. One failure != total failure."""
    run_id = new_run_id("ingest")
    repo.start_run(run_id, "ingest")
    report = IngestReport(run_id)
    sem = asyncio.Semaphore(concurrency)

    async def one(conn: Connector) -> None:
        since = repo.get_watermark(conn.key)
        async with sem:
            try:
                events = await conn.fetch(since)
            except Exception as exc:  # isolate connector failures
                log.exception("connector %s failed", conn.key)
                report.per_connector[conn.key] = {"error": f"{type(exc).__name__}: {exc}"}
                return
        stats = repo.upsert_signals(events, run_id)
        if events:
            repo.advance_watermark(conn.key, max(e.occurred_at for e in events))
        report.per_connector[conn.key] = {
            "fetched": len(events),
            **stats,
            "watermark_before": since.isoformat() if since else None,
        }

    await asyncio.gather(*(one(c) for c in connectors))
    repo.finish_run(run_id, {"connectors": report.per_connector, "inserted": report.inserted})
    return report


def resolve(repo: Repository, resolver: EntityResolver | None = None) -> dict[str, Any]:
    run_id = new_run_id("resolve")
    repo.start_run(run_id, "resolve")
    resolver = resolver or EntityResolver()
    mentions = repo.mentions()
    result = resolver.resolve(mentions)
    repo.replace_accounts(result.accounts)
    stats = {
        "mentions": len(mentions),
        "accounts": len(result.accounts),
        "candidate_pairs": result.blocking.candidate_pairs,
        "reduction_ratio": round(result.blocking.reduction_ratio, 4),
        "threshold": resolver.threshold,
        "timings_s": {k: round(v, 4) for k, v in result.timings_s.items()},
    }
    repo.finish_run(run_id, stats)
    return stats


def score(
    repo: Repository, scoring: ScoringConfig, icp: IcpConfig, *, as_of: datetime | None = None
) -> dict[str, Any]:
    run_id = new_run_id("score")
    repo.start_run(run_id, "score")
    as_of = as_of or datetime.now(UTC)
    by_account = repo.signals_by_account()
    breakdowns = [
        score_account(acct, by_account.get(acct.account_id, []), scoring, icp, as_of)
        for acct in repo.accounts()
    ]
    repo.replace_scores(breakdowns)
    stats = {
        "accounts_scored": len(breakdowns),
        "icp_fit": sum(b.icp_fit for b in breakdowns),
        "as_of": as_of.isoformat(),
    }
    repo.finish_run(run_id, stats)
    return stats
