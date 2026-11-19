"""Ingest -> resolve -> score on fixtures; idempotency and incremental behaviour."""

from datetime import UTC, date, datetime
from typing import Any

from signalgraph.connectors import Connector, PoliteClient
from signalgraph.domain.models import SignalEvent, SignalType, Source
from signalgraph.scoring.config import IcpConfig, ScoringConfig
from signalgraph.services import pipeline
from signalgraph.storage import Repository
from tests.conftest import CONFIG, FIXTURES

AS_OF = datetime(2026, 11, 19, tzinfo=UTC)


def _snapshot(repo: Repository) -> dict[str, Any]:
    q = repo.con.execute
    return {
        "signals": q(
            "SELECT * EXCLUDE (last_seen_at, first_run_id) FROM signals ORDER BY event_id"
        ).fetchall(),
        "mentions": q("SELECT * FROM mentions ORDER BY mention_id").fetchall(),
        "accounts": q("SELECT * FROM accounts ORDER BY account_id").fetchall(),
        "members": q("SELECT * FROM account_members ORDER BY mention_id").fetchall(),
        "scores": q(
            "SELECT account_id, score, raw_points, icp_fit FROM scores ORDER BY account_id"
        ).fetchall(),
        "watermarks": q(
            "SELECT connector_key, watermark FROM watermarks ORDER BY connector_key"
        ).fetchall(),
    }


async def _full_run(repo: Repository, client: PoliteClient) -> pipeline.IngestReport:
    wl = pipeline.load_watchlist(FIXTURES / "watchlist.yaml")
    conns = pipeline.build_connectors(wl, client, today=date(2026, 11, 19))
    report = await pipeline.ingest(repo, conns)
    pipeline.resolve(repo)
    pipeline.score(
        repo,
        ScoringConfig.load(CONFIG / "scoring.yaml"),
        IcpConfig.load(CONFIG / "icp.yaml"),
        as_of=AS_OF,
    )
    return report


async def test_running_twice_adds_nothing(repo: Repository, offline_client: PoliteClient) -> None:
    first = await _full_run(repo, offline_client)
    assert first.inserted == 35
    assert not first.errors
    snap1 = _snapshot(repo)

    second = await _full_run(repo, offline_client)
    assert second.inserted == 0
    assert not second.errors
    # second run started from stored watermarks...
    assert all(v["watermark_before"] is not None for v in second.per_connector.values())
    # ...and the database is identical apart from bookkeeping columns
    assert _snapshot(repo) == snap1
    assert repo.count("signals") == 35


async def test_cross_source_resolution(repo: Repository, offline_client: PoliteClient) -> None:
    await _full_run(repo, offline_client)
    rows = {r["name"]: r for r in repo.ranked_accounts(limit=100)}
    # Zentro: Ashby (new name) + Lever board registered under OLD name + EDGAR f/k/a
    zentro = next(r for r in rows.values() if r["domain"] == "zentro.ai")
    assert zentro["member_count"] == 3
    # Northwind: Greenhouse + GitHub + RSS (watchlist) + EDGAR legal name
    nw = next(r for r in rows.values() if r["domain"] == "northwindrobotics.com")
    assert nw["n_signals"] == 6 + 7 + 3 + 1  # 6 jobs, 7 daily GitHub aggregates, 3 news, 1 Form D
    assert nw == max(rows.values(), key=lambda r: r["score"])
    # EDGAR legal name "HALCYON BIO, INC." (no domain) joins the Greenhouse board
    halcyon = next(r for r in rows.values() if r["domain"] == "halcyonbio.com")
    assert halcyon["member_count"] == 2
    # two different Kestrel Data companies stay apart
    kestrel = [r for r in rows.values() if r["name"].startswith("Kestrel")]
    assert {r["domain"] for r in kestrel} == {"kestreldata.in", "kestreldata.com"}
    # the VC fund is scored but fails ICP
    fund = next(r for r in rows.values() if "Fund I" in r["name"])
    assert fund["icp_fit"] is False
    assert fund["score"] == 0


class _Boom(Connector):
    source_name = "boom"

    @property
    def key(self) -> str:
        return "boom:1"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        raise RuntimeError("upstream down")


class _Static(Connector):
    source_name = "static"

    def __init__(self, client: PoliteClient, events: list[SignalEvent]):
        super().__init__(client)
        self.events = events
        self.seen_since: list[datetime | None] = []

    @property
    def key(self) -> str:
        return "static:1"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        self.seen_since.append(since)
        return self.keep(self.events, since)


def _ev(native: str, day: int) -> SignalEvent:
    return SignalEvent(
        source=Source.RSS,
        native_id=native,
        signal_type=SignalType.NEWS,
        subtype="general",
        company_name="Acme",
        occurred_at=datetime(2026, 11, day, tzinfo=UTC),
        title=native,
    )


async def test_failure_isolation_and_watermarks(
    repo: Repository, offline_client: PoliteClient
) -> None:
    static = _Static(offline_client, [_ev("a", 1), _ev("b", 5)])
    report = await pipeline.ingest(repo, [_Boom(offline_client), static])
    assert "RuntimeError" in report.errors["boom:1"]
    assert report.per_connector["static:1"]["inserted"] == 2
    assert repo.get_watermark("static:1") == datetime(2026, 11, 5, tzinfo=UTC)
    assert repo.get_watermark("boom:1") is None

    static.events.append(_ev("c", 9))
    report2 = await pipeline.ingest(repo, [static])
    assert static.seen_since[-1] == datetime(2026, 11, 5, tzinfo=UTC)
    assert report2.per_connector["static:1"] == {
        "fetched": 2,
        "inserted": 1,
        "updated": 1,
        "watermark_before": "2026-11-05T00:00:00+00:00",
    }
    assert repo.get_watermark("static:1") == datetime(2026, 11, 9, tzinfo=UTC)


def test_watermark_never_moves_backwards(repo: Repository) -> None:
    repo.advance_watermark("k", datetime(2026, 11, 10, tzinfo=UTC))
    repo.advance_watermark("k", datetime(2026, 1, 1, tzinfo=UTC))
    assert repo.get_watermark("k") == datetime(2026, 11, 10, tzinfo=UTC)


def test_upsert_same_batch_twice(repo: Repository) -> None:
    events = [_ev("x", 1), _ev("x", 1), _ev("y", 2)]
    assert repo.upsert_signals(events, "r1") == {"inserted": 2, "updated": 0}
    assert repo.upsert_signals(events, "r2") == {"inserted": 0, "updated": 2}
    assert repo.count("signals") == 2
    assert repo.count("mentions") == 1
