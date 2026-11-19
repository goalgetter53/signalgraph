"""Read-only FastAPI over the resolved, scored account graph."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from signalgraph import __version__
from signalgraph.domain.models import Account, ScoreBreakdown, SignalEvent
from signalgraph.storage import Repository


class AccountSummary(BaseModel):
    account_id: str
    name: str
    domain: str | None
    country: str | None
    member_count: int
    score: float
    icp_fit: bool
    by_type: dict[str, float]
    icp_reasons: list[str]
    n_signals: int


class SignalOut(BaseModel):
    event_id: str
    source: str
    signal_type: str
    subtype: str
    company_name: str
    occurred_at: datetime
    title: str
    url: str | None
    magnitude: float

    @classmethod
    def of(cls, e: SignalEvent) -> SignalOut:
        return cls(
            event_id=e.event_id,
            source=e.source.value,
            signal_type=e.signal_type.value,
            subtype=e.subtype,
            company_name=e.company_name,
            occurred_at=e.occurred_at,
            title=e.title,
            url=e.url,
            magnitude=e.magnitude,
        )


class AccountDetail(BaseModel):
    account: Account
    score: float | None
    icp_fit: bool | None
    signals: list[SignalOut]


def get_repo(request: Request) -> Iterator[Repository]:
    """One DuckDB cursor per request (connections are not shared across threads)."""
    cur: Repository = request.app.state.root.cursor()
    try:
        yield cur
    finally:
        cur.close()


RepoDep = Annotated[Repository, Depends(get_repo)]


def create_app(db_path: Path | str) -> FastAPI:
    app = FastAPI(
        title="signalgraph",
        version=__version__,
        description="Resolved, scored account graph built from public buying signals.",
    )
    app.state.root = Repository(db_path)

    @app.get("/health")
    def health(r: RepoDep) -> dict[str, Any]:
        return {
            "status": "ok",
            "signals": r.count("signals"),
            "accounts": r.count("accounts"),
            "watermarks": {k: v.isoformat() for k, v in r.watermarks().items()},
        }

    @app.get("/accounts", response_model=list[AccountSummary])
    def accounts(
        r: RepoDep,
        icp_only: bool = False,
        min_score: Annotated[float, Query(ge=0, le=100)] = 0.0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[dict[str, Any]]:
        return r.ranked_accounts(icp_only=icp_only, min_score=min_score, limit=limit, offset=offset)

    @app.get("/accounts/{account_id}", response_model=AccountDetail)
    def account(account_id: str, r: RepoDep) -> AccountDetail:
        acct = r.account(account_id)
        if acct is None:
            raise HTTPException(404, "account not found")
        bd = r.breakdown(account_id)
        return AccountDetail(
            account=acct,
            score=bd.score if bd else None,
            icp_fit=bd.icp_fit if bd else None,
            signals=[SignalOut.of(e) for e in r.signals_for_account(account_id, limit=50)],
        )

    @app.get("/accounts/{account_id}/explain", response_model=ScoreBreakdown)
    def explain(account_id: str, r: RepoDep) -> ScoreBreakdown:
        bd = r.breakdown(account_id)
        if bd is None:
            raise HTTPException(404, "no score for account (run `signalgraph score`)")
        return bd

    @app.get("/signals", response_model=list[SignalOut])
    def signals(
        r: RepoDep,
        signal_type: str | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> list[SignalOut]:
        return [SignalOut.of(e) for e in r.list_signals(signal_type, limit, offset)]

    return app
