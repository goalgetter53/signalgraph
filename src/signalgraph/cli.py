"""``signalgraph ingest | resolve | score | export | serve | runs``."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import click
import httpx

from signalgraph.config import Settings
from signalgraph.connectors import FixtureTransport, PoliteClient
from signalgraph.resolution.engine import DEFAULT_THRESHOLD, EntityResolver
from signalgraph.scoring.config import IcpConfig, ScoringConfig
from signalgraph.services import export as exporter
from signalgraph.services import pipeline
from signalgraph.storage import Repository


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@click.group()
@click.option("--db", type=click.Path(path_type=Path), default=None, help="DuckDB file.")
@click.option("--config-dir", type=click.Path(path_type=Path), default=None)
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def cli(ctx: click.Context, db: Path | None, config_dir: Path | None, verbose: bool) -> None:
    """signalgraph: public buying signals -> resolved, scored account graph."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    ctx.obj = {
        "settings": settings,
        "db": db or settings.db_path,
        "config_dir": config_dir or settings.config_dir,
    }


def _repo(ctx: click.Context) -> Repository:
    return Repository(ctx.obj["db"])


@cli.command()
@click.option("--watchlist", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--fixtures",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Replay recorded responses from this directory (offline).",
)
@click.option("--today", default=None, help="Pin 'today' (YYYY-MM-DD) for EDGAR date windows.")
@click.pass_context
def ingest(
    ctx: click.Context, watchlist: Path | None, fixtures: Path | None, today: str | None
) -> None:
    """Fetch new signals from every configured connector (incremental)."""
    settings: Settings = ctx.obj["settings"]
    wl_path = watchlist or (
        fixtures / "watchlist.yaml" if fixtures else ctx.obj["config_dir"] / "watchlist.yaml"
    )
    wl = pipeline.load_watchlist(wl_path)
    transport: httpx.AsyncBaseTransport | None = FixtureTransport(fixtures) if fixtures else None
    headers = {"Authorization": f"Bearer {settings.github_token}"} if settings.github_token else {}

    async def run() -> pipeline.IngestReport:
        async with PoliteClient(
            settings.user_agent,
            timeout_s=settings.http_timeout_s,
            max_retries=settings.max_retries,
            transport=transport,
            extra_headers=headers,
        ) as client:
            conns = pipeline.build_connectors(
                wl, client, today=date.fromisoformat(today) if today else None
            )
            return await pipeline.ingest(repo, conns)

    repo = _repo(ctx)
    report = asyncio.run(run())
    click.echo(
        json.dumps(
            {
                "run_id": report.run_id,
                "inserted": report.inserted,
                "connectors": report.per_connector,
            },
            indent=2,
            default=str,
        )
    )
    if report.errors and len(report.errors) == len(report.per_connector):
        sys.exit(1)


@cli.command()
@click.option("--threshold", type=float, default=DEFAULT_THRESHOLD, show_default=True)
@click.option(
    "--no-domain-constraint",
    is_flag=True,
    help="Allow clusters with >1 distinct domain (helps rebrands, hurts precision).",
)
@click.pass_context
def resolve(ctx: click.Context, threshold: float, no_domain_constraint: bool) -> None:
    """Entity-resolve all company mentions into canonical accounts."""
    resolver = EntityResolver(threshold=threshold, max_domains=None if no_domain_constraint else 1)
    click.echo(json.dumps(pipeline.resolve(_repo(ctx), resolver), indent=2))


@cli.command()
@click.option("--as-of", default=None, help="Score as of this ISO timestamp (default: now).")
@click.pass_context
def score(ctx: click.Context, as_of: str | None) -> None:
    """Score every account (decayed, weighted signals + ICP filters)."""
    cfg_dir: Path = ctx.obj["config_dir"]
    stats = pipeline.score(
        _repo(ctx),
        ScoringConfig.load(cfg_dir / "scoring.yaml"),
        IcpConfig.load(cfg_dir / "icp.yaml"),
        as_of=_parse_dt(as_of),
    )
    click.echo(json.dumps(stats, indent=2))


@cli.command()
@click.option("--format", "fmt", type=click.Choice(["csv", "clay", "hubspot"]), default="csv")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Default: stdout.")
@click.option("--icp-only", is_flag=True)
@click.option("--limit", type=int, default=500, show_default=True)
@click.pass_context
def export(ctx: click.Context, fmt: str, out: Path | None, icp_only: bool, limit: int) -> None:
    """Export ranked accounts as CSV, Clay webhook rows or HubSpot batch-upsert JSON."""
    repo = _repo(ctx)
    rows = repo.ranked_accounts(icp_only=icp_only, limit=limit)
    top: dict[str, str] = {}
    for row in rows:
        bd = repo.breakdown(row["account_id"])
        if bd and bd.contributions:
            top[row["account_id"]] = bd.contributions[0].title
    payload: Any
    if fmt == "csv":
        text = exporter.to_csv(rows, top)
    else:
        payload = (
            exporter.to_clay_rows(rows, top)
            if fmt == "clay"
            else exporter.to_hubspot_batches(rows, top)
        )
        text = json.dumps(payload, indent=2) + "\n"
    if out:
        out.write_text(text, encoding="utf-8")
        click.echo(f"wrote {len(rows)} accounts to {out}", err=True)
    else:
        click.echo(text, nl=False)


@cli.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Serve the read API (FastAPI + uvicorn)."""
    import uvicorn

    from signalgraph.api.app import create_app

    uvicorn.run(create_app(ctx.obj["db"]), host=host, port=port)


@cli.command()
@click.pass_context
def runs(ctx: click.Context) -> None:
    """Show run history and watermarks."""
    repo = _repo(ctx)
    click.echo(
        json.dumps({"runs": repo.runs(), "watermarks": repo.watermarks()}, indent=2, default=str)
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
