"""Every connector against recorded fixtures (offline, FICTIONAL companies)."""

from datetime import UTC, date, datetime

import pytest

from signalgraph.connectors import (
    AshbyConnector,
    EdgarFormDConnector,
    FixtureTransport,
    GitHubOrgConnector,
    GreenhouseConnector,
    LeverConnector,
    PoliteClient,
    RssConnector,
)
from signalgraph.connectors.edgar import clean_display_name, parse_form_d
from signalgraph.connectors.rss import parse_feed
from signalgraph.domain.models import SignalType, Source
from tests.conftest import FIXTURES


async def test_edgar_form_d(offline_client: PoliteClient) -> None:
    conn = EdgarFormDConnector(offline_client, today=date(2026, 11, 19))
    events = await conn.fetch(None)
    assert conn.key == "edgar:form_d"
    assert len(events) == 5
    by_name = {e.company_name: e for e in events}
    nw = by_name["Northwind Robotics, Inc."]
    assert nw.signal_type is SignalType.FUNDING
    assert nw.subtype == "form_d"
    assert nw.magnitude == 12_000_000
    assert nw.attributes["cik"] == "9900001"
    assert nw.country == "US"
    assert nw.occurred_at == datetime(2026, 11, 12, tzinfo=UTC)
    assert by_name["Zentro, Inc."].attributes["aliases"] == ["Acornfield Labs, Inc."]
    assert by_name["Ember & Oak Ventures Fund I, L.P."].subtype == "form_d_fund"
    assert by_name["Blue Peak Health Inc"].attributes["is_amendment"] is True


async def test_edgar_respects_watermark(offline_client: PoliteClient) -> None:
    conn = EdgarFormDConnector(offline_client, today=date(2026, 11, 19))
    events = await conn.fetch(datetime(2026, 11, 10, tzinfo=UTC))
    assert {e.occurred_at.day for e in events} == {10, 11, 12}


async def test_edgar_survives_missing_detail_doc(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "manifest.yaml").write_text(
        "efts.sec.gov/LATEST/search-index: search.json\n", encoding="utf-8"
    )
    (tmp_path / "search.json").write_bytes((FIXTURES / "edgar/search_form_d.json").read_bytes())

    async def no_sleep(_: float) -> None: ...

    async with PoliteClient("t", transport=FixtureTransport(tmp_path), sleep=no_sleep) as c:
        events = await EdgarFormDConnector(c, today=date(2026, 11, 19)).fetch(None)
    assert len(events) == 5  # falls back to search metadata
    assert all(e.magnitude == 0 for e in events)


def test_edgar_helpers() -> None:
    assert clean_display_name("58EOS, LLC  (CIK 0002164858)") == "58EOS, LLC"
    parsed = parse_form_d((FIXTURES / "edgar/0009900002-26-000001.xml").read_text())
    assert parsed["total_amount_sold"] == 4_500_000
    assert parsed["previous_names"] == ["Acornfield Labs, Inc."]


async def test_greenhouse(offline_client: PoliteClient) -> None:
    conn = GreenhouseConnector(
        offline_client,
        board="northwindrobotics",
        company="Northwind Robotics",
        domain="northwindrobotics.com",
    )
    events = await conn.fetch(None)
    assert len(events) == 6
    families = sorted(e.subtype for e in events)
    assert families == ["eng", "gtm", "gtm", "gtm", "gtm", "ops"]
    assert all(e.source is Source.GREENHOUSE and e.signal_type is SignalType.HIRING for e in events)
    assert events[0].native_id.startswith("northwindrobotics:")


async def test_greenhouse_watermark_is_inclusive(offline_client: PoliteClient) -> None:
    conn = GreenhouseConnector(offline_client, board="northwindrobotics", company="NW")
    events = await conn.fetch(datetime(2026, 11, 13, 14, 0, tzinfo=UTC))
    assert len(events) == 2  # the item AT the watermark is re-delivered; upsert dedupes


async def test_lever(offline_client: PoliteClient) -> None:
    events = await LeverConnector(
        offline_client, board="kestreldatainc", company="Kestrel Data, Inc."
    ).fetch(None)
    assert [e.subtype for e in events] == ["gtm", "gtm"]
    assert events[0].country == "US"
    assert events[0].company_domain == "jobs.lever.co"  # ATS host: canonicaliser drops it later


async def test_ashby_skips_unlisted(offline_client: PoliteClient) -> None:
    events = await AshbyConnector(offline_client, board="zentro", company="Zentro").fetch(None)
    assert len(events) == 2
    assert {e.subtype for e in events} == {"gtm", "data"}
    assert events[0].country == "United States"


async def test_github_aggregates_events(offline_client: PoliteClient) -> None:
    conn = GitHubOrgConnector(
        offline_client,
        org="northwind-robotics",
        company="Northwind Robotics",
        domain="northwindrobotics.com",
    )
    events = await conn.fetch(None)
    kinds = {e.subtype for e in events}
    assert kinds == {"push", "release", "new_public_repo", "pull_request", "stars"}
    assert sum(e.magnitude for e in events if e.subtype == "push") == 6
    assert all(e.signal_type is SignalType.ENG_ACTIVITY for e in events)


@pytest.mark.parametrize(
    ("url", "n", "labels"),
    [
        (
            "https://news.northwindrobotics.com/feed.xml",
            3,
            {"funding", "leadership_change", "expansion"},
        ),
        ("https://hyperionvale.com/blog/atom.xml", 2, {"partnership", "product_launch"}),
    ],
)
async def test_rss_and_atom(
    offline_client: PoliteClient, url: str, n: int, labels: set[str]
) -> None:
    events = await RssConnector(offline_client, url=url, company="X").fetch(None)
    assert len(events) == n
    assert {e.subtype for e in events} == labels
    assert all(e.occurred_at.tzinfo is not None for e in events)


def test_parse_feed_rejects_entity_expansion() -> None:
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]>'
        "<rss><channel><item><title>&b;</title></item></channel></rss>"
    )
    with pytest.raises(Exception, match=r"(?i)entit"):
        parse_feed(bomb)


async def test_unknown_route_is_404_not_network(offline_client: PoliteClient) -> None:
    from signalgraph.connectors.http import HttpError

    with pytest.raises(HttpError) as info:
        await GreenhouseConnector(offline_client, board="does-not-exist", company="x").fetch(None)
    assert info.value.status == 404
