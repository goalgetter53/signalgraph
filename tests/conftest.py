from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from signalgraph.connectors import FixtureTransport, PoliteClient
from signalgraph.storage import Repository

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
CONFIG = ROOT / "config"
GOLD = ROOT / "data" / "gold" / "companies_gold.csv"


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fixture_transport() -> FixtureTransport:
    return FixtureTransport(FIXTURES)


@pytest.fixture
async def offline_client(fixture_transport: FixtureTransport) -> AsyncIterator[PoliteClient]:
    async with PoliteClient(
        "signalgraph-test", transport=fixture_transport, sleep=_no_sleep
    ) as client:
        yield client


@pytest.fixture
def repo(tmp_path: Path) -> Iterator[Repository]:
    r = Repository(tmp_path / "test.duckdb")
    yield r
    r.close()
