"""Live smoke tests against real public APIs. Opt-in: SIGNALGRAPH_LIVE=1 pytest tests/live."""

import os

import pytest

from signalgraph.config import Settings
from signalgraph.connectors import EdgarFormDConnector, GreenhouseConnector, PoliteClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("SIGNALGRAPH_LIVE") != "1", reason="set SIGNALGRAPH_LIVE=1"),
]


async def test_live_edgar_form_d() -> None:
    async with PoliteClient(Settings.from_env().user_agent) as client:
        events = await EdgarFormDConnector(client, lookback_days=3, max_filings=3).fetch(None)
    assert events
    assert all(e.company_name for e in events)


async def test_live_greenhouse() -> None:
    board = os.environ.get("SIGNALGRAPH_LIVE_GREENHOUSE_BOARD", "stripe")
    async with PoliteClient(Settings.from_env().user_agent) as client:
        events = await GreenhouseConnector(client, board=board).fetch(None)
    assert events
