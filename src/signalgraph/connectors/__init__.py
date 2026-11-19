"""Source connectors. Each one normalises a public API into ``SignalEvent``s."""

from signalgraph.connectors.base import Connector, FixtureTransport
from signalgraph.connectors.edgar import EdgarFormDConnector
from signalgraph.connectors.github import GitHubOrgConnector
from signalgraph.connectors.http import HostRateLimiter, PoliteClient
from signalgraph.connectors.jobs import AshbyConnector, GreenhouseConnector, LeverConnector
from signalgraph.connectors.rss import RssConnector

__all__ = [
    "AshbyConnector",
    "Connector",
    "EdgarFormDConnector",
    "FixtureTransport",
    "GitHubOrgConnector",
    "GreenhouseConnector",
    "HostRateLimiter",
    "LeverConnector",
    "PoliteClient",
    "RssConnector",
]
