"""Core typed domain objects shared by every layer.

Nothing in this module does I/O. Connectors produce ``SignalEvent``s, the
resolution engine turns the company fields of those events into
``CompanyMention``s and clusters them into ``Account``s, and the scoring engine
produces ``ScoreBreakdown``s.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SignalType(StrEnum):
    FUNDING = "funding"
    HIRING = "hiring"
    ENG_ACTIVITY = "eng_activity"
    NEWS = "news"


class Source(StrEnum):
    EDGAR = "edgar"
    GREENHOUSE = "greenhouse"
    LEVER = "lever"
    ASHBY = "ashby"
    GITHUB = "github"
    RSS = "rss"


def stable_id(*parts: str) -> str:
    """Deterministic short id: same inputs always give the same id (idempotency)."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class SignalEvent(BaseModel):
    """One normalised public buying signal about one company."""

    model_config = ConfigDict(frozen=True)

    source: Source
    native_id: str = Field(description="Identifier of the item in the source system")
    signal_type: SignalType
    subtype: str = Field(description="e.g. role family for hiring, 'form_d' for funding")
    company_name: str
    company_domain: str | None = None
    location: str | None = None
    country: str | None = None
    occurred_at: datetime
    title: str
    url: str | None = None
    magnitude: float = Field(default=1.0, ge=0.0, description="Size of the signal, e.g. USD")
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_validator("company_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("company_name must not be empty")
        return value

    @property
    def event_id(self) -> str:
        return stable_id(self.source.value, self.native_id)


class CompanyMention(BaseModel):
    """A single (source-scoped) reference to a company: the unit of entity resolution."""

    model_config = ConfigDict(frozen=True)

    mention_id: str
    source: str
    raw_name: str
    raw_domain: str | None = None
    location: str | None = None
    country: str | None = None
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_event(cls, event: SignalEvent) -> CompanyMention:
        key = (
            event.source.value,
            event.company_name.strip().casefold(),
            (event.company_domain or "").strip().casefold(),
            (event.country or "").strip().casefold(),
        )
        return cls(
            mention_id=stable_id(*key),
            source=event.source.value,
            raw_name=event.company_name,
            raw_domain=event.company_domain,
            location=event.location,
            country=event.country,
            aliases=tuple(str(a) for a in event.attributes.get("aliases") or ()),
        )


class Account(BaseModel):
    """A canonical company produced by entity resolution."""

    account_id: str
    name: str
    domain: str | None
    country: str | None
    member_mention_ids: list[str]
    aliases: list[str] = Field(default_factory=list)


class Contribution(BaseModel):
    event_id: str
    signal_type: SignalType
    subtype: str
    title: str
    occurred_at: datetime
    age_days: float
    base_weight: float
    magnitude_factor: float
    decay: float
    points: float


class ScoreBreakdown(BaseModel):
    account_id: str
    score: float = Field(description="0-100, saturating transform of raw points")
    raw_points: float
    by_type: dict[str, float]
    contributions: list[Contribution]
    icp_fit: bool
    icp_reasons: list[str]
    as_of: datetime
