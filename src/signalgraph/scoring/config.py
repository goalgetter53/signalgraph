"""Typed scoring + ICP configuration loaded from YAML."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class Magnitude(BaseModel):
    """How a signal's size maps to a multiplier.

    * ``none``  -> 1.0
    * ``log10`` -> clip(1 + log10(magnitude / reference), floor, cap)   (e.g. USD raised)
    * ``log1p`` -> clip(log1p(magnitude), floor, cap)                    (e.g. event counts)
    """

    kind: Literal["none", "log10", "log1p"] = "none"
    reference: float = Field(default=1.0, gt=0)
    floor: float = Field(default=0.25, ge=0)
    cap: float = Field(default=3.0, gt=0)


class SignalRule(BaseModel):
    half_life_days: float = Field(gt=0)
    base_weight: float = Field(ge=0)
    subtypes: dict[str, float] = Field(default_factory=dict)
    default_subtype_weight: float = Field(default=1.0, ge=0)
    magnitude: Magnitude = Field(default_factory=Magnitude)
    max_points: float | None = Field(default=None, gt=0)
    amendment_multiplier: float = Field(default=1.0, ge=0)
    max_age_days: float | None = Field(default=None, gt=0)


class ScoringConfig(BaseModel):
    saturation_points: float = Field(default=40.0, gt=0)
    signals: dict[str, SignalRule]

    @classmethod
    def load(cls, path: Path) -> ScoringConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class IcpConfig(BaseModel):
    countries: list[str] = Field(default_factory=list, description="ISO2; empty = any")
    allow_unknown_country: bool = True
    exclude_subtypes: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    exclude_name_patterns: list[str] = Field(default_factory=list)
    require_any_signal_types: list[str] = Field(default_factory=list)
    min_signals: int = Field(default=1, ge=0)
    min_score: float = Field(default=0.0, ge=0, le=100)

    @field_validator("exclude_name_patterns")
    @classmethod
    def _valid_regex(cls, patterns: list[str]) -> list[str]:
        for p in patterns:
            re.compile(p)
        return patterns

    @field_validator("countries")
    @classmethod
    def _upper(cls, values: list[str]) -> list[str]:
        return [v.upper() for v in values]

    @classmethod
    def load(cls, path: Path) -> IcpConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
