"""Explainable account scoring: weighted signals with exponential time decay.

For every signal ``s`` on an account::

    points(s) = base_weight[type] * subtype_weight[type][subtype]
                * magnitude_factor(s) * amendment_multiplier
                * 0.5 ** (age_days(s) / half_life_days[type])

Points are summed per signal type, optionally capped per type (so 40 open
engineering roles cannot drown out a funding round), then summed and mapped
to 0-100 with a saturating transform ``100 * (1 - exp(-raw / saturation))``.
Every term is returned in the breakdown, so "why is this account 87?" has a
line-by-line answer.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

from signalgraph.domain.models import Account, Contribution, ScoreBreakdown, SignalEvent
from signalgraph.scoring.config import IcpConfig, Magnitude, ScoringConfig


def decay(age_days: float, half_life_days: float) -> float:
    return float(0.5 ** (max(age_days, 0.0) / half_life_days))


def magnitude_factor(value: float, spec: Magnitude) -> float:
    if spec.kind == "none":
        return 1.0
    if spec.kind == "log10":
        raw = 1.0 + math.log10(value / spec.reference) if value > 0 else spec.floor
    else:
        raw = math.log1p(max(value, 0.0))
    return min(max(raw, spec.floor), spec.cap)


def contribution(event: SignalEvent, cfg: ScoringConfig, as_of: datetime) -> Contribution | None:
    rule = cfg.signals.get(event.signal_type.value)
    if rule is None:
        return None
    age = max((as_of - event.occurred_at).total_seconds() / 86400.0, 0.0)
    if rule.max_age_days is not None and age > rule.max_age_days:
        return None
    sub_w = rule.subtypes.get(event.subtype, rule.default_subtype_weight)
    mag = magnitude_factor(event.magnitude, rule.magnitude)
    if event.attributes.get("is_amendment"):
        mag *= rule.amendment_multiplier
    d = decay(age, rule.half_life_days)
    base = rule.base_weight * sub_w
    return Contribution(
        event_id=event.event_id,
        signal_type=event.signal_type,
        subtype=event.subtype,
        title=event.title,
        occurred_at=event.occurred_at,
        age_days=round(age, 2),
        base_weight=base,
        magnitude_factor=round(mag, 4),
        decay=round(d, 4),
        points=base * mag * d,
    )


def icp_check(
    account: Account,
    events: Sequence[SignalEvent],
    score: float,
    icp: IcpConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if icp.countries:
        if account.country is None:
            if not icp.allow_unknown_country:
                reasons.append("country unknown")
        elif account.country not in icp.countries:
            reasons.append(f"country {account.country} not in ICP")
    if account.domain and account.domain in icp.exclude_domains:
        reasons.append(f"domain {account.domain} excluded")
    for pattern in icp.exclude_name_patterns:
        if re.search(pattern, account.name, re.IGNORECASE):
            reasons.append(f"name matches excluded pattern {pattern!r}")
    usable = [e for e in events if e.subtype not in icp.exclude_subtypes]
    if len(usable) < icp.min_signals:
        reasons.append(f"{len(usable)} qualifying signals < min_signals={icp.min_signals}")
    if icp.require_any_signal_types and not any(
        e.signal_type.value in icp.require_any_signal_types for e in usable
    ):
        reasons.append(f"no signal of type {icp.require_any_signal_types}")
    if score < icp.min_score:
        reasons.append(f"score {score:.1f} < min_score {icp.min_score}")
    return (not reasons), reasons


def score_account(
    account: Account,
    events: Sequence[SignalEvent],
    cfg: ScoringConfig,
    icp: IcpConfig,
    as_of: datetime,
) -> ScoreBreakdown:
    contributions = [c for e in events if (c := contribution(e, cfg, as_of)) is not None]
    by_type: dict[str, float] = defaultdict(float)
    for c in contributions:
        by_type[c.signal_type.value] += c.points
    for sig_type, total in list(by_type.items()):
        cap = cfg.signals[sig_type].max_points
        if cap is not None and total > cap:
            by_type[sig_type] = cap
    raw = sum(by_type.values())
    score = 100.0 * (1.0 - math.exp(-raw / cfg.saturation_points))
    fit, reasons = icp_check(account, events, score, icp)
    contributions.sort(key=lambda c: c.points, reverse=True)
    return ScoreBreakdown(
        account_id=account.account_id,
        score=round(score, 2),
        raw_points=round(raw, 4),
        by_type={k: round(v, 4) for k, v in sorted(by_type.items())},
        contributions=contributions,
        icp_fit=fit,
        icp_reasons=reasons,
        as_of=as_of,
    )
