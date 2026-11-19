"""End-to-end entity resolution: mentions -> records -> blocks -> scores -> clusters -> accounts."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from signalgraph.domain.models import Account, CompanyMention, stable_id
from signalgraph.resolution.blocking import BlockingStats, candidate_pairs
from signalgraph.resolution.cluster import cluster
from signalgraph.resolution.features import FEATURE_NAMES, Record, TfIdf, pair_features
from signalgraph.resolution.scorers import PairScorer, load_default_model

DEFAULT_THRESHOLD = 0.4  # selected on synthetic validation, see scripts/eval_er.py


@dataclass
class ScoredPairs:
    pairs: list[tuple[int, int]]
    features: np.ndarray
    probs: np.ndarray


@dataclass
class ResolutionResult:
    accounts: list[Account]
    mention_to_account: dict[str, str]
    blocking: BlockingStats
    labels: list[int]
    timings_s: dict[str, float] = field(default_factory=dict)


def build_records(mentions: Sequence[CompanyMention]) -> list[Record]:
    return [
        Record.build(i, m.mention_id, m.raw_name, m.raw_domain, m.country, m.aliases)
        for i, m in enumerate(mentions)
    ]


def score_pairs(
    records: Sequence[Record], pairs: set[tuple[int, int]], scorer: PairScorer
) -> ScoredPairs:
    tfidf = TfIdf(r.name.tokens for r in records)
    ordered = sorted(pairs)
    if not ordered:
        empty = np.zeros((0, len(FEATURE_NAMES)))
        return ScoredPairs([], empty, np.zeros(0))
    x = np.array([pair_features(records[i], records[j], tfidf) for i, j in ordered])
    return ScoredPairs(ordered, x, scorer.predict_proba(x))


def _account_for(members: list[Record], taken: set[str]) -> Account:
    name_votes = Counter(r.raw_name.strip() for r in members)
    # most frequent raw name; prefer non-SHOUTING names, then shorter, then alphabetical
    name = min(name_votes, key=lambda n: (-name_votes[n], n.isupper(), len(n), n))
    domains = Counter(r.domain for r in members if r.domain)
    countries = Counter(r.country for r in members if r.country)
    domain = min(domains, key=lambda d: (-domains[d], d)) if domains else None
    country = min(countries, key=lambda c: (-countries[c], c)) if countries else None
    member_ids = sorted(r.key for r in members)
    # Domain-keyed ids stay stable as clusters grow across runs. If two clusters
    # share a domain (the model kept them apart), the later one is keyed by member.
    account_id = stable_id("acct", domain) if domain else stable_id("acct", member_ids[0])
    if account_id in taken:
        account_id = stable_id("acct", domain or "", member_ids[0])
    taken.add(account_id)
    aliases = sorted({a for r in members for a in r.name.all if a} - {""})
    return Account(
        account_id=account_id,
        name=name,
        domain=domain,
        country=country,
        member_mention_ids=member_ids,
        aliases=aliases,
    )


class EntityResolver:
    def __init__(
        self,
        scorer: PairScorer | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_domains: int | None = 1,
        max_block: int = 150,
    ) -> None:
        self.scorer = scorer or load_default_model()
        self.threshold = threshold
        self.max_domains = max_domains
        self.max_block = max_block

    def resolve(self, mentions: Sequence[CompanyMention]) -> ResolutionResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        records = build_records(mentions)
        timings["normalise"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        pairs, stats = candidate_pairs(records, max_block=self.max_block)
        timings["blocking"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        scored = score_pairs(records, pairs, self.scorer)
        timings["scoring"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        edges = [(i, j, float(p)) for (i, j), p in zip(scored.pairs, scored.probs, strict=True)]
        labels = cluster(
            len(records),
            edges,
            threshold=self.threshold,
            domains=[r.domain for r in records],
            max_domains=self.max_domains,
        )
        groups: dict[int, list[Record]] = {}
        for rec, label in zip(records, labels, strict=True):
            groups.setdefault(label, []).append(rec)
        taken: set[str] = set()
        ordered_groups = sorted(groups.values(), key=lambda ms: (-len(ms), min(r.key for r in ms)))
        accounts = [_account_for(members, taken) for members in ordered_groups]
        mention_to_account = {
            mid: acct.account_id for acct in accounts for mid in acct.member_mention_ids
        }
        timings["clustering"] = time.perf_counter() - t3
        timings["total"] = time.perf_counter() - t0
        return ResolutionResult(accounts, mention_to_account, stats, labels, timings)
