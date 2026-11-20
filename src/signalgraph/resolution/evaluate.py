"""Training and evaluation harness for the ER engine.

Metrics are *pairwise* over the full record set: every true co-referent pair
missed by blocking counts as a false negative, so blocking recall is part of
the reported recall, not hidden behind it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from signalgraph.resolution.blocking import candidate_pairs
from signalgraph.resolution.cluster import cluster, clusters_to_pairs
from signalgraph.resolution.engine import DEFAULT_VETO, score_pairs
from signalgraph.resolution.features import Record
from signalgraph.resolution.scorers import FellegiSunterMatcher, LogisticMatcher, PairScorer
from signalgraph.resolution.synth import LabeledRecord


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf(predicted: set[tuple[int, int]], truth: set[tuple[int, int]]) -> PRF:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return PRF(round(p, 4), round(r, 4), round(f, 4), tp, fp, fn)


def load_gold(path: Path) -> list[LabeledRecord]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [
        LabeledRecord(
            record_id=row["record_id"],
            entity_id=row["entity_id"],
            name=row["name"],
            domain=row["domain"] or None,
            country=row["country"] or None,
            aliases=tuple(a for a in row.get("aliases", "").split("|") if a),
        )
        for row in rows
    ]


def to_records(labeled: Sequence[LabeledRecord]) -> list[Record]:
    return [
        Record.build(i, r.record_id, r.name, r.domain, r.country, r.aliases)
        for i, r in enumerate(labeled)
    ]


def true_pairs(labeled: Sequence[LabeledRecord]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(labeled):
        groups.setdefault(r.entity_id, []).append(i)
    return {(a, b) for m in groups.values() for i, a in enumerate(m) for b in m[i + 1 :]}


def training_matrix(
    labeled: Sequence[LabeledRecord],
) -> tuple[np.ndarray, np.ndarray]:
    records = to_records(labeled)
    pairs, _ = candidate_pairs(records)
    dummy = LogisticMatcher()
    scored = score_pairs(records, pairs, dummy)
    y = np.array([int(labeled[i].entity_id == labeled[j].entity_id) for i, j in scored.pairs])
    return scored.features, y


def fit_models(labeled: Sequence[LabeledRecord]) -> tuple[LogisticMatcher, FellegiSunterMatcher]:
    x, y = training_matrix(labeled)
    return LogisticMatcher(l2=1.0).fit(x, y), FellegiSunterMatcher().fit(x, y)


def baseline_pairs(records: Sequence[Record]) -> set[tuple[int, int]]:
    """Naive rule most spreadsheets use: exact normalised name OR same domain."""
    by_key: dict[str, list[int]] = {}
    for r in records:
        by_key.setdefault(f"n:{r.name.primary}", []).append(r.idx)
        if r.domain:
            by_key.setdefault(f"d:{r.domain}", []).append(r.idx)
    edges = [(m[0], j, 1.0) for m in by_key.values() for j in m[1:]]
    return clusters_to_pairs(cluster(len(records), edges, threshold=0.5))


DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.05 * step, 2) for step in range(1, 20))


def _best_row(sweep: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Highest F1; ties go to the threshold closest to 0.5."""
    return max(sweep, key=lambda row: (row[key]["f1"], -abs(row["threshold"] - 0.5)))


def evaluate(
    labeled: Sequence[LabeledRecord],
    scorers: dict[str, PairScorer],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_domains_options: Sequence[int | None] = (None, 1),
    veto_below: float | None = DEFAULT_VETO,
) -> dict[str, Any]:
    records = to_records(labeled)
    truth = true_pairs(labeled)
    pairs, stats = candidate_pairs(records)
    thresholds = list(thresholds)
    report: dict[str, Any] = {
        "n_records": len(records),
        "n_entities": len({r.entity_id for r in labeled}),
        "true_pairs": len(truth),
        "blocking": {
            "candidate_pairs": stats.candidate_pairs,
            "total_pairs": stats.total_pairs,
            "reduction_ratio": round(stats.reduction_ratio, 4),
            "pair_completeness": round(len(pairs & truth) / len(truth), 4) if truth else 1.0,
        },
        "baseline_exact_name_or_domain": asdict(prf(baseline_pairs(records), truth)),
        "models": {},
    }
    domains = [r.site or r.domain for r in records]  # same cannot-link key as the engine
    for name, scorer in scorers.items():
        scored = score_pairs(records, pairs, scorer)
        sweep = []
        for t in thresholds:
            pairwise = {p for p, s in zip(scored.pairs, scored.probs, strict=True) if s >= t}
            row: dict[str, Any] = {"threshold": t, "pairwise": asdict(prf(pairwise, truth))}
            edges = [(i, j, float(s)) for (i, j), s in zip(scored.pairs, scored.probs, strict=True)]
            for md in max_domains_options:
                labels = cluster(
                    len(records),
                    edges,
                    threshold=t,
                    domains=domains,
                    max_domains=md,
                    veto_below=veto_below if veto_below is not None and veto_below < t else None,
                )
                key = "clustered" if md is None else f"clustered_max_domains_{md}"
                row[key] = asdict(prf(clusters_to_pairs(labels), truth))
            sweep.append(row)
        best: dict[str, Any] = {}
        for key in ["pairwise"] + [
            "clustered" if md is None else f"clustered_max_domains_{md}"
            for md in max_domains_options
        ]:
            top = _best_row(sweep, key)
            best[key] = {"threshold": top["threshold"], **top[key]}
        report["models"][name] = {"best": best, "sweep": sweep}
    return report
