"""Split the real Wikidata gold set into DEV and TEST by company group (fixed seed).

A group is a connected component over QIDs linked by (a) a parent/subsidiary edge
(P749, when both companies are in the set) or (b) any shared normalised record
name (homonyms). So one company never straddles the split, and the hard negatives
of a family (parent vs subsidiary, siblings, same-label companies) land on the same
side. Groups are shuffled with a fixed seed and assigned greedily to the smaller
side (by record count) to get a ~50/50 split.

All diagnosis, feature work, retraining and threshold choice use DEV only; TEST is
scored once at the end by ``run_split_eval.py``.

    python benchmarks/real/make_split.py   ->  data/real/wikidata_split.csv
"""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from signalgraph.resolution.names import normalize_name

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
OUT = ROOT / "data" / "real" / "wikidata_split.csv"
SEED = 20261120


def main() -> None:
    with GOLD.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    qids = sorted({r["entity_id"] for r in rows}, key=lambda q: (len(q), q))
    parent = {q: q for q in qids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in rows:
        for p in filter(None, r["parents"].split("|")):
            if p in parent:
                union(r["entity_id"], p)
    by_name: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_name[normalize_name(r["name"])].add(r["entity_id"])
    for members in by_name.values():
        first, *rest = sorted(members)
        for q in rest:
            union(first, q)

    group_of = {q: find(q) for q in qids}
    size = Counter(group_of[r["entity_id"]] for r in rows)
    groups = sorted(size)
    random.Random(SEED).shuffle(groups)
    side: dict[str, str] = {}
    load = {"dev": 0, "test": 0}
    for g in groups:
        pick = "dev" if load["dev"] <= load["test"] else "test"
        side[g] = pick
        load[pick] += size[g]
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["entity_id", "group", "split"])
        for q in qids:
            writer.writerow([q, group_of[q], side[group_of[q]]])
    n_groups = Counter(side.values())
    print(f"groups: {len(groups)} {dict(n_groups)}; records: {load}; seed {SEED} -> {OUT}")


if __name__ == "__main__":
    main()
