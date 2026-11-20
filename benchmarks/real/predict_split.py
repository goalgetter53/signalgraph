"""Run the engine *as checked out* (``EntityResolver()`` defaults) on each split of the
real gold set and write its clusters (lists of record ids) to JSON.

Used to freeze the "before" engine: ``reports/real_eval_v2_before_clusters.json`` was
written by running this file against commit 7ab183f (the engine before the v2 changes),
so the before/after comparison in ``run_split_eval.py`` uses the real shipped engine,
not a re-implementation of it. Each split is resolved as its own batch.

    python benchmarks/real/predict_split.py OUT.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from signalgraph.domain.models import CompanyMention
from signalgraph.resolution.engine import EntityResolver

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
SPLIT = ROOT / "data" / "real" / "wikidata_split.csv"


def load_split_rows() -> dict[str, list[dict[str, str]]]:
    with SPLIT.open(encoding="utf-8", newline="") as fh:
        side = {r["entity_id"]: r["split"] for r in csv.DictReader(fh)}
    with GOLD.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, list[dict[str, str]]] = {"dev": [], "test": [], "full": rows}
    for r in rows:
        out[side[r["entity_id"]]].append(r)
    return out


def resolve_clusters(rows: list[dict[str, str]], resolver: EntityResolver) -> list[list[str]]:
    mentions = [
        CompanyMention(
            mention_id=r["record_id"],
            source=r["source"],
            raw_name=r["name"],
            raw_domain=r["domain"] or None,
            country=r["country"] or None,
        )
        for r in rows
    ]
    result = resolver.resolve(mentions)
    groups: dict[int, list[str]] = {}
    for r, label in zip(rows, result.labels, strict=True):
        groups.setdefault(label, []).append(r["record_id"])
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def main() -> None:
    out = Path(sys.argv[1])
    splits = load_split_rows()
    payload = {name: resolve_clusters(rows, EntityResolver()) for name, rows in splits.items()}
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: " + ", ".join(f"{k}={len(v)} clusters" for k, v in payload.items()))


if __name__ == "__main__":
    main()
