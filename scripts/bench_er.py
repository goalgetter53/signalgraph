"""Entity-resolution throughput benchmark on SYNTHETIC records.

Runs the full production resolver (normalise -> block -> score -> cluster ->
materialise accounts) on 10k and 50k generated records, ``--repeats`` times
each with cold caches, and reports the median wall time, records/sec, stage breakdown and the
pairwise P/R/F1 against the generator's labels.

    python scripts/bench_er.py [--sizes 10000 50000] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from signalgraph.domain.models import CompanyMention
from signalgraph.resolution.cluster import clusters_to_pairs
from signalgraph.resolution.domains import canonical_domain
from signalgraph.resolution.engine import EntityResolver
from signalgraph.resolution.evaluate import prf, true_pairs
from signalgraph.resolution.features import _jw_ordered
from signalgraph.resolution.synth import LabeledRecord, generate

OUT = Path(__file__).resolve().parents[1] / "reports" / "benchmark.json"


def records_for(n: int, seed: int) -> list[LabeledRecord]:
    entities = max(10, int(n / 2.2))
    labeled = generate(entities, seed=seed)
    while len(labeled) < n:
        entities = int(entities * 1.2)
        labeled = generate(entities, seed=seed)
    return labeled[:n]


def bench(n: int, repeats: int, seed: int) -> dict[str, Any]:
    labeled = records_for(n, seed)
    mentions = [
        CompanyMention(
            mention_id=r.record_id, source="synthetic", raw_name=r.name,
            raw_domain=r.domain, country=r.country,
        )
        for r in labeled
    ]  # fmt: skip
    runs = []
    result = None
    for _ in range(repeats):
        # cold caches every repeat, so later repeats are not flattered by memoisation
        _jw_ordered.cache_clear()
        canonical_domain.cache_clear()
        resolver = EntityResolver()
        t0 = time.perf_counter()
        result = resolver.resolve(mentions)
        wall = time.perf_counter() - t0
        runs.append((wall, result.timings_s))
    assert result is not None
    walls = [w for w, _ in runs]
    median_wall = statistics.median(walls)
    median_run = min(runs, key=lambda r: abs(r[0] - median_wall))
    quality = prf(clusters_to_pairs(result.labels), true_pairs(labeled))
    return {
        "records": n,
        "entities": len({r.entity_id for r in labeled}),
        "accounts_out": len(result.accounts),
        "repeats": repeats,
        "wall_s_median": round(median_wall, 3),
        "wall_s_all": [round(w, 3) for w in walls],
        "records_per_s": round(n / median_wall),
        "stage_s": {k: round(v, 3) for k, v in median_run[1].items()},
        "candidate_pairs": result.blocking.candidate_pairs,
        "all_pairs": result.blocking.total_pairs,
        "reduction_ratio": round(result.blocking.reduction_ratio, 6),
        "skipped_oversized_blocks": result.blocking.n_skipped_blocks,
        "pairwise_quality_vs_synthetic_labels": quality.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    machine = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or platform.machine(),
    }
    results = []
    for n in args.sizes:
        row = bench(n, args.repeats, args.seed)
        results.append(row)
        print(
            f"{n:>6} records: {row['wall_s_median']:.2f}s median -> {row['records_per_s']:,} rec/s"
            f" | pairs {row['candidate_pairs']:,} of {row['all_pairs']:,}"
            f" (RR {row['reduction_ratio']:.5f}) | stages {row['stage_s']}"
            f" | P/R/F1 {row['pairwise_quality_vs_synthetic_labels']}"
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"machine": machine, "results": results}, indent=2) + "\n")
    print(f"machine: {machine}\nwrote {OUT}")


if __name__ == "__main__":
    main()
