"""Train the pair scorer and write the model JSON.

Training data: SYNTHETIC records plus the candidate pairs of the DEV half of the real
Wikidata gold set (``data/real/wikidata_split.csv``), repeated ``--real-weight`` times
(10, chosen out-of-fold on DEV by ``benchmarks/real/tune_dev.py``). The real TEST half
and the hand-labelled synthetic gold set (``data/gold``) are never used here.

    python scripts/train_er.py [--entities 3000] [--seed 13] [--real-weight 10]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from signalgraph.resolution.evaluate import load_gold, training_matrix
from signalgraph.resolution.scorers import DEFAULT_MODEL_PATH, LogisticMatcher
from signalgraph.resolution.synth import generate

ROOT = Path(__file__).resolve().parents[1]
REAL_GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
REAL_SPLIT = ROOT / "data" / "real" / "wikidata_split.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--real-weight", type=int, default=10)
    args = parser.parse_args()

    t0 = time.perf_counter()
    labeled = generate(args.entities, seed=args.seed)
    x, y = training_matrix(labeled)
    with REAL_SPLIT.open(encoding="utf-8", newline="") as fh:
        side = {r["entity_id"]: r["split"] for r in csv.DictReader(fh)}
    dev = [g for g in load_gold(REAL_GOLD) if side[g.entity_id] == "dev"]
    x_real, y_real = training_matrix(dev)
    model = LogisticMatcher(l2=args.l2).fit(
        np.vstack([x] + [x_real] * args.real_weight),
        np.concatenate([y] + [y_real] * args.real_weight),
    )
    data = model.to_dict()
    data["training"] = {
        "data": "synthetic (signalgraph.resolution.synth.generate) + real Wikidata DEV split",
        "seed": args.seed,
        "n_entities": args.entities,
        "n_records": len(labeled),
        "n_pairs": len(y),
        "n_positive_pairs": int(y.sum()),
        "real_dev_records": len(dev),
        "real_dev_pairs": len(y_real),
        "real_dev_positive_pairs": int(y_real.sum()),
        "real_weight": args.real_weight,
    }
    DEFAULT_MODEL_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    print(f"trained in {time.perf_counter() - t0:.1f}s -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
