"""Train the pair scorer on SYNTHETIC records and write the model JSON.

The hand-labelled gold set is never used here; it is the held-out test set
for ``scripts/eval_er.py``.

    python scripts/train_er.py [--entities 3000] [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import time

from signalgraph.resolution.evaluate import training_matrix
from signalgraph.resolution.scorers import DEFAULT_MODEL_PATH, LogisticMatcher
from signalgraph.resolution.synth import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    labeled = generate(args.entities, seed=args.seed)
    x, y = training_matrix(labeled)
    model = LogisticMatcher(l2=args.l2).fit(x, y)
    data = model.to_dict()
    data["training"] = {
        "data": "synthetic (signalgraph.resolution.synth.generate)",
        "seed": args.seed,
        "n_entities": args.entities,
        "n_records": len(labeled),
        "n_pairs": len(y),
        "n_positive_pairs": int(y.sum()),
    }
    DEFAULT_MODEL_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2))
    print(f"trained in {time.perf_counter() - t0:.1f}s -> {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
