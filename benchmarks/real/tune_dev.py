"""Choose the v2 engine settings on the DEV half of the real gold set only.

Out-of-fold protocol, so the threshold is not picked on pairs the model was fitted on:
DEV companies are split into 5 folds by split group; for each fold the pair scorer is
fitted on synthetic pairs + real DEV pairs from the other 4 folds and scores the held-out
fold's pairs. The whole DEV batch is then clustered with those out-of-fold scores, and
threshold / veto / blocking width / real-pair weight are picked by DEV F1.
TEST is never loaded.

    python benchmarks/real/tune_dev.py   ->  reports/real_tune_dev.json
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from signalgraph.resolution.blocking import candidate_pairs
from signalgraph.resolution.cluster import cluster, clusters_to_pairs
from signalgraph.resolution.engine import score_pairs
from signalgraph.resolution.evaluate import (
    load_gold,
    prf,
    to_records,
    training_matrix,
    true_pairs,
)
from signalgraph.resolution.scorers import LogisticMatcher
from signalgraph.resolution.synth import LabeledRecord, generate

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
SPLIT = ROOT / "data" / "real" / "wikidata_split.csv"
OUT = ROOT / "reports" / "real_tune_dev.json"
N_FOLDS = 5
THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 20))
VETOES: tuple[float | None, ...] = (None, 0.01, 0.02, 0.05, 0.1, 0.2)
N_RARE = (1, 2, 3, 4)
REAL_WEIGHTS = (0, 1, 3, 10)  # 0 = synthetic only (the shipped v1 training data)


def dev_records() -> tuple[list[LabeledRecord], list[int]]:
    """DEV labelled records and a fold id per record (folds = split groups mod N_FOLDS)."""
    with SPLIT.open(encoding="utf-8", newline="") as fh:
        split_rows = list(csv.DictReader(fh))
    side = {r["entity_id"]: r["split"] for r in split_rows}
    groups = sorted({r["group"] for r in split_rows if r["split"] == "dev"})
    fold_of_group = {g: k % N_FOLDS for k, g in enumerate(groups)}
    group = {r["entity_id"]: r["group"] for r in split_rows}
    dev = [g for g in load_gold(GOLD) if side[g.entity_id] == "dev"]
    return dev, [fold_of_group[group[g.entity_id]] for g in dev]


def main() -> None:
    dev, fold = dev_records()
    records = to_records(dev)
    truth = true_pairs(dev)
    sites = [r.site or r.domain for r in records]
    x_syn, y_syn = training_matrix(generate(3000, seed=13))
    rows: list[dict[str, Any]] = []
    for n_rare in N_RARE:
        cands, _ = candidate_pairs(records, n_rare=n_rare)
        scored = score_pairs(records, cands, LogisticMatcher())
        x, pairs = scored.features, scored.pairs
        y = np.array([int(dev[i].entity_id == dev[j].entity_id) for i, j in pairs])
        completeness = len(cands & truth) / len(truth)
        for weight in REAL_WEIGHTS:
            oof = np.zeros(len(pairs))
            for k in range(N_FOLDS):
                held = np.array([fold[i] == k or fold[j] == k for i, j in pairs])
                train = ~np.array([fold[i] == k or fold[j] == k for i, j in pairs])
                xs = [x_syn] + [x[train]] * weight
                ys = [y_syn] + [y[train]] * weight
                model = LogisticMatcher(l2=1.0).fit(np.vstack(xs), np.concatenate(ys))
                oof[held] = model.predict_proba(x[held])
            edges = [(i, j, float(p)) for (i, j), p in zip(pairs, oof, strict=True)]
            for t, veto in product(THRESHOLDS, VETOES):
                if veto is not None and veto >= t:
                    continue
                labels = cluster(
                    len(records), edges, threshold=t, domains=sites, max_domains=1, veto_below=veto
                )
                m = prf(clusters_to_pairs(labels), truth)
                rows.append(
                    {
                        "n_rare": n_rare,
                        "real_weight": weight,
                        "threshold": t,
                        "veto_below": veto,
                        "blocking_completeness": round(completeness, 4),
                        "candidate_pairs": len(cands),
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                    }
                )
    # ties -> fewer candidate pairs, then threshold closest to 0.5
    best = max(rows, key=lambda r: (r["f1"], -r["candidate_pairs"], -abs(r["threshold"] - 0.5)))

    def best_where(**cond: Any) -> dict[str, Any]:
        pool = [r for r in rows if all(r[k] == v for k, v in cond.items())]
        return max(pool, key=lambda r: (r["f1"], -abs(r["threshold"] - 0.5)))

    ablations = {
        "chosen": best,
        "no_veto": best_where(
            n_rare=best["n_rare"], real_weight=best["real_weight"], veto_below=None
        ),
        "synthetic_only_training": best_where(
            n_rare=best["n_rare"], real_weight=0, veto_below=best["veto_below"]
        ),
        "one_rare_token_key": best_where(
            n_rare=1, real_weight=best["real_weight"], veto_below=best["veto_below"]
        ),
    }
    report = {
        "protocol": (
            f"DEV only ({len(dev)} records, {len(truth)} true pairs); {N_FOLDS}-fold out-of-fold "
            "scores by split group; scorer = LR fitted on synthetic(3000, seed 13) + real DEV "
            "pairs of the other folds repeated real_weight times; site-key cannot-link on"
        ),
        "chosen": best,
        "ablations_best_f1_on_dev": ablations,
        "grid": rows,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, row in ablations.items():
        print(f"{name:26s} {row}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
