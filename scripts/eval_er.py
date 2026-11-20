"""Evaluate ER on the hand-labelled gold set (held out from training).

Writes ``reports/er_eval.json`` (full threshold sweep) and prints a summary
table. Every ER number in the README is copied from this script's output.

    python scripts/eval_er.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from signalgraph.resolution.engine import DEFAULT_THRESHOLD
from signalgraph.resolution.evaluate import evaluate, fit_models, load_gold
from signalgraph.resolution.scorers import PairScorer, load_default_model
from signalgraph.resolution.synth import generate

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "companies_gold.csv"
OUT = ROOT / "reports" / "er_eval.json"


def _row(name: str, key: str, best: dict[str, Any]) -> str:
    b = best[key]
    return (
        f"| {name} | {key} | {b['threshold']:.2f} | {b['precision']:.3f} | {b['recall']:.3f} "
        f"| {b['f1']:.3f} | {b['tp']} | {b['fp']} | {b['fn']} |"
    )


def main() -> None:
    gold = load_gold(GOLD)
    lr = load_default_model()
    # FS is fitted on the same synthetic training distribution as the shipped LR model.
    _, fs = fit_models(generate(3000, seed=13))
    scorers: dict[str, PairScorer] = {"logistic_regression": lr, "fellegi_sunter": fs}
    # v1 picked the operating threshold on a SYNTHETIC validation set (seed 99). Since v2
    # the shipped threshold is picked on the DEV half of the real Wikidata gold set
    # (benchmarks/real/tune_dev.py); the gold set here is reported at that shipped
    # threshold, and the synthetic-validation optimum is kept as a diagnostic.
    validation = evaluate(generate(1500, seed=99), {"logistic_regression": lr})
    val_best = validation["models"]["logistic_regression"]["best"]["clustered_max_domains_1"]
    selected = DEFAULT_THRESHOLD
    report = evaluate(gold, scorers)
    at_selected: dict[str, Any] = {}
    for name in scorers:
        row = next(r for r in report["models"][name]["sweep"] if r["threshold"] == selected)
        at_selected[name] = row
    report["synthetic_validation"] = {
        "seed": 99,
        "n_records": validation["n_records"],
        "blocking": validation["blocking"],
        "baseline_exact_name_or_domain": validation["baseline_exact_name_or_domain"],
        "best_clustered_max_domains_1": val_best,
    }
    report["selected_threshold"] = selected
    report["gold_at_selected_threshold"] = at_selected
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"gold: {report['n_records']} records, {report['n_entities']} entities, "
        f"{report['true_pairs']} true pairs"
    )
    print(f"blocking: {report['blocking']}")
    print(f"baseline (exact name OR domain): {report['baseline_exact_name_or_domain']}")
    print("| model | stage | best t | P | R | F1 | TP | FP | FN |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in scorers:
        best = report["models"][name]["best"]
        for key in best:
            print(_row(name, key, best))
    print(f"shipped threshold (picked on real DEV): {selected}")
    print(f"synthetic-validation optimum (diagnostic): {val_best}")
    for name, row in at_selected.items():
        print(f"GOLD {name} @{selected}:")
        for key, val in row.items():
            if key != "threshold":
                print(f"   {key}: {val}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
