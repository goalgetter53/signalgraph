"""v2 real-data evaluation: before vs after on the held-out TEST split.

* Split: ``data/real/wikidata_split.csv`` (``make_split.py``; by company group, fixed seed).
* before: the v1 shipped engine, frozen as clusters in
  ``reports/real_eval_v2_before_clusters.json`` (``predict_split.py`` run at commit 7ab183f).
* after: the engine as checked out (``EntityResolver()`` defaults), whose model, threshold,
  veto and blocking width were fitted / picked on DEV only (``tune_dev.py``, ``train_er.py``).
* naive: exact normalised name OR same registrable domain, transitively closed.

Each split is resolved as its own batch. TEST is the result; DEV is in-sample for the
after engine (it was trained and tuned there) and FULL mixes both, so those two are
reported only as clearly labelled context.

    python benchmarks/real/run_split_eval.py  ->  reports/real_eval_v2.{json,md}
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_split import load_split_rows, resolve_clusters

from signalgraph.benchmarks.wikidata import (
    bootstrap_ci,
    component_counts,
    pair_components,
)
from signalgraph.resolution.blocking import candidate_pairs
from signalgraph.resolution.engine import DEFAULT_THRESHOLD, DEFAULT_VETO, EntityResolver
from signalgraph.resolution.evaluate import PRF, baseline_pairs, prf
from signalgraph.resolution.features import Record

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "reports" / "real_eval_v2_before_clusters.json"
OUT_JSON = ROOT / "reports" / "real_eval_v2.json"
OUT_MD = ROOT / "reports" / "real_eval_v2.md"
N_EXAMPLES = 20


def _pairs(clusters: list[list[str]], index: dict[str, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for cl in clusters:
        ids = sorted(index[r] for r in cl)
        out.update((a, b) for k, a in enumerate(ids) for b in ids[k + 1 :])
    return out


def _relation(a: dict[str, str], b: dict[str, str]) -> str:
    pa = set(filter(None, a["parents"].split("|")))
    pb = set(filter(None, b["parents"].split("|")))
    if a["entity_id"] in pb or b["entity_id"] in pa:
        return "parent_subsidiary"
    return "sibling_subsidiaries" if pa & pb else "unrelated"


def evaluate_split(rows: list[dict[str, str]], before_clusters: list[list[str]]) -> dict[str, Any]:
    index = {r["record_id"]: i for i, r in enumerate(rows)}
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["entity_id"], []).append(i)
    truth = {(a, b) for m in groups.values() for k, a in enumerate(m) for b in m[k + 1 :]}
    records = [
        Record.build(i, r["record_id"], r["name"], r["domain"] or None, r["country"] or None)
        for i, r in enumerate(rows)
    ]
    methods = {
        "after": _pairs(resolve_clusters(rows, EntityResolver()), index),
        "before": _pairs(before_clusters, index),
        "naive": baseline_pairs(records),
    }
    scores: dict[str, PRF] = {k: prf(v, truth) for k, v in methods.items()}
    comps = pair_components(len(rows), [truth, *methods.values()])
    counts = {k: component_counts(comps, v, truth) for k, v in methods.items()}
    ci_ab = bootstrap_ci({"after": counts["after"], "before": counts["before"]}, seed=0)
    ci_an = bootstrap_ci({"after": counts["after"], "naive": counts["naive"]}, seed=0)
    ci_bn = bootstrap_ci({"before": counts["before"], "naive": counts["naive"]}, seed=0)
    cands, _ = candidate_pairs(records)
    fp_rel = {
        k: dict(sorted(Counter(_relation(rows[i], rows[j]) for i, j in v - truth).items()))
        for k, v in methods.items()
    }

    def ex(i: int, j: int) -> dict[str, Any]:
        return {
            "a": f"{rows[i]['name']} ({rows[i]['source']}, {rows[i]['domain'] or '-'})",
            "b": f"{rows[j]['name']} ({rows[j]['source']}, {rows[j]['domain'] or '-'})",
            "relation": "same_entity"
            if rows[i]["entity_id"] == rows[j]["entity_id"]
            else _relation(rows[i], rows[j]),
            "blocked": (i, j) in cands,
        }

    after = methods["after"]
    return {
        "n_records": len(rows),
        "n_entities": len(groups),
        "true_pairs": len(truth),
        "blocking_pair_completeness_after": round(len(cands & truth) / len(truth), 4),
        "candidate_pairs_after": len(cands),
        "scores": {k: asdict(v) for k, v in scores.items()},
        "bootstrap_95ci": {
            "after": ci_ab["after"],
            "before": ci_ab["before"],
            "naive": ci_an["naive"],
            "f1_diff_after_minus_before": ci_ab["f1_diff_after_minus_before"],
            "f1_diff_after_minus_naive": ci_an["f1_diff_after_minus_naive"],
            "f1_diff_before_minus_naive": ci_bn["f1_diff_before_minus_naive"],
            "n_components": ci_ab["n_components"],
            "n_resamples": ci_ab["n_resamples"],
        },
        "false_merges_by_relation": fp_rel,
        "examples_after": {
            "misses": [ex(*p) for p in sorted(truth - after)[:N_EXAMPLES]],
            "false_merges": [ex(*p) for p in sorted(after - truth)[:N_EXAMPLES]],
        },
    }


def _row(name: str, s: dict[str, Any], ci: list[float] | None) -> str:
    c = f"{ci[0]:.3f}–{ci[1]:.3f}" if ci else "–"
    return (
        f"| {name} | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** "
        f"| {s['tp']} / {s['fp']} / {s['fn']} | {c} |"
    )


def main() -> None:
    splits = load_split_rows()
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    tune = json.loads((ROOT / "reports" / "real_tune_dev.json").read_text(encoding="utf-8"))
    model = json.loads(
        (ROOT / "src" / "signalgraph" / "resolution" / "er_model.json").read_text(encoding="utf-8")
    )
    with (ROOT / "data" / "real" / "wikidata_split.csv").open(encoding="utf-8") as fh:
        n_groups = Counter(r["split"] for r in csv.DictReader(fh))
    results = {name: evaluate_split(splits[name], before[name]) for name in ("test", "dev", "full")}
    report = {
        "protocol": {
            "split": "data/real/wikidata_split.csv (make_split.py, seed 20261120, "
            "by company group)",
            "entities_per_split": dict(n_groups),
            "before": "v1 engine (commit 7ab183f), frozen clusters in "
            "reports/real_eval_v2_before_clusters.json",
            "after": {
                "threshold": DEFAULT_THRESHOLD,
                "veto_below": DEFAULT_VETO,
                "chosen_on": "DEV only, out-of-fold (reports/real_tune_dev.json)",
                "dev_oof_choice": tune["chosen"],
                "model_training": model["training"],
            },
            "test_status": "held out: not used for any fitting, feature or threshold choice",
            "dev_status": "IN-SAMPLE for the after engine (trained and tuned on it)",
            "full_status": "PARTLY IN-SAMPLE (half the records are DEV)",
            "bootstrap": "paired percentile bootstrap over connected components, 1000 resamples",
        },
        **results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Real-data ER v2: held-out TEST split, before vs after",
        "",
        "Generated by `benchmarks/real/run_split_eval.py`. Gold set: Wikidata companies "
        "(CC0), split by company group into DEV / TEST (`data/real/wikidata_split.csv`, "
        f"seed 20261120, entities per split {dict(n_groups)}). All v2 changes were fitted "
        "and tuned on DEV only; TEST was scored once.",
        "",
        f"After-engine settings (chosen on DEV, out-of-fold): threshold {DEFAULT_THRESHOLD}, "
        f"veto {DEFAULT_VETO}, model trained on synthetic + DEV real pairs "
        f"(real weight {model['training']['real_weight']}).",
        "",
    ]
    titles = {
        "test": "TEST (held out) — the result",
        "dev": "DEV — in-sample for *after* (trained and tuned here), context only",
        "full": "FULL set — partly in-sample (half is DEV), context only",
    }
    for name in ("test", "dev", "full"):
        r = results[name]
        ci = r["bootstrap_95ci"]
        lines += [
            f"## {titles[name]}",
            "",
            f"{r['n_records']} records, {r['n_entities']} entities, {r['true_pairs']} true "
            f"pairs. After-engine blocking: {r['candidate_pairs_after']} candidate pairs, "
            f"pair completeness {r['blocking_pair_completeness_after']:.3f}.",
            "",
            "| Method | Precision | Recall | F1 | TP / FP / FN | 95% CI F1 |",
            "|---|---|---|---|---|---|",
            _row(
                "Naive: exact normalised name OR same domain",
                r["scores"]["naive"],
                ci["naive"]["f1"],
            ),
            _row("Engine before (v1, shipped)", r["scores"]["before"], ci["before"]["f1"]),
            _row("Engine after (v2)", r["scores"]["after"], ci["after"]["f1"]),
            "",
            f"Paired bootstrap 95% CI of F1 difference ({ci['n_resamples']} resamples over "
            f"{ci['n_components']} components): after − before "
            f"**{ci['f1_diff_after_minus_before'][0]:+.3f} to "
            f"{ci['f1_diff_after_minus_before'][1]:+.3f}**; after − naive "
            f"{ci['f1_diff_after_minus_naive'][0]:+.3f} to "
            f"{ci['f1_diff_after_minus_naive'][1]:+.3f}; before − naive "
            f"{ci['f1_diff_before_minus_naive'][0]:+.3f} to "
            f"{ci['f1_diff_before_minus_naive'][1]:+.3f}.",
            "",
            "False merges by relation: "
            + "; ".join(f"{k} {v}" for k, v in r["false_merges_by_relation"].items()),
            "",
        ]
    t = results["test"]["examples_after"]
    for title, key in (("misses", "misses"), ("false merges", "false_merges")):
        lines += [f"## TEST: after-engine {title} (first {N_EXAMPLES})", ""]
        lines += ["| A | B | relation | blocked |", "|---|---|---|---|"]
        lines += [f"| `{e['a']}` | `{e['b']}` | {e['relation']} | {e['blocked']} |" for e in t[key]]
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: 8 + 14 * 3]))
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
