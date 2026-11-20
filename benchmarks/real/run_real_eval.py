"""Evaluate the shipped ER engine on the real Wikidata gold set.

No fitting and no tuning: the shipped model (``er_model.json``, trained on synthetic
data only) runs at the shipped threshold (``DEFAULT_THRESHOLD``, chosen on synthetic
validation) through the production ``EntityResolver``. The threshold sweep in the
report is a diagnostic ("oracle", picked on the test data) and is labelled as such.

Writes ``reports/real_eval.json`` and ``reports/real_eval.md``. Every real-data
number in the README is copied from this script's output.

    python benchmarks/real/run_real_eval.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from signalgraph.benchmarks.wikidata import bootstrap_ci, component_counts, pair_components
from signalgraph.domain.models import CompanyMention
from signalgraph.resolution.blocking import candidate_pairs
from signalgraph.resolution.cluster import clusters_to_pairs
from signalgraph.resolution.engine import DEFAULT_THRESHOLD, EntityResolver, score_pairs
from signalgraph.resolution.evaluate import (
    PRF,
    baseline_pairs,
    evaluate,
    load_gold,
    prf,
    to_records,
    true_pairs,
)
from signalgraph.resolution.scorers import load_default_model

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
META = ROOT / "data" / "real" / "wikidata_meta.json"
OUT_JSON = ROOT / "reports" / "real_eval.json"
OUT_MD = ROOT / "reports" / "real_eval.md"
N_EXAMPLES = 25


def _relation(a: dict[str, str], b: dict[str, str]) -> str:
    pa = set(filter(None, a["parents"].split("|")))
    pb = set(filter(None, b["parents"].split("|")))
    if a["entity_id"] in pb or b["entity_id"] in pa:
        return "parent_subsidiary"
    if pa & pb:
        return "sibling_subsidiaries"
    return "unrelated"


def _fmt(p: PRF | dict[str, Any]) -> str:
    d = asdict(p) if isinstance(p, PRF) else p
    return (
        f"{d['precision']:.3f} | {d['recall']:.3f} | **{d['f1']:.3f}** | "
        f"{d['tp']} / {d['fp']} / {d['fn']}"
    )


def main() -> None:
    gold = load_gold(GOLD)
    with GOLD.open(encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    meta = json.loads(META.read_text(encoding="utf-8"))
    records = to_records(gold)
    truth = true_pairs(gold)

    # 1) shipped engine, exactly as production runs it
    resolver = EntityResolver()
    mentions = [
        CompanyMention(
            mention_id=g.record_id,
            source=raw[i]["source"],
            raw_name=g.name,
            raw_domain=g.domain,
            country=g.country,
        )
        for i, g in enumerate(gold)
    ]
    result = resolver.resolve(mentions)
    engine = clusters_to_pairs(result.labels)
    # 2) README baseline: exact normalised name OR same domain, transitively closed
    baseline = baseline_pairs(records)
    engine_prf, baseline_prf = prf(engine, truth), prf(baseline, truth)

    # pair probabilities, for error analysis only (same computation as the resolver)
    cands, _ = candidate_pairs(records)
    scored = score_pairs(records, cands, load_default_model())
    prob = {p: float(s) for p, s in zip(scored.pairs, scored.probs, strict=True)}

    comps = pair_components(len(records), [truth, engine, baseline])
    ci = bootstrap_ci(
        {
            "engine": component_counts(comps, engine, truth),
            "baseline": component_counts(comps, baseline, truth),
        },
        n_resamples=1000,
        seed=0,
    )

    # diagnostic sweep (threshold picked ON the test data -> oracle, not a result)
    sweep = evaluate(gold, {"logistic_regression": load_default_model()})
    oracle = sweep["models"]["logistic_regression"]["best"]["clustered_max_domains_1"]

    # per sampling slice: same clustering, pairs restricted to records of that slice
    by_seed: dict[str, dict[str, Any]] = {}
    for group in sorted({r["seed"] for r in raw}):
        idx = {i for i, r in enumerate(raw) if r["seed"] == group}

        def within(pairs: set[tuple[int, int]], idx: set[int] = idx) -> set[tuple[int, int]]:
            return {(i, j) for i, j in pairs if i in idx and j in idx}

        by_seed[group] = {
            "n_records": len(idx),
            "n_entities": len({raw[i]["entity_id"] for i in idx}),
            "true_pairs": len(within(truth)),
            "engine": asdict(prf(within(engine), within(truth))),
            "baseline_exact_name_or_domain": asdict(prf(within(baseline), within(truth))),
        }

    # breakdowns
    def view_pair(i: int, j: int) -> str:
        return "+".join(sorted((raw[i]["source"], raw[j]["source"])))

    recall_by_view: dict[str, dict[str, int]] = {}
    for i, j in sorted(truth):
        row = recall_by_view.setdefault(view_pair(i, j), {"true": 0, "engine": 0, "baseline": 0})
        row["true"] += 1
        row["engine"] += int((i, j) in engine)
        row["baseline"] += int((i, j) in baseline)
    fp_by_relation: dict[str, Counter[str]] = {
        name: Counter(_relation(raw[i], raw[j]) for i, j in pairs - truth)
        for name, pairs in (("engine", engine), ("baseline", baseline))
    }

    def example(i: int, j: int) -> dict[str, Any]:
        return {
            "a": {k: raw[i][k] for k in ("record_id", "name", "domain", "country", "source")},
            "b": {k: raw[j][k] for k in ("record_id", "name", "domain", "country", "source")},
            "relation": "same_entity"
            if raw[i]["entity_id"] == raw[j]["entity_id"]
            else _relation(raw[i], raw[j]),
            "blocked_as_candidate": (i, j) in cands,
            "p_match": round(prob[(i, j)], 4) if (i, j) in prob else None,
            "canonical": [records[i].name.primary, records[j].name.primary],
        }

    misses = sorted(truth - engine, key=lambda p: (raw[p[0]]["record_id"], raw[p[1]]["record_id"]))
    false_merges = sorted(
        engine - truth, key=lambda p: (raw[p[0]]["record_id"], raw[p[1]]["record_id"])
    )
    blocking_completeness = len(cands & truth) / len(truth) if truth else 1.0

    report: dict[str, Any] = {
        "data": {
            "source": meta["source"],
            "pulled_utc": meta["pulled_utc"],
            "gold_csv": str(GOLD.relative_to(ROOT)),
            "seed_counts": meta["seed_counts"],
            "records_by_source": meta["records_by_source"],
        },
        "n_records": len(records),
        "n_entities": len({g.entity_id for g in gold}),
        "true_pairs": len(truth),
        "total_pairs": len(records) * (len(records) - 1) // 2,
        "protocol": {
            "model": "shipped er_model.json (trained on synthetic data only)",
            "threshold": DEFAULT_THRESHOLD,
            "threshold_source": "chosen on synthetic validation; not tuned on this data",
            "max_domains": resolver.max_domains,
            "metrics": "pairwise over all record pairs, transitive clusters",
        },
        "blocking": {
            "candidate_pairs": len(cands),
            "pair_completeness": round(blocking_completeness, 4),
            "skipped_blocks": result.blocking.n_skipped_blocks,
        },
        "engine": asdict(engine_prf),
        "baseline_exact_name_or_domain": asdict(baseline_prf),
        "bootstrap_95ci": ci,
        "oracle_threshold_diagnostic": oracle,
        "by_sampling_slice": by_seed,
        "recall_by_view_pair": recall_by_view,
        "false_merges_by_relation": {k: dict(sorted(v.items())) for k, v in fp_by_relation.items()},
        "engine_misses_total": len(misses),
        "engine_false_merges_total": len(false_merges),
        "examples": {
            "misses": [example(*p) for p in misses[:N_EXAMPLES]],
            "false_merges": [example(*p) for p in false_merges[:N_EXAMPLES]],
        },
        "timings_s": {k: round(v, 3) for k, v in result.timings_s.items()},
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    e_ci, b_ci = ci["engine"], ci["baseline"]
    diff_ci = ci["f1_diff_engine_minus_baseline"]
    lines = [
        "# Entity resolution on real data (Wikidata companies)",
        "",
        f"Generated by `benchmarks/real/run_real_eval.py`. Data: {meta['source']}, "
        f"pulled {meta['pulled_utc']}.",
        "",
        f"- Records: **{report['n_records']}**, entities (QIDs): **{report['n_entities']}**, "
        f"true co-referent pairs: **{report['true_pairs']}** "
        f"(of {report['total_pairs']:,} record pairs)",
        f"- Records by source view: {meta['records_by_source']}",
        f"- Protocol: shipped model, threshold {DEFAULT_THRESHOLD} (picked on synthetic "
        "validation), no fitting on this data",
        f"- Blocking: {len(cands)} candidate pairs, pair completeness {blocking_completeness:.3f}",
        "",
        "| Method | Precision | Recall | F1 | TP / FP / FN | 95% CI F1 |",
        "|---|---|---|---|---|---|",
        f"| Baseline: exact normalised name OR same domain | {_fmt(baseline_prf)} "
        f"| {b_ci['f1'][0]:.3f}–{b_ci['f1'][1]:.3f} |",
        f"| signalgraph engine (LR + constrained union-find) | {_fmt(engine_prf)} "
        f"| {e_ci['f1'][0]:.3f}–{e_ci['f1'][1]:.3f} |",
        "",
        f"F1 difference (engine − baseline), paired bootstrap 95% CI: "
        f"{diff_ci[0]:+.3f} to {diff_ci[1]:+.3f} "
        f"({ci['n_resamples']} resamples over {ci['n_components']} connected components).",
        "",
        f"Oracle diagnostic (threshold chosen on this test set, NOT a result): best F1 "
        f"{oracle['f1']:.3f} at threshold {oracle['threshold']}.",
        "",
        "## By sampling slice (pairs within the slice; same run)",
        "",
        "| Slice | Records | Entities | True pairs | Engine P / R / F1 | Baseline P / R / F1 |",
        "|---|---|---|---|---|---|",
        *[
            f"| {g} | {v['n_records']} | {v['n_entities']} | {v['true_pairs']} "
            f"| {v['engine']['precision']:.3f} / {v['engine']['recall']:.3f} / "
            f"{v['engine']['f1']:.3f} "
            f"| {v['baseline_exact_name_or_domain']['precision']:.3f} / "
            f"{v['baseline_exact_name_or_domain']['recall']:.3f} / "
            f"{v['baseline_exact_name_or_domain']['f1']:.3f} |"
            for g, v in by_seed.items()
        ],
        "",
        "## Recall by source-view pair",
        "",
        "| View pair | True pairs | Engine recall | Baseline recall |",
        "|---|---|---|---|",
    ]
    for key, row in sorted(recall_by_view.items()):
        lines.append(
            f"| {key} | {row['true']} | {row['engine'] / row['true']:.3f} "
            f"| {row['baseline'] / row['true']:.3f} |"
        )
    lines += [
        "",
        "## False merges by relation",
        "",
        "| Relation | Engine | Baseline |",
        "|---|---|---|",
    ]
    for rel in sorted(set(fp_by_relation["engine"]) | set(fp_by_relation["baseline"])):
        lines.append(
            f"| {rel} | {fp_by_relation['engine'][rel]} | {fp_by_relation['baseline'][rel]} |"
        )
    for title, key in (("Engine misses", "misses"), ("Engine false merges", "false_merges")):
        total = len(misses) if key == "misses" else len(false_merges)
        lines += ["", f"## {title} (first {N_EXAMPLES} of {total})", ""]
        lines += ["| A | B | relation | blocked | p_match |", "|---|---|---|---|---|"]
        for ex in report["examples"][key]:
            a, b = ex["a"], ex["b"]
            lines.append(
                f"| `{a['name']}` ({a['source']}, {a['domain'] or '–'}) "
                f"| `{b['name']}` ({b['source']}, {b['domain'] or '–'}) "
                f"| {ex['relation']} | {ex['blocked_as_candidate']} | {ex['p_match']} |"
            )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"wrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
