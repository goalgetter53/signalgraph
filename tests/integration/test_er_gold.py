"""Regression guard: ER quality on the hand-labelled gold set must not silently drop.

Floors sit just under the measured values in reports/er_eval.json.
"""

from signalgraph.resolution.cluster import clusters_to_pairs
from signalgraph.resolution.engine import DEFAULT_THRESHOLD, EntityResolver
from signalgraph.resolution.evaluate import evaluate, load_gold, prf, true_pairs
from signalgraph.resolution.scorers import load_default_model
from tests.conftest import GOLD


def test_gold_set_shape() -> None:
    gold = load_gold(GOLD)
    assert len(gold) >= 150
    assert len({g.record_id for g in gold}) == len(gold)


def test_blocking_keeps_almost_all_true_pairs() -> None:
    report = evaluate(load_gold(GOLD), {"lr": load_default_model()}, thresholds=[DEFAULT_THRESHOLD])
    assert report["blocking"]["pair_completeness"] >= 0.95
    assert report["blocking"]["reduction_ratio"] >= 0.95


def test_default_resolver_quality_on_gold() -> None:
    from signalgraph.domain.models import CompanyMention

    gold = load_gold(GOLD)
    mentions = [
        CompanyMention(
            mention_id=g.record_id,
            source="gold",
            raw_name=g.name,
            raw_domain=g.domain,
            country=g.country,
            aliases=g.aliases,
        )
        for g in gold
    ]
    result = EntityResolver().resolve(mentions)
    m = prf(clusters_to_pairs(result.labels), true_pairs(gold))
    assert m.precision >= 0.95
    assert m.recall >= 0.90
    assert m.f1 >= 0.93
    baseline = evaluate(gold, {})["baseline_exact_name_or_domain"]
    assert m.f1 > baseline["f1"] + 0.1


def test_shipped_threshold_matches_committed_eval_report() -> None:
    """The README quotes reports/er_eval.json; the code must ship the threshold it reports."""
    import json

    report = json.loads((GOLD.parents[2] / "reports" / "er_eval.json").read_text())
    assert report["selected_threshold"] == DEFAULT_THRESHOLD
