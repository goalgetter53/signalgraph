import pytest

from signalgraph.resolution.features import (
    FEATURE_NAMES,
    Record,
    TfIdf,
    jaro,
    jaro_winkler,
    pair_features,
    residual_similarity,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("martha", "marhta", 0.9611),  # textbook values (Winkler 1990)
        ("dwayne", "duane", 0.84),
        ("dixon", "dicksonx", 0.8133),
        ("abc", "abc", 1.0),
        ("abc", "xyz", 0.0),
        ("", "abc", 0.0),
    ],
)
def test_jaro_winkler_reference_values(a: str, b: str, expected: float) -> None:
    assert jaro_winkler(a, b) == pytest.approx(expected, abs=1e-4)


def test_jaro_symmetric() -> None:
    assert jaro("crate", "trace") == pytest.approx(jaro("trace", "crate"))


def test_tfidf_rare_tokens_dominate() -> None:
    docs = [frozenset({"acme", "technologies"})] + [
        frozenset({f"brand{i}", "technologies"}) for i in range(50)
    ]
    tf = TfIdf(docs)
    shared_rare = tf.cosine(frozenset({"acme", "technologies"}), frozenset({"acme"}))
    shared_common = tf.cosine(frozenset({"acme", "technologies"}), frozenset({"technologies"}))
    assert shared_rare > 0.9 > 0.3 > shared_common
    assert tf.cosine(frozenset({"x"}), frozenset({"x"})) == pytest.approx(1.0)
    assert tf.cosine(frozenset(), frozenset({"x"})) == 0.0


def test_residual_similarity() -> None:
    assert residual_similarity(frozenset({"halcyon", "bio"}), frozenset({"halcyon"})) == 1.0
    studio = residual_similarity(
        frozenset({"tessellate", "studio"}), frozenset({"tessellate", "bio"})
    )
    bios = residual_similarity(frozenset({"halcyon", "bio"}), frozenset({"halcyon", "biosciences"}))
    assert bios > 0.8 > studio


def test_pair_features_missing_values_are_neutral() -> None:
    a = Record.build(0, "a", "Acme Robotics", None, None)
    b = Record.build(1, "b", "Acme Robotics Inc", "acme.com", "US")
    feats = dict(
        zip(FEATURE_NAMES, pair_features(a, b, TfIdf([a.name.tokens, b.name.tokens])), strict=True)
    )
    assert feats["jaro_winkler"] == 1.0
    assert feats["domain_match"] == feats["domain_mismatch"] == 0.0
    assert feats["country_match"] == feats["country_mismatch"] == 0.0


def test_pair_features_use_aliases() -> None:
    a = Record.build(0, "a", "Zentro Inc (f/k/a Acornfield Labs)", None, "US")
    b = Record.build(1, "b", "Acornfield Labs Inc", "zentro.ai", "United States")
    feats = dict(
        zip(FEATURE_NAMES, pair_features(a, b, TfIdf([a.name.tokens, b.name.tokens])), strict=True)
    )
    assert feats["jaro_winkler"] == 1.0
    assert feats["country_match"] == 1.0


def test_number_mismatch_separates_series() -> None:
    idx = FEATURE_NAMES.index("number_mismatch")

    def nm(x: str, y: str) -> float:
        a, b = Record.build(0, "a", x, None, None), Record.build(1, "b", y, None, None)
        return pair_features(a, b, TfIdf([a.name.tokens, b.name.tokens]))[idx]

    assert nm("Lakemont Partners 16 LLC", "Lakemont Partners 18 LLC") == 1.0
    assert nm("Ridgeline Capital Fund II, L.P.", "Ridgeline Capital Fund III LP") == 1.0
    assert nm("Ridgeline Capital Fund II, L.P.", "Ridgeline Capital Fund II LP") == 0.0
    assert nm("Ember Fund I", "Ember Ventures") == 0.0  # one side has no number: unknown


def test_v2_site_mismatch_and_acronym_features() -> None:
    from signalgraph.resolution.features import acronym_match, initialism

    ca = Record.build(0, "a", "Microsoft Canada", "https://www.microsoft.com/en-ca", None)
    ma = Record.build(1, "b", "Microsoft Morocco", "https://www.microsoft.com/fr-ma/", None)
    hq = Record.build(2, "c", "Microsoft", "microsoft.com/careers", None)
    hq2 = Record.build(3, "d", "Microsoft Corp", "https://www.microsoft.com/", None)
    tf = TfIdf(r.name.tokens for r in (ca, ma, hq, hq2))
    names = list(FEATURE_NAMES)

    def feat(a: Record, b: Record) -> dict[str, float]:
        return dict(zip(names, pair_features(a, b, tf), strict=True))

    sib = feat(ca, ma)
    assert sib["site_mismatch"] == 1.0
    assert sib["domain_match"] == 0.0
    assert sib["domain_mismatch"] == 0.0
    same = feat(hq, hq2)
    assert same["domain_match"] == 1.0
    assert same["site_mismatch"] == 0.0

    assert initialism(["international", "business", "machines"]) == "ibm"
    assert initialism(["bank", "of", "america"]) == "ba"
    assert acronym_match("ibm", "international business machines")
    assert acronym_match("international business machines", "ibm")
    assert not acronym_match("ibm", "ibm")
    ibm = Record.build(0, "a", "IBM", None, None)
    full = Record.build(1, "b", "International Business Machines Corporation", None, None)
    assert feat(ibm, full)["acronym_match"] == 1.0
