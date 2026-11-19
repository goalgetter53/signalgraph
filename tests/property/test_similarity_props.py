"""Similarity-function invariants (symmetry is what makes the JW cache valid)."""

from hypothesis import given
from hypothesis import strategies as st

from signalgraph.resolution.features import TfIdf, cached_jaro_winkler, jaro_winkler

short = st.text(alphabet="abcdefgh ", max_size=12)
token_sets = st.frozensets(st.sampled_from(["acme", "data", "labs", "inc", "zentro", "ai"]))


@given(short, short)
def test_jaro_winkler_symmetric_bounded_and_cache_consistent(a: str, b: str) -> None:
    s = jaro_winkler(a, b)
    assert 0.0 <= s <= 1.0 + 1e-12
    assert abs(s - jaro_winkler(b, a)) < 1e-12
    assert cached_jaro_winkler(a, b) == cached_jaro_winkler(b, a)
    assert jaro_winkler(a, a) == 1.0


@given(st.lists(token_sets, min_size=1, max_size=8), token_sets, token_sets)
def test_tfidf_cosine_symmetric_and_bounded(docs: list[frozenset[str]], a: frozenset[str],
                                            b: frozenset[str]) -> None:  # fmt: skip
    tf = TfIdf(docs)
    c = tf.cosine(a, b)
    assert -1e-12 <= c <= 1.0 + 1e-9
    assert abs(c - tf.cosine(b, a)) < 1e-9
    if a:
        assert abs(tf.cosine(a, a) - 1.0) < 1e-9
