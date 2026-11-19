"""Property-based invariants for union-find clustering."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from signalgraph.resolution.cluster import cluster, clusters_to_pairs


@st.composite
def graphs(draw: st.DrawFn) -> tuple[int, list[tuple[int, int, float]]]:
    n = draw(st.integers(min_value=1, max_value=30))
    edges = draw(
        st.lists(
            st.tuples(
                st.integers(0, n - 1), st.integers(0, n - 1), st.floats(0, 1, allow_nan=False)
            ),
            max_size=60,
        )
    )
    return n, [(i, j, p) for i, j, p in edges if i != j]


def _components(n: int, edges: list[tuple[int, int, float]], t: float) -> set[frozenset[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j, p in edges:
        if p >= t:
            adj[i].add(j)
            adj[j].add(i)
    seen: set[int] = set()
    comps = set()
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            stack.extend(adj[v] - comp)
        seen |= comp
        comps.add(frozenset(comp))
    return comps


def _partition(labels: list[int]) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, set()).add(i)
    return {frozenset(g) for g in groups.values()}


@given(graphs(), st.floats(0, 1, allow_nan=False))
@settings(max_examples=300)
def test_equals_connected_components(g: tuple[int, list[tuple[int, int, float]]], t: float) -> None:
    n, edges = g
    labels = cluster(n, edges, threshold=t)
    assert len(labels) == n
    assert _partition(labels) == _components(n, edges, t)
    # labels are canonical: each label is the smallest index of its cluster
    assert all(labels[i] == min(c) for c in _partition(labels) for i in c)


@given(graphs(), st.floats(0, 1, allow_nan=False), st.randoms(use_true_random=False))
def test_edge_order_invariance(
    g: tuple[int, list[tuple[int, int, float]]], t: float, rnd: random.Random
) -> None:
    n, edges = g
    shuffled = edges[:]
    rnd.shuffle(shuffled)
    flipped = [(j, i, p) for i, j, p in shuffled]
    assert cluster(n, edges, threshold=t) == cluster(n, flipped, threshold=t)


@given(graphs(), st.floats(0, 1, allow_nan=False), st.floats(0, 1, allow_nan=False))
def test_higher_threshold_refines_partition(
    g: tuple[int, list[tuple[int, int, float]]], t1: float, t2: float
) -> None:
    n, edges = g
    lo, hi = sorted((t1, t2))
    fine = clusters_to_pairs(cluster(n, edges, threshold=hi))
    coarse = clusters_to_pairs(cluster(n, edges, threshold=lo))
    assert fine <= coarse


@given(
    graphs(), st.lists(st.sampled_from([None, "a.com", "b.com", "c.com"]), min_size=30, max_size=30)
)
def test_domain_constraint_holds_and_only_splits(
    g: tuple[int, list[tuple[int, int, float]]], doms: list[str | None]
) -> None:
    n, edges = g
    domains = doms[:n]
    labels = cluster(n, edges, threshold=0.5, domains=domains, max_domains=1)
    for comp in _partition(labels):
        assert len({domains[i] for i in comp if domains[i] is not None}) <= 1
    # constrained clustering is a refinement of the unconstrained one
    assert clusters_to_pairs(labels) <= clusters_to_pairs(cluster(n, edges, threshold=0.5))
