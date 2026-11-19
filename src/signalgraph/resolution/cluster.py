"""Union-find clustering of scored pairs into entities.

Transitive closure is what turns pairwise decisions into accounts, and also
what makes ER fail at scale: one bad edge chains two large clusters together.
``cluster`` therefore processes edges strongest-first and can enforce a
*cannot-link* constraint: never merge two clusters that already hold
different canonical domains (``max_domains=1``). Whether that helps depends on
the data (it blocks some rebrand merges), so it is a switch measured on the
gold set rather than an assumption.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def cluster(
    n: int,
    edges: Iterable[tuple[int, int, float]],
    *,
    threshold: float,
    domains: Sequence[Hashable | None] | None = None,
    max_domains: int | None = None,
) -> list[int]:
    """Return a cluster label (the root index) for each of ``n`` items.

    ``edges`` are ``(i, j, p_match)``. Edges below ``threshold`` are ignored.
    Labels are canonicalised to the smallest member index, so the output is
    independent of edge order whenever no constraint is active.
    """
    uf = UnionFind(n)
    constrained = domains is not None and max_domains is not None
    root_domains: dict[int, frozenset[Hashable]] = {}
    if constrained:
        assert domains is not None
        root_domains = {
            i: (frozenset({d}) if d is not None else frozenset()) for i, d in enumerate(domains)
        }
    strong = sorted(
        ((p, i, j) for i, j, p in edges if p >= threshold), key=lambda e: (-e[0], e[1], e[2])
    )
    for _, i, j in strong:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if constrained:
            merged = root_domains[ri] | root_domains[rj]
            assert max_domains is not None
            if len(merged) > max_domains:
                continue
            root = uf.union(ri, rj)
            root_domains[root] = merged
        else:
            uf.union(ri, rj)
    smallest: dict[int, int] = {}
    roots = [uf.find(i) for i in range(n)]
    for i, r in enumerate(roots):
        smallest.setdefault(r, i)
    return [smallest[r] for r in roots]


def clusters_to_pairs(labels: Sequence[int]) -> set[tuple[int, int]]:
    """All within-cluster pairs (i < j): used for pairwise evaluation."""
    groups: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    pairs: set[tuple[int, int]] = set()
    for members in groups.values():
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1 :]:
                pairs.add((a, b))
    return pairs
