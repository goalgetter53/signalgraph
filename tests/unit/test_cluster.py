from signalgraph.resolution.cluster import UnionFind, cluster, clusters_to_pairs


def test_union_find_basics() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 4)
    assert len({uf.find(i) for i in range(5)}) == 2
    assert uf.find(0) == uf.find(3)


def test_threshold_filters_edges() -> None:
    labels = cluster(4, [(0, 1, 0.9), (1, 2, 0.4), (2, 3, 0.8)], threshold=0.5)
    assert labels == [0, 0, 2, 2]


def test_domain_cannot_link_blocks_chaining() -> None:
    # 0 (a.com) - 1 (no domain) - 2 (b.com): without the constraint all three chain.
    edges = [(0, 1, 0.95), (1, 2, 0.7)]
    domains = ["a.com", None, "b.com"]
    assert cluster(3, edges, threshold=0.5) == [0, 0, 0]
    assert cluster(3, edges, threshold=0.5, domains=domains, max_domains=1) == [0, 0, 2]


def test_strongest_edge_wins_under_constraint() -> None:
    edges = [(0, 1, 0.6), (1, 2, 0.99)]
    labels = cluster(3, edges, threshold=0.5, domains=["a.com", None, "b.com"], max_domains=1)
    assert labels == [0, 1, 1]


def test_clusters_to_pairs() -> None:
    assert clusters_to_pairs([0, 0, 2, 0]) == {(0, 1), (0, 3), (1, 3)}
