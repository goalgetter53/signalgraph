from collections import Counter

from signalgraph.resolution.blocking import blocking_keys, candidate_pairs
from signalgraph.resolution.features import Record


def _rec(i: int, name: str, domain: str | None = None) -> Record:
    return Record.build(i, str(i), name, domain, None)


def test_keys_include_domain_rarest_token_and_prefix() -> None:
    r = _rec(0, "Acme Technologies", "https://www.acme.com")
    df = Counter({"acme": 1, "technologies": 40})
    assert blocking_keys(r, df) == {"d:acme.com", "r:acme", "p:acme", "a:at"}


def test_typo_in_rare_token_still_blocked_by_prefix() -> None:
    recs = [_rec(0, "Northwind Robotics"), _rec(1, "Northwnd Robotics")]
    pairs, _ = candidate_pairs(recs)
    assert (0, 1) in pairs


def test_domain_only_link_is_found() -> None:
    recs = [_rec(0, "Payloom", "payloom.com"), _rec(1, "Brightline Payroll", "payloom.com")]
    pairs, _ = candidate_pairs(recs)
    assert pairs == {(0, 1)}


def test_oversized_name_blocks_are_skipped_and_reported() -> None:
    recs = [_rec(i, f"Global {chr(97 + i % 26)}{i}") for i in range(30)]
    pairs, stats = candidate_pairs(recs, max_block=10)
    assert stats.n_skipped_blocks >= 1
    assert stats.candidate_pairs == len(pairs)
    assert stats.reduction_ratio > 0.5


def test_pairs_are_ordered_and_unique() -> None:
    recs = [_rec(0, "Acme"), _rec(1, "Acme Inc"), _rec(2, "ACME LLC")]
    pairs, stats = candidate_pairs(recs)
    assert pairs == {(0, 1), (0, 2), (1, 2)}
    assert stats.total_pairs == 3


def test_v2_acronym_and_second_rare_token_keys() -> None:
    from collections import Counter

    from signalgraph.resolution.blocking import blocking_keys, candidate_pairs
    from signalgraph.resolution.features import Record

    ibm = Record.build(0, "a", "IBM", None, None)
    full = Record.build(1, "b", "International Business Machines Corp", None, None)
    pairs, _ = candidate_pairs([ibm, full])
    assert (0, 1) in pairs
    rec = Record.build(0, "a", "Carl Zeiss Microscopy", None, None)
    df = Counter({"carl": 1, "zeiss": 2, "microscopy": 3})
    assert {"r:carl", "r:zeiss"} <= blocking_keys(rec, df, n_rare=2)
    assert "r:zeiss" not in blocking_keys(rec, df, n_rare=1)
