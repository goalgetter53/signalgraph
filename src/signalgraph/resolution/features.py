"""Comparison features for candidate record pairs.

* Jaro-Winkler on normalised names (robust to typos and suffix noise),
* token-set TF-IDF cosine (rare shared tokens count, "technologies" doesn't),
* token containment (one name's tokens inside the other's),
* residual similarity: Jaro-Winkler of the tokens left after removing the
  shared ones ("tessellate STUDIO" vs "tessellate BIO" -> low; "halcyon BIO"
  vs "halcyon BIOSCIENCES" -> high). Added after gold-set error analysis showed
  whole-string JW over-rewarding a long shared brand prefix.
* number mismatch: both names carry series numbers that differ ("Fund I" vs
  "Fund II", "Partners 16 LLC" vs "Partners 18 LLC"). Added after the first
  live EDGAR run merged two numbered SPVs.
* domain agreement / disagreement, country agreement / disagreement.

Missing values are *not* disagreement: a record without a domain gets 0 on
both domain features, which is what lets the model learn that "both have a
domain and they differ" is strong negative evidence while "unknown" is neutral.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final

from signalgraph.resolution.domains import canonical_domain
from signalgraph.resolution.names import NormalizedName, normalize_company, normalize_country

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "jaro_winkler",
    "tfidf_cosine",
    "token_containment",
    "residual_similarity",
    "number_mismatch",
    "domain_match",
    "domain_mismatch",
    "country_match",
    "country_mismatch",
)


def jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if not len1 or not len2:
        return 0.0
    window = max(0, max(len1, len2) // 2 - 1)
    flags1 = [False] * len1
    flags2 = [False] * len2
    matches = 0
    for i, ch in enumerate(s1):
        lo, hi = max(0, i - window), min(i + window + 1, len2)
        for j in range(lo, hi):
            if not flags2[j] and s2[j] == ch:
                flags1[i] = flags2[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if flags1[i]:
            while not flags2[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
    t = transpositions / 2
    return (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0


def jaro_winkler(s1: str, s2: str, prefix_scale: float = 0.1, max_prefix: int = 4) -> float:
    sim = jaro(s1, s2)
    prefix = 0
    for a, b in zip(s1[:max_prefix], s2[:max_prefix], strict=False):
        if a != b:
            break
        prefix += 1
    return sim + prefix * prefix_scale * (1.0 - sim)


@lru_cache(maxsize=1 << 19)
def _jw_ordered(a: str, b: str) -> float:
    return jaro_winkler(a, b)


def cached_jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler is symmetric, so cache on the ordered pair. Many records share a
    normalised name (in the 10k benchmark, 59k candidate pairs had only 26k distinct
    name pairs), so this roughly halves comparison work."""
    return _jw_ordered(a, b) if a <= b else _jw_ordered(b, a)


class TfIdf:
    """Smoothed IDF over token *sets*; vectors are L2-normalised binary-TF x IDF."""

    def __init__(self, documents: Iterable[frozenset[str]]):
        df: Counter[str] = Counter()
        n = 0
        for doc in documents:
            n += 1
            df.update(doc)
        self.n_docs = n
        self.df = df
        self._cache: dict[frozenset[str], dict[str, float]] = {}

    def idf(self, token: str) -> float:
        return math.log((1 + self.n_docs) / (1 + self.df.get(token, 0))) + 1.0

    def vector(self, tokens: frozenset[str]) -> dict[str, float]:
        cached = self._cache.get(tokens)
        if cached is not None:
            return cached
        weights = {t: self.idf(t) for t in tokens}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vec = {t: w / norm for t, w in weights.items()}
        self._cache[tokens] = vec
        return vec

    def cosine(self, a: frozenset[str], b: frozenset[str]) -> float:
        if not a or not b:
            return 0.0
        va, vb = self.vector(a), self.vector(b)
        if len(va) > len(vb):
            va, vb = vb, va
        return sum(w * vb.get(t, 0.0) for t, w in va.items())


@dataclass(frozen=True, slots=True)
class Record:
    """A mention prepared for matching."""

    idx: int
    key: str
    name: NormalizedName
    domain: str | None
    country: str | None
    raw_name: str = ""
    token_sets: tuple[frozenset[str], ...] = field(default=())

    @classmethod
    def build(
        cls,
        idx: int,
        key: str,
        raw_name: str,
        raw_domain: str | None,
        country: str | None,
        aliases: Sequence[str] = (),
    ) -> Record:
        name = normalize_company(raw_name, tuple(aliases))
        return cls(
            idx=idx,
            key=key,
            name=name,
            domain=canonical_domain(raw_domain),
            country=normalize_country(country),
            raw_name=raw_name,
            token_sets=tuple(frozenset(n.split()) for n in name.all if n),
        )


_ROMAN = frozenset(
    {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv"}
)


def series_tokens(tokens: frozenset[str]) -> frozenset[str]:
    """Digits and roman numerals: the tokens that distinguish fund/SPV series."""
    return frozenset(t for t in tokens if t.isdigit() or t in _ROMAN)


def residual_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """JW of the non-shared tokens; 1.0 when one side has nothing left over."""
    ra, rb = a - b, b - a
    if not ra or not rb:
        return 1.0
    return cached_jaro_winkler(" ".join(sorted(ra)), " ".join(sorted(rb)))


def pair_features(a: Record, b: Record, tfidf: TfIdf) -> list[float]:
    jw = 0.0
    cos = 0.0
    contain = 0.0
    residual = 0.0
    number_mismatch = 1.0
    for na, ta in zip(a.name.all, a.token_sets, strict=False):
        for nb, tb in zip(b.name.all, b.token_sets, strict=False):
            jw = max(jw, cached_jaro_winkler(na, nb))
            cos = max(cos, tfidf.cosine(ta, tb))
            if ta and tb:
                contain = max(contain, len(ta & tb) / min(len(ta), len(tb)))
            residual = max(residual, residual_similarity(ta, tb))
            sa, sb = series_tokens(ta), series_tokens(tb)
            if not (sa and sb and sa != sb):
                number_mismatch = 0.0  # any alias pair without conflicting numbers clears it
    both_domain = a.domain is not None and b.domain is not None
    both_country = a.country is not None and b.country is not None
    return [
        jw,
        cos,
        contain,
        residual,
        number_mismatch,
        1.0 if both_domain and a.domain == b.domain else 0.0,
        1.0 if both_domain and a.domain != b.domain else 0.0,
        1.0 if both_country and a.country == b.country else 0.0,
        1.0 if both_country and a.country != b.country else 0.0,
    ]
