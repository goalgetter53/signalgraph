"""Blocking: generate candidate pairs without comparing all n^2 pairs.

A pair is a candidate if the two records share *any* blocking key:

* ``d:<domain>``       exact canonical domain,
* ``r:<token>``        the record's rarest name token (per name/alias), using
                       document frequency over the batch, so ``acme`` blocks
                       rather than ``technologies``,
* ``p:<prefix>``       first ``prefix_len`` characters of the space-less name
                       (catches typos in the rare token, e.g. ``acmee``),
* ``a:<initialism>``   (v2) initials of a multi-token name, or a short single-token
                       name itself, so ``IBM`` meets ``International Business Machines``.

``n_rare`` (v2) sets how many of the rarest tokens per name get an ``r:`` key; the
real-data analysis showed one key per name missed rebrands that kept a *second*
distinctive token ("Carl Zeiss Microscopy" / "Zeiss Microscopy").

Oversized blocks (> ``max_block``) are skipped for name keys: a block of
5 000 records sharing the prefix ``inte`` would cost 12.5M comparisons and
is almost all non-matches. Domain blocks get a larger cap. The skip count is
reported so the trade-off is visible, not silent.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from signalgraph.resolution.features import Record, initialism

_MAX_ACRONYM = 6


@dataclass(frozen=True)
class BlockingStats:
    n_records: int
    n_blocks: int
    n_skipped_blocks: int
    candidate_pairs: int

    @property
    def total_pairs(self) -> int:
        return self.n_records * (self.n_records - 1) // 2

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - self.candidate_pairs / self.total_pairs if self.total_pairs else 0.0


def blocking_keys(
    record: Record, df: Counter[str], prefix_len: int = 4, n_rare: int = 1
) -> set[str]:
    keys: set[str] = set()
    if record.domain:
        keys.add(f"d:{record.domain}")
    for name, tokens in zip(record.name.all, record.token_sets, strict=False):
        if not tokens:
            continue
        for tok in sorted(tokens, key=lambda t: (df[t], t))[:n_rare]:
            keys.add(f"r:{tok}")
        squashed = name.replace(" ", "")
        if len(squashed) >= 3:
            keys.add(f"p:{squashed[:prefix_len]}")
        words = name.split()
        acro = initialism(words) if len(words) >= 2 else name
        if 2 <= len(acro) <= _MAX_ACRONYM:
            keys.add(f"a:{acro}")
    return keys


def candidate_pairs(
    records: Sequence[Record],
    *,
    prefix_len: int = 4,
    max_block: int = 150,
    max_domain_block: int = 2000,
    n_rare: int = 3,
) -> tuple[set[tuple[int, int]], BlockingStats]:
    df: Counter[str] = Counter()
    for r in records:
        df.update(r.name.tokens)
    blocks: dict[str, list[int]] = defaultdict(list)
    for pos, r in enumerate(records):
        for key in blocking_keys(r, df, prefix_len, n_rare):
            blocks[key].append(pos)
    pairs: set[tuple[int, int]] = set()
    skipped = 0
    for key, members in blocks.items():
        if len(members) < 2:
            continue
        cap = max_domain_block if key.startswith("d:") else max_block
        if len(members) > cap:
            skipped += 1
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pairs.add((a, b) if a < b else (b, a))
    return pairs, BlockingStats(len(records), len(blocks), skipped, len(pairs))
