"""Real-data ER benchmark built from Wikidata (CC0).

Pure functions only: parsing SPARQL JSON bindings into per-company facts, turning
those facts into labelled "source-like" records, and the pairwise bootstrap used
for confidence intervals. Network I/O lives in ``benchmarks/real/``.

Ground truth is the Wikidata QID. Records are built from real strings only (no
injected typos or synthetic noise):

* ``careers``  - English label + official website (like a Greenhouse board),
* ``registry`` - current official/legal name (P1448) + country, no website
  (like an SEC Form D filing),
* ``news``     - one English alias, no website, no country (like a news mention),
* ``watchlist`` - a former official name (P1448 with an end date), nothing else
  (like a stale CRM row after a rebrand).

Hard negatives come from the sampling, not from the record builder: parent and
subsidiary companies (P749) are both included as separate entities, and other
companies that share an exact English label are added on purpose.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

USER_AGENT = "signalgraph-benchmark/0.1 (+https://github.com/goalgetter53/signalgraph)"
SPARQL_URL = "https://query.wikidata.org/sparql"

# business, enterprise, company, public company, subsidiary, corporation,
# privately held company, technology company, software company
COMPANY_CLASSES: tuple[str, ...] = (
    "Q4830453", "Q6881511", "Q783794", "Q891723", "Q658255", "Q167037",
    "Q1589009", "Q18388277", "Q1058914",
)  # fmt: skip

VIEWS: tuple[str, ...] = ("careers", "registry", "news", "watchlist")

_DEPRECATED = "http://wikiba.se/ontology#DeprecatedRank"
_PREFERRED = "http://wikiba.se/ontology#PreferredRank"


@dataclass
class WikidataCompany:
    qid: str
    label: str | None = None
    aliases: set[str] = field(default_factory=set)
    official_names: set[str] = field(default_factory=set)
    former_names: set[str] = field(default_factory=set)
    preferred_websites: set[str] = field(default_factory=set)
    websites: set[str] = field(default_factory=set)
    countries: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)

    def website(self) -> str | None:
        pool = self.preferred_websites or self.websites
        return min(pool) if pool else None


@dataclass(frozen=True)
class GoldRow:
    record_id: str
    entity_id: str
    name: str
    domain: str | None
    country: str | None
    source: str
    parents: tuple[str, ...]
    note: str
    seed: str = ""


def qid_from_uri(uri: str) -> str:
    """``http://www.wikidata.org/entity/Q95`` -> ``Q95``."""
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _value(binding: Mapping[str, Any], key: str) -> str | None:
    cell = binding.get(key)
    if not isinstance(cell, Mapping):
        return None
    value = cell.get("value")
    return value if isinstance(value, str) and value.strip() else None


def bindings(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``results.bindings`` list of a SPARQL JSON response (empty if malformed)."""
    results = payload.get("results")
    rows = results.get("bindings") if isinstance(results, Mapping) else None
    return [dict(r) for r in rows if isinstance(r, Mapping)] if isinstance(rows, list) else []


def is_latin(text: str) -> bool:
    """True if every letter is Latin script (and there is at least one letter)."""
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(ch, "") for ch in letters)


def merge_bindings(
    companies: dict[str, WikidataCompany], kind: str, rows: Iterable[Mapping[str, Any]]
) -> None:
    """Fold one property query's rows into ``companies`` (keyed by QID).

    Expected variables per ``kind``: ``c`` always, plus
    label: ``l``; alias: ``a``; official: ``n``, optional ``end`` and ``rank``;
    website: ``w``, optional ``end`` and ``rank``; country: ``iso``; parent: ``p``.
    """
    for row in rows:
        uri = _value(row, "c")
        if uri is None:
            continue
        qid = qid_from_uri(uri)
        co = companies.setdefault(qid, WikidataCompany(qid))
        rank = _value(row, "rank")
        if rank == _DEPRECATED:
            continue
        ended = _value(row, "end") is not None
        if kind == "label":
            label = _value(row, "l")
            if label:
                co.label = label.strip()
        elif kind == "alias":
            alias = _value(row, "a")
            if alias:
                co.aliases.add(alias.strip())
        elif kind == "official":
            name = _value(row, "n")
            if name:
                (co.former_names if ended else co.official_names).add(name.strip())
        elif kind == "website":
            url = _value(row, "w")
            if url and not ended:
                (co.preferred_websites if rank == _PREFERRED else co.websites).add(url.strip())
        elif kind == "country":
            iso = _value(row, "iso")
            if iso:
                co.countries.add(iso.strip().upper())
        elif kind == "parent":
            parent = _value(row, "p")
            if parent:
                co.parents.add(qid_from_uri(parent))
        else:
            raise ValueError(f"unknown binding kind: {kind}")


def _pick(candidates: Iterable[str], used: set[str]) -> str | None:
    """First (alphabetical) Latin-script name whose casefolded form is not used yet."""
    for name in sorted(candidates):
        if len(name) >= 2 and is_latin(name) and name.casefold() not in used:
            return name
    return None


def build_rows(
    companies: Mapping[str, WikidataCompany], seeds: Mapping[str, str] | None = None
) -> list[GoldRow]:
    """Deterministic records per company; companies without an English label are skipped.
    ``seeds`` maps QID -> the sampling group it came from (reported per slice)."""
    seeds = seeds or {}
    rows: list[GoldRow] = []
    for qid in sorted(companies, key=lambda q: (len(q), q)):
        co = companies[qid]
        if not co.label or not is_latin(co.label):
            continue
        parents = tuple(sorted(co.parents))
        country = min(co.countries) if len(co.countries) == 1 else None
        used = {co.label.casefold()}
        views: list[tuple[str, str, str | None, str | None, str]] = [
            ("careers", co.label, co.website(), None, "en label + official website (P856)")
        ]
        legal = _pick(co.official_names, used)
        if legal:
            used.add(legal.casefold())
            views.append(("registry", legal, None, country, "current official name (P1448)"))
        alias = _pick(co.aliases, used)
        if alias:
            used.add(alias.casefold())
            views.append(("news", alias, None, None, "en alias"))
        former = _pick(co.former_names, used)
        if former:
            views.append(("watchlist", former, None, None, "former official name (P1448 + P582)"))
        for view, name, domain, ctry, note in views:
            rows.append(
                GoldRow(
                    f"{qid}-{view}",
                    qid,
                    name,
                    domain,
                    ctry,
                    view,
                    parents,
                    note,
                    seeds.get(qid, ""),
                )
            )
    return rows


# ---------------------------------------------------------------- evaluation helpers


def pair_components(n: int, edge_sets: Sequence[set[tuple[int, int]]]) -> list[int]:
    """Connected-component id per record over the union of several pair sets."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edges in edge_sets:
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n)]


def component_counts(
    components: Sequence[int], predicted: set[tuple[int, int]], truth: set[tuple[int, int]]
) -> dict[int, tuple[int, int, int]]:
    """(tp, fp, fn) per component; every pair lies inside one component by construction."""
    counts: dict[int, list[int]] = {}
    for pair in predicted | truth:
        c = components[pair[0]]
        slot = counts.setdefault(c, [0, 0, 0])
        if pair in predicted and pair in truth:
            slot[0] += 1
        elif pair in predicted:
            slot[1] += 1
        else:
            slot[2] += 1
    return {c: (v[0], v[1], v[2]) for c, v in counts.items()}


def _prf_arrays(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, ...]:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 1.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def bootstrap_ci(
    per_method: Mapping[str, Mapping[int, tuple[int, int, int]]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap over connected components (shared across methods, so the
    F1 difference between the first two methods gets a paired CI as well)."""
    comps = sorted({c for counts in per_method.values() for c in counts})
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(comps), size=(n_resamples, len(comps))) if comps else None
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, Any] = {"n_components": len(comps), "n_resamples": n_resamples, "seed": seed}
    f1s: dict[str, np.ndarray] = {}
    for name, counts in per_method.items():
        mat = np.array([counts.get(c, (0, 0, 0)) for c in comps], dtype=float).reshape(-1, 3)
        if idx is None:
            continue
        sums = mat[idx].sum(axis=1)
        p, r, f = _prf_arrays(sums[:, 0], sums[:, 1], sums[:, 2])
        f1s[name] = f
        out[name] = {
            k: [round(float(np.percentile(v, lo)), 4), round(float(np.percentile(v, hi)), 4)]
            for k, v in (("precision", p), ("recall", r), ("f1", f))
        }
    names = list(f1s)
    if len(names) >= 2:
        diff = f1s[names[0]] - f1s[names[1]]
        out[f"f1_diff_{names[0]}_minus_{names[1]}"] = [
            round(float(np.percentile(diff, lo)), 4),
            round(float(np.percentile(diff, hi)), 4),
        ]
    return out
