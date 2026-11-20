"""Build a real, labelled company-matching gold set from Wikidata (CC0).

Steps (every SPARQL response is cached under ``data/real/raw/`` and reused, so a
re-run is offline and reproducible; delete the cache to re-pull):

1. Seeds: (a) companies with a parent company (P749), both with a website, and the
   parents themselves; (b) businesses with a *former* official name (P1448 with an
   end date, i.e. a rebrand).
2. Hard negatives: other companies whose English label is exactly the same as a
   seed's label (same name, different QID).
3. Details per QID: English label and aliases, official names (current/former),
   official websites, country ISO code, parents.
4. ``signalgraph.benchmarks.wikidata.build_rows`` turns that into records.

Writes ``data/real/wikidata_companies_gold.csv`` and ``data/real/wikidata_meta.json``.

    python benchmarks/real/build_wikidata_gold.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalgraph.benchmarks.wikidata import (
    COMPANY_CLASSES,
    SPARQL_URL,
    USER_AGENT,
    WikidataCompany,
    bindings,
    build_rows,
    merge_bindings,
    qid_from_uri,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "real" / "raw"
GOLD = ROOT / "data" / "real" / "wikidata_companies_gold.csv"
META = ROOT / "data" / "real" / "wikidata_meta.json"

N_SUBSIDIARY_ROWS = 800
N_RENAMED = 700
N_GENERIC = 700
BATCH = 200

_CLASSES = " ".join(f"wd:{q}" for q in COMPANY_CLASSES)

SEED_SUBSIDIARIES = f"""
SELECT DISTINCT ?c ?p WHERE {{
  VALUES ?t {{ {_CLASSES} }}
  ?c wdt:P31 ?t ; wdt:P749 ?p ; wdt:P856 [] .
  ?p wdt:P856 [] .
}} LIMIT {N_SUBSIDIARY_ROWS}"""

SEED_RENAMED = f"""
SELECT DISTINCT ?c WHERE {{
  ?st pq:P582 ?e . ?c p:P1448 ?st . ?c wdt:P856 [] . ?c wdt:P31 wd:Q4830453 .
}} LIMIT {N_RENAMED}"""

SEED_GENERIC = f"""
SELECT DISTINCT ?c WHERE {{
  ?c wdt:P31 wd:Q4830453 ; wdt:P856 [] ; wdt:P1448 [] .
}} LIMIT {N_GENERIC}"""

HOMONYMS = """
SELECT DISTINCT ?c ?l WHERE {{
  VALUES ?l {{ {labels} }}
  VALUES ?t {{ {classes} }}
  ?c rdfs:label ?l ; wdt:P31 ?t .
}}"""

DETAILS: dict[str, str] = {
    "label": "?c rdfs:label ?l . FILTER(LANG(?l) = 'en')",
    "alias": "?c skos:altLabel ?a . FILTER(LANG(?a) = 'en')",
    "official": "?c p:P1448 ?st . ?st ps:P1448 ?n ; wikibase:rank ?rank . "
    "OPTIONAL { ?st pq:P582 ?end }",
    "website": "?c p:P856 ?st . ?st ps:P856 ?w ; wikibase:rank ?rank . "
    "OPTIONAL { ?st pq:P582 ?end }",
    "country": "?c wdt:P17 ?k . ?k wdt:P297 ?iso .",
    "parent": "?c wdt:P749 ?p .",
}


def sparql(query: str) -> dict[str, Any]:
    """POST one query (stdlib client; cached by query hash)."""
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    path = RAW / f"{key}.json"
    if path.exists():
        cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"), strict=False)
        return cached
    body = urllib.parse.urlencode({"query": query}).encode()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"}
    for attempt in range(5):
        req = urllib.request.Request(SPARQL_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"), strict=False)
        except urllib.error.HTTPError as exc:
            wait = float(exc.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"  HTTP {exc.code}, retrying in {wait:.0f}s")
            time.sleep(min(wait, 60))
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  {exc!r}, retrying")
            time.sleep(2 ** (attempt + 1))
            continue
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(1.0)  # be polite to the public endpoint
        return payload
    raise RuntimeError(f"SPARQL failed after retries: {query[:80]}")


def _qids(qids: list[str]) -> str:
    return " ".join(f"wd:{q}" for q in qids)


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False) + "@en"


def main() -> None:
    companies: dict[str, WikidataCompany] = {}
    seeds: dict[str, set[str]] = {
        "subsidiary_or_parent": set(),
        "renamed": set(),
        "generic": set(),
        "homonym": set(),
    }
    print("seed: subsidiaries + parents")
    for row in bindings(sparql(SEED_SUBSIDIARIES)):
        c, p = row["c"]["value"], row["p"]["value"]
        seeds["subsidiary_or_parent"] |= {qid_from_uri(c), qid_from_uri(p)}
    print("seed: renamed businesses")
    for row in bindings(sparql(SEED_RENAMED)):
        seeds["renamed"].add(qid_from_uri(row["c"]["value"]))
    print("seed: generic businesses")
    for row in bindings(sparql(SEED_GENERIC)):
        seeds["generic"].add(qid_from_uri(row["c"]["value"]))
    pool = sorted(seeds["subsidiary_or_parent"] | seeds["renamed"] | seeds["generic"])

    print(f"labels for {len(pool)} seed QIDs")
    for i in range(0, len(pool), BATCH):
        q = f"SELECT ?c ?l WHERE {{ VALUES ?c {{ {_qids(pool[i : i + BATCH])} }} "
        q += DETAILS["label"] + " }"
        merge_bindings(companies, "label", bindings(sparql(q)))
    seeded = set(pool)
    labels = sorted({co.label for co in companies.values() if co.label})
    print(f"homonyms for {len(labels)} labels")
    for i in range(0, len(labels), BATCH):
        chunk = " ".join(_literal(lab) for lab in labels[i : i + BATCH])
        q = HOMONYMS.format(labels=chunk, classes=_CLASSES)
        for row in bindings(sparql(q)):
            qid = qid_from_uri(row["c"]["value"])
            if qid not in seeded:
                seeds["homonym"].add(qid)
    pool = sorted(set(pool) | seeds["homonym"])
    seed_of = {q: g for g in reversed(list(seeds)) for q in seeds[g]}  # first group wins

    print(f"details for {len(pool)} QIDs")
    for kind, pattern in DETAILS.items():
        for i in range(0, len(pool), BATCH):
            q = f"SELECT * WHERE {{ VALUES ?c {{ {_qids(pool[i : i + BATCH])} }} {pattern} }}"
            merge_bindings(companies, kind, bindings(sparql(q)))

    companies = {q: companies.get(q, WikidataCompany(q)) for q in pool}
    rows = build_rows(companies, seed_of)
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    with GOLD.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["record_id", "entity_id", "name", "domain", "country", "aliases", "source",
             "parents", "seed", "note"]
        )  # fmt: skip
        for r in rows:
            writer.writerow(
                [r.record_id, r.entity_id, r.name, r.domain or "", r.country or "", "",
                 r.source, "|".join(r.parents), r.seed, r.note]
            )  # fmt: skip
    pulled = sorted(p.stat().st_mtime for p in RAW.glob("*.json"))
    meta = {
        "source": "Wikidata (https://www.wikidata.org), CC0 1.0",
        "endpoint": SPARQL_URL,
        "pulled_utc": datetime.fromtimestamp(pulled[0], UTC).strftime("%Y-%m-%d")
        if pulled
        else None,
        "seed_counts": {k: len(v) for k, v in seeds.items()},
        "qids_in_pool": len(pool),
        "entities_with_records": len({r.entity_id for r in rows}),
        "records": len(rows),
        "records_by_source": {
            v: sum(r.source == v for r in rows) for v in sorted({r.source for r in rows})
        },
        "entities_by_seed": {g: len({r.entity_id for r in rows if r.seed == g}) for g in seeds},
        "queries": {
            "subsidiaries": SEED_SUBSIDIARIES.strip(),
            "renamed": SEED_RENAMED.strip(),
            "generic": SEED_GENERIC.strip(),
        },
    }
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "queries"}, indent=2))
    print(f"wrote {GOLD} and {META}")


if __name__ == "__main__":
    main()
