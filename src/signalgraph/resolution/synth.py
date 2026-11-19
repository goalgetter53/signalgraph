"""Seeded synthetic company-record generator (FICTIONAL data).

Used for two things only:
1. training data for the pair scorer (so the hand-labelled gold set stays a
   clean held-out test set), and
2. the ER throughput benchmark at 10k / 50k records.

It produces realistic *noise* (legal suffixes, casing, typos, abbreviations,
URL formats, country spellings, missing fields) and realistic *hard
negatives* (sibling companies sharing a brand token, country subsidiaries
with their own ccTLD domain).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_ONSETS = [
    "b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "br", "cl", "dr", "fl", "gr", "kr", "pl", "pr", "st", "tr", "vr", "zh",
]  # fmt: skip
_VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "io"]
_CODAS = ["", "", "n", "r", "x", "s", "l", "m", "th", "nt", "rk", "ra", "ix", "on", "ly"]
_INDUSTRY = [
    "Robotics", "Analytics", "Labs", "Health", "Systems", "Software", "Technologies", "Bio",
    "Logistics", "Energy", "Capital", "AI", "Cloud", "Security", "Payments", "Networks",
    "Foods", "Motors", "Learning", "Data", "Devices", "Media",
]  # fmt: skip
_LEGAL = ["Inc", "Inc.", "Incorporated", "LLC", "Ltd", "Ltd.", "Limited", "Corp", "Corporation",
          "GmbH", "Co.", "PLC", "S.A.", "B.V.", "Pty Ltd"]  # fmt: skip
_ABBREV = {"Technologies": "Tech", "Laboratories": "Labs", "International": "Intl",
           "Systems": "Sys", "Corporation": "Corp", "Analytics": "Analytix"}  # fmt: skip
_COUNTRIES = {
    "US": ["US", "USA", "United States", "us"],
    "GB": ["GB", "UK", "United Kingdom"],
    "DE": ["DE", "Germany", "Deutschland"],
    "IN": ["IN", "India"],
    "FR": ["FR", "France"],
    "CA": ["CA", "Canada"],
}
_CCTLD = {"US": "com", "GB": "co.uk", "DE": "de", "IN": "co.in", "FR": "fr", "CA": "ca"}
_SUBSIDIARY_COUNTRY = {"GB": "UK", "DE": "Deutschland", "IN": "India", "FR": "France",
                       "CA": "Canada"}  # fmt: skip


@dataclass(frozen=True)
class LabeledRecord:
    record_id: str
    entity_id: str
    name: str
    domain: str | None
    country: str | None
    aliases: tuple[str, ...] = ()


def _coined(rng: random.Random) -> str:
    syllables = rng.choice([2, 2, 2, 3])
    word = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(syllables))
    return (word + rng.choice(_CODAS)).capitalize()


def _typo(rng: random.Random, text: str) -> str:
    if len(text) < 5:
        return text
    i = rng.randrange(1, len(text) - 2)
    op = rng.choice(["swap", "drop", "dup"])
    if op == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    if op == "drop":
        return text[:i] + text[i + 1 :]
    return text[:i] + text[i] + text[i:]


def _name_variant(rng: random.Random, brand: str, industry: str | None) -> str:
    words = [brand] + ([industry] if industry else [])
    if industry and rng.random() < 0.15:
        words = [brand]
    if rng.random() < 0.15:
        words = [_ABBREV.get(w, w) for w in words]
    name = " ".join(words)
    if len(words) > 1 and rng.random() < 0.05:  # "Nimbus Grid" -> "NimbusGrid"
        name = "".join(words)
    if rng.random() < 0.12:
        name = _typo(rng, name)
    if rng.random() < 0.6:
        sep = rng.choice([" ", ", "])
        name = f"{name}{sep}{rng.choice(_LEGAL)}"
    if rng.random() < 0.05:
        name = "The " + name
    case = rng.random()
    if case < 0.1:
        name = name.upper()
    elif case < 0.15:
        name = name.lower()
    return name


def _domain_variant(rng: random.Random, domain: str) -> str | None:
    r = rng.random()
    if r < 0.4:
        return None
    if r < 0.43:
        return rng.choice(["gmail.com", "boards.greenhouse.io/x", "linkedin.com/company/x"])
    fmt = rng.choice(
        ["{d}", "www.{d}", "https://www.{d}/", "http://{d}/careers", "jobs.{d}", "{D}",
         "https://{d}/about?utm_source=x"]
    )  # fmt: skip
    return fmt.replace("{D}", domain.upper()).replace("{d}", domain)


def _country_variant(rng: random.Random, country: str) -> str | None:
    if rng.random() < 0.3:
        return None
    return rng.choice(_COUNTRIES[country])


def generate(
    n_entities: int,
    *,
    seed: int = 7,
    max_records_per_entity: int = 4,
    sibling_rate: float = 0.2,
    subsidiary_rate: float = 0.08,
    series_rate: float = 0.03,
) -> list[LabeledRecord]:
    rng = random.Random(seed)
    records: list[LabeledRecord] = []
    used_brands: set[str] = set()
    entity_no = 0

    def emit(brand: str, industry: str | None, domain: str, country: str, n: int) -> None:
        nonlocal entity_no
        entity_id = f"E{entity_no:06d}"
        entity_no += 1
        for k in range(n):
            records.append(
                LabeledRecord(
                    record_id=f"{entity_id}-{k}",
                    entity_id=entity_id,
                    name=_name_variant(rng, brand, industry),
                    domain=_domain_variant(rng, domain),
                    country=_country_variant(rng, country),
                )
            )

    while entity_no < n_entities:
        brand = _coined(rng)
        if brand in used_brands:
            continue
        used_brands.add(brand)
        industry = rng.choice([None, *_INDUSTRY])
        country = rng.choice(list(_COUNTRIES))
        slug = (brand + (industry or "")).lower()
        domain = f"{slug}.{_CCTLD[country]}"
        emit(brand, industry, domain, country, rng.randint(1, max_records_per_entity))
        if rng.random() < sibling_rate:  # hard negative: same brand, other industry
            other = rng.choice([i for i in _INDUSTRY if i != industry])
            emit(brand, other, f"{brand.lower()}{other.lower()}.com", country, rng.randint(1, 3))
        if rng.random() < series_rate:  # hard negative: numbered fund / SPV series
            kind = rng.choice(["Capital Fund", "Partners", "Ventures Fund", "Holdings"])
            numerals = rng.choice([["I", "II", "III"], ["16", "17", "18"], ["2", "3"]])
            for num in rng.sample(numerals, k=2):
                emit(brand, f"{kind} {num}", f"{slug}{num.lower()}.com", country, rng.randint(1, 2))
        if rng.random() < subsidiary_rate:  # hard negative: country subsidiary
            sub = rng.choice(list(_SUBSIDIARY_COUNTRY))
            label = (
                f"{industry} {_SUBSIDIARY_COUNTRY[sub]}" if industry else _SUBSIDIARY_COUNTRY[sub]
            )
            emit(brand, label, f"{slug}.{_CCTLD[sub]}", sub, rng.randint(1, 2))
    return records
