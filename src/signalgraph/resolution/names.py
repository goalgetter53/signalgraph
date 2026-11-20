"""Company-name normalisation.

Goal: two strings that a human would call "the same name" map to the same
token sequence, while information that distinguishes companies (``Acme
Robotics`` vs ``Acme Analytics``, ``Acme Japan``) survives.

Steps: alias extraction (``f/k/a``, ``d/b/a``, ``formerly``) -> NFKC + casefold
-> accent stripping -> ``&``/``+`` to ``and`` -> drop trailing TLDs in names
(``Monday.com``) -> collapse abbreviation dots (``S.A.``) -> non-alphanumerics
to spaces -> drop leading ``the`` -> strip trailing legal-form tokens (and the qualifiers of a
multi-word form: "Public Limited Company", "Joint Stock Company") -> strip a leading
legal form ("PT", "OOO").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

LEGAL_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "co", "company", "cos",
        "llc", "llp", "lp", "ltd", "limited", "plc", "pllc",
        "gmbh", "mbh", "ag", "kg", "ug", "se", "ev",
        "sa", "sas", "sarl", "srl", "spa", "sl", "sau", "sca", "nv", "bv", "cv",
        "oy", "oyj", "ab", "as", "asa", "aps", "a/s",
        "kk", "kabushiki", "kaisha", "gk", "pty", "pvt", "pte", "private", "sdn", "bhd",
        "zoo", "sp", "z", "oo", "sro", "ooo", "pjsc",
        # v2 (real-data error analysis): more jurisdictions' legal forms
        "tbk", "ltee", "ltda", "publ", "jsc", "ojsc", "cjsc", "pao", "oao", "zao",
        "kft", "zrt", "nyrt", "doo", "dd", "sae", "saa", "sab", "spolka", "akcyjna",
    }
)  # fmt: skip

# Qualifiers that are only legal-form words when a legal form follows them:
# "Public Limited Company", "Joint Stock Company", "Open Joint-Stock Company".
LEGAL_QUALIFIERS: Final[frozenset[str]] = frozenset({"public", "joint", "stock", "open", "closed"})

# Legal forms written *before* the name: "PT Bumi Resources Tbk", "OOO Yandex".
LEGAL_PREFIXES: Final[frozenset[str]] = frozenset(
    {"pt", "ooo", "oao", "pao", "zao", "pjsc", "ojsc", "jsc", "cjsc"}
)

# Connectors left dangling once a legal form is stripped ("Helio & Co", "Grupo X S.A. de C.V.").
_DANGLING: Final[frozenset[str]] = frozenset({"and", "de"})

_ALIAS_SPLIT: Final = re.compile(
    r"\s*(?:\(|,)?\s*\b(?:f/?k/?a|formerly(?:\s+known\s+as)?|d/?b/?a|doing\s+business\s+as|"
    r"a/?k/?a|also\s+known\s+as|trading\s+as|t/a)\b[\s:]*",
    re.IGNORECASE,
)
_NAME_TLD: Final = re.compile(r"\.(?:com|io|ai|co|net|org|app|dev|so|xyz)\b")
_ABBREV_DOT: Final = re.compile(r"(?<=\w)\.(?=\w)|(?<=\b\w)\.")
_NON_ALNUM: Final = re.compile(r"[^\w]+|_+")
_SPACES: Final = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def split_aliases(raw: str) -> tuple[str, list[str]]:
    """``"Zentro Inc (f/k/a Acme Labs)"`` -> ``("Zentro Inc", ["Acme Labs"])``."""
    parts = [p.strip(" ()[],;") for p in _ALIAS_SPLIT.split(raw)]
    parts = [p for p in parts if p]
    if not parts:
        return raw.strip(), []
    return parts[0], parts[1:]


def normalize_name(raw: str) -> str:
    """Canonical token string for one name (no alias handling). Idempotent."""
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = strip_accents(text)
    text = text.replace("&", " and ").replace("+", " and ")
    text = _NAME_TLD.sub(" ", text)
    text = _ABBREV_DOT.sub("", text)
    text = _NON_ALNUM.sub(" ", text)
    tokens = _SPACES.split(text.strip()) if text.strip() else []
    if len(tokens) > 1 and tokens[0] == "the":
        tokens = tokens[1:]
    # Strip trailing legal forms, plus a connector left dangling ("Helio & Co" -> "helio").
    stripped = False
    while len(tokens) > 1 and (
        tokens[-1] in LEGAL_SUFFIXES
        or tokens[-1] in _DANGLING
        or (stripped and tokens[-1] in LEGAL_QUALIFIERS)
    ):
        stripped = stripped or tokens[-1] in LEGAL_SUFFIXES
        tokens.pop()
    if len(tokens) > 1 and tokens[0] in LEGAL_PREFIXES:
        tokens = tokens[1:]
    return " ".join(tokens)


@dataclass(frozen=True, slots=True)
class NormalizedName:
    primary: str
    aliases: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return (self.primary, *self.aliases)

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(t for name in self.all for t in name.split())


def normalize_company(raw: str, extra_aliases: tuple[str, ...] | list[str] = ()) -> NormalizedName:
    primary_raw, alias_raw = split_aliases(raw)
    primary = normalize_name(primary_raw)
    seen = {primary}
    aliases: list[str] = []
    for alias in [*alias_raw, *extra_aliases]:
        norm = normalize_name(alias)
        if norm and norm not in seen:
            seen.add(norm)
            aliases.append(norm)
    return NormalizedName(primary=primary, aliases=tuple(aliases))


_COUNTRIES: Final[dict[str, str]] = {
    "united states": "US", "united states of america": "US", "usa": "US", "us": "US",
    "america": "US", "united kingdom": "GB", "uk": "GB", "great britain": "GB",
    "england": "GB", "germany": "DE", "deutschland": "DE", "france": "FR", "india": "IN",
    "canada": "CA", "australia": "AU", "netherlands": "NL", "the netherlands": "NL",
    "spain": "ES", "italy": "IT", "ireland": "IE", "sweden": "SE", "denmark": "DK",
    "norway": "NO", "finland": "FI", "switzerland": "CH", "austria": "AT", "japan": "JP",
    "singapore": "SG", "brazil": "BR", "mexico": "MX", "israel": "IL", "poland": "PL",
    "portugal": "PT", "belgium": "BE", "new zealand": "NZ", "south korea": "KR",
    "korea": "KR", "china": "CN", "united arab emirates": "AE", "uae": "AE",
    "romania": "RO", "croatia": "HR", "estonia": "EE",
}  # fmt: skip


def normalize_country(value: str | None) -> str | None:
    """Map a country name or ISO code to ISO 3166-1 alpha-2 (``None`` if unknown)."""
    if not value:
        return None
    text = strip_accents(value).strip().casefold().replace(".", "")
    if not text:
        return None
    if text in _COUNTRIES:
        return _COUNTRIES[text]
    if len(text) == 2 and text.isalpha():
        return "GB" if text == "uk" else text.upper()
    return None
