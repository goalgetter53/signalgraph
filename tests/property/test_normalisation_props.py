"""Property-based invariants for name and domain normalisation."""

import re
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from signalgraph.resolution.domains import canonical_domain
from signalgraph.resolution.names import LEGAL_SUFFIXES, normalize_company, normalize_name

ascii_names = st.text(alphabet=string.ascii_letters + string.digits + " .,&-'()+/", max_size=40)
words = st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=10).filter(
    lambda w: w not in LEGAL_SUFFIXES and w not in {"the", "and"}
)
brand_names = st.lists(words, min_size=1, max_size=4).map(" ".join)


@given(st.text(max_size=60))
@settings(max_examples=400)
def test_normalize_is_idempotent(raw: str) -> None:
    once = normalize_name(raw)
    assert normalize_name(once) == once


@given(st.text(max_size=60))
@settings(max_examples=400)
def test_output_charset_and_spacing(raw: str) -> None:
    out = normalize_name(raw)
    assert out == out.strip()
    assert "  " not in out
    assert "_" not in out
    assert all(ch.isalnum() or ch == " " for ch in out)
    assert out == out.casefold() or not out.isascii()


@given(ascii_names)
def test_case_and_whitespace_insensitive(raw: str) -> None:
    assert normalize_name(raw.upper()) == normalize_name(raw)
    assert normalize_name(f"  {raw}\t") == normalize_name(raw)


@given(
    brand_names,
    st.sampled_from(
        [
            "Inc",
            "Inc.",
            ", Inc.",
            "LLC",
            "Ltd",
            "Limited",
            "GmbH",
            "Corporation",
            "S.A.",
            "Pvt. Ltd.",
            "Pty Ltd",
            "K.K.",
        ]
    ),
)
def test_legal_suffix_invariance(brand: str, suffix: str) -> None:
    sep = "" if suffix.startswith(",") else " "
    assert normalize_name(f"{brand}{sep}{suffix}") == normalize_name(brand)


@given(brand_names, brand_names)
def test_alias_extraction(primary: str, old: str) -> None:
    n = normalize_company(f"{primary} Inc (formerly {old} LLC)")
    assert n.primary == normalize_name(primary)
    if normalize_name(old) != n.primary:
        assert normalize_name(old) in n.aliases


labels = st.text(alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=12)
tlds = st.sampled_from(["com", "io", "ai", "co.uk", "de", "com.au", "co.jp", "dev"])


@given(
    labels,
    tlds,
    st.sampled_from(["", "www.", "jobs.", "app.eu."]),
    st.sampled_from(["", "http://", "https://", "HTTPS://"]),
    st.sampled_from(["", "/", "/careers", "/about?utm_source=x#top"]),
)
def test_domain_canonical_form(label: str, tld: str, sub: str, scheme: str, path: str) -> None:
    expected = f"{label}.{tld}"
    if canonical_domain(expected) is None:  # e.g. a label that is itself a public suffix
        return
    assert canonical_domain(f"{scheme}{sub}{label}.{tld}{path}") == expected


@given(st.text(max_size=80))
@settings(max_examples=400)
def test_domain_idempotent_and_total(raw: str) -> None:
    out = canonical_domain(raw)  # must never raise
    if out is not None:
        assert canonical_domain(out) == out
        assert out == out.lower()
        assert re.fullmatch(r"[a-z0-9.\-]+", out)
