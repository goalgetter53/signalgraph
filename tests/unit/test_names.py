import pytest

from signalgraph.resolution.names import (
    normalize_company,
    normalize_country,
    normalize_name,
    split_aliases,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Northwind Robotics, Inc.", "northwind robotics"),
        ("NORTHWIND ROBOTICS INCORPORATED", "northwind robotics"),
        ("Café Lumière S.A.S.", "cafe lumiere"),
        ("Müller & Söhne Präzision GmbH", "muller and sohne prazision"),
        ("KOEN SYSTEMS KABUSHIKI KAISHA", "koen systems"),
        ("Kōen Systems K.K.", "koen systems"),
        ("Helio & Co Ltd", "helio"),
        ("Helio & Company Limited", "helio"),
        ("The Paper Crane Company", "paper crane"),
        ("Orbitly.io", "orbitly"),
        ("Ostrava Mobility s.r.o.", "ostrava mobility"),
        ("IRONBARK ANALYTICS PTY. LTD.", "ironbark analytics"),
        ("Kestrel Data Systems Private Limited", "kestrel data systems"),
        ("Sable + Finch LLP", "sable and finch"),
        ("Inc", "inc"),  # never strip a name down to nothing
        ("", ""),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    ("raw", "primary", "aliases"),
    [
        ("Zentro Inc. (f/k/a Acornfield Labs, Inc.)", "Zentro Inc.", ["Acornfield Labs, Inc."]),
        ("Payloom, Inc. (formerly Brightline Payroll)", "Payloom, Inc.", ["Brightline Payroll"]),
        ("Acme Holdings d/b/a Acme Cloud", "Acme Holdings", ["Acme Cloud"]),
        ("Plain Name LLC", "Plain Name LLC", []),
    ],
)
def test_split_aliases(raw: str, primary: str, aliases: list[str]) -> None:
    assert split_aliases(raw) == (primary, aliases)


def test_normalize_company_merges_extra_aliases_and_dedupes() -> None:
    n = normalize_company("Zentro, Inc.", ["Acornfield Labs", "ZENTRO INC"])
    assert n.primary == "zentro"
    assert n.aliases == ("acornfield labs",)
    assert n.tokens == frozenset({"zentro", "acornfield", "labs"})


@pytest.mark.parametrize(
    ("raw", "iso"),
    [("United States", "US"), ("usa", "US"), ("UK", "GB"), ("Deutschland", "DE"),
     ("in", "IN"), ("Atlantis", None), (None, None), ("", None)],
)  # fmt: skip
def test_normalize_country(raw: str | None, iso: str | None) -> None:
    assert normalize_country(raw) == iso


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Crown Resorts Public Limited Company", "crown resorts"),
        ("Acme Holdings PLC", "acme holdings"),
        ("Gazprom Neft Public Joint Stock Company", "gazprom neft"),
        ("PT Bumi Resources Tbk", "bumi resources"),
        ("Grupo Bimbo S.A.B. de C.V.", "grupo bimbo"),
        ("Volvo AB (publ)", "volvo"),
        ("OOO Yandex", "yandex"),
        ("Public Storage", "public storage"),  # "public" alone is part of the name
        ("Open Text", "open text"),
        ("Stock", "stock"),
    ],
)
def test_v2_legal_forms(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected
    assert normalize_name(expected) == expected or normalize_name(normalize_name(raw)) == expected
