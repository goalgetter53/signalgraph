import pytest

from signalgraph.resolution.domains import canonical_domain


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme.com", "acme.com"),
        ("https://www.acme.com/careers?utm_source=x", "acme.com"),
        ("HTTPS://WWW.ACME.COM:8443/", "acme.com"),
        ("jobs.acme.co.uk", "acme.co.uk"),
        ("koen.co.jp", "koen.co.jp"),
        ("hr@orbitly.io", "orbitly.io"),
        ("https://user:pw@app.acme.io/x", "acme.io"),
        # PSL private section: each GitHub Pages / Heroku tenant is its own site
        ("latticebloom.github.io", "latticebloom.github.io"),
        ("https://pinehollow.github.io/docs", "pinehollow.github.io"),
        ("api.myapp.herokuapp.com", "myapp.herokuapp.com"),
        # tenant platforms not on the PSL
        ("https://marigoldrye.substack.com/p/launch", "marigoldrye.substack.com"),
        ("www.substack.com", None),
        # redirect wrappers
        ("https://www.google.com/url?q=https://hyperionvale.com/pricing&sa=D", "hyperionvale.com"),
        (
            "https://l.facebook.com/l.php?u=https%3A%2F%2Fwww.hyperionvale.com%2F",
            "hyperionvale.com",
        ),
        # non-identifying hosts
        ("founders@gmail.com", None),
        ("boards.greenhouse.io/redwoodledger", None),
        ("https://jobs.lever.co/acme/123", None),
        ("linkedin.com/company/acme", None),
        # junk
        ("http://192.168.0.1/admin", None),
        ("localhost", None),
        ("co.uk", None),
        ("", None),
        (None, None),
        # IDN -> punycode
        ("bücher.de", "xn--bcher-kva.de"),
        ("https://www.BÜCHER.de/", "xn--bcher-kva.de"),
    ],
)
def test_canonical_domain(raw: str | None, expected: str | None) -> None:
    assert canonical_domain(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # locale / region sections of a parent's site are kept as scope
        ("https://www.microsoft.com/en-ca", "microsoft.com/en-ca"),
        ("https://news.microsoft.com/sv-se/", "microsoft.com/sv-se"),
        ("https://www.nintendo.com/hk/", "nintendo.com/hk"),
        ("http://poland.arcelormittal.com/", "arcelormittal.com/poland"),
        ("http://www.zeiss.com/microscopy", "zeiss.com/microscopy"),
        # generic hosts / paths / pure language codes are not a scope
        ("http://www.nintendo.com/", "nintendo.com"),
        ("https://www.acme.com/careers?utm_source=x", "acme.com"),
        ("jobs.acme.co.uk", "acme.co.uk"),
        ("https://www.siemens.com/de/", "siemens.com"),
        ("http://www.budvar.cz/index.html", "budvar.cz"),
        ("hr@orbitly.io", "orbitly.io"),
        # wrappers, tenants and non-identifying hosts behave like canonical_domain
        ("https://www.google.com/url?q=https://hyperionvale.com/pricing&sa=D", "hyperionvale.com"),
        ("https://marigoldrye.substack.com/p/launch", "marigoldrye.substack.com"),
        ("boards.greenhouse.io/redwoodledger", None),
        (None, None),
    ],
)
def test_site_key(raw: str | None, expected: str | None) -> None:
    from signalgraph.resolution.domains import registrable_of, site_key

    assert site_key(raw) == expected
    assert registrable_of(site_key(raw)) == canonical_domain(raw)
