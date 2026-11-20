"""Domain canonicalisation: any URL-ish string -> registrable domain (eTLD+1).

Uses the Public Suffix List (bundled snapshot via ``tldextract``, *including*
the private section, so ``acme.github.io`` stays ``acme.github.io`` instead of
collapsing every GitHub Pages site into ``github.io``). Also:

* unwraps redirect wrappers (``google.com/url?q=``, ``l.facebook.com/l.php?u=``,
  Outlook safelinks, LinkedIn redirects),
* accepts emails and bare hosts, strips credentials/ports/paths,
* converts IDNs to punycode so ``bücher.de`` == ``xn--bcher-kva.de``,
* returns ``None`` for hosts that do not identify a company: free-mail,
  social networks and job-board hosts (``boards.greenhouse.io/acme`` says
  nothing about the domain of Acme).
"""

from __future__ import annotations

import ipaddress
from functools import lru_cache
from typing import Final
from urllib.parse import parse_qs, urlsplit

import tldextract

_EXTRACT: Final = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)

NON_IDENTIFYING: Final[frozenset[str]] = frozenset(
    {
        # free / personal mail
        "gmail.com", "googlemail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
        "icloud.com", "proton.me", "protonmail.com", "aol.com", "gmx.com", "zoho.com",
        # social / content platforms (path-based identity)
        "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com",
        "youtube.com", "medium.com", "crunchbase.com", "angel.co", "wellfound.com",
        # job boards and ATS hosts
        "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com", "smartrecruiters.com",
        "bamboohr.com", "myworkdayjobs.com", "recruitee.com", "jobvite.com", "indeed.com",
        # generic infrastructure
        "bit.ly", "lnkd.in", "t.co", "sec.gov",
    }
)  # fmt: skip

# Hosted platforms that are *not* on the PSL but give each tenant a subdomain.
SUBDOMAIN_TENANT_HOSTS: Final[frozenset[str]] = frozenset(
    {"substack.com", "webflow.io", "wixsite.com", "myshopify.com", "notion.site", "carrd.co"}
)

_REDIRECT_PARAMS: Final = ("url", "q", "u", "target", "dest", "destination", "redirect")
_REDIRECTORS: Final[frozenset[str]] = frozenset(
    {
        "google.com", "www.google.com", "l.facebook.com", "lm.facebook.com",
        "www.linkedin.com", "linkedin.com", "out.reddit.com", "href.li",
    }
)  # fmt: skip
_TENANT_SKIP: Final = frozenset({"www", "boards", "jobs", "careers", "app"})


def _host_of(value: str) -> tuple[str | None, str]:
    """Return (hostname, query) for a URL, bare host or email address."""
    value = value.strip().strip("<>\"'")
    if "@" in value and "://" not in value and "/" not in value:
        value = value.rsplit("@", 1)[1]
    if "://" not in value:
        value = "http://" + value.lstrip("/")
    try:
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError:
        return None, ""
    return host, parts.query


def _unwrap(host: str, query: str, depth: int) -> str | None:
    if depth >= 3:
        return None
    if host in _REDIRECTORS or host.endswith("safelinks.protection.outlook.com"):
        params = parse_qs(query)
        for key in _REDIRECT_PARAMS:
            if params.get(key):
                return canonical_domain(params[key][0], _depth=depth + 1)
    return ""  # sentinel: not a redirect


def _to_ascii(host: str) -> str | None:
    host = host.strip(".").lower()
    if not host:
        return None
    try:
        return ".".join(
            label.encode("idna").decode("ascii") if not label.isascii() else label
            for label in host.split(".")
        )
    except UnicodeError:
        return None


@lru_cache(maxsize=200_000)
def canonical_domain(raw: str | None, *, _depth: int = 0) -> str | None:
    """Canonical, company-identifying registrable domain for ``raw`` (or ``None``)."""
    if raw is None or not raw.strip():
        return None
    host, query = _host_of(raw)
    if not host:
        return None
    unwrapped = _unwrap(host, query, _depth)
    if unwrapped != "":
        return unwrapped
    ascii_host = _to_ascii(host)
    if ascii_host is None:
        return None
    try:
        ipaddress.ip_address(ascii_host)
        return None
    except ValueError:
        pass
    ext = _EXTRACT(ascii_host)
    # tldextract >= 5.3 renamed registered_domain -> top_domain_under_public_suffix
    attr = (
        "top_domain_under_public_suffix"
        if hasattr(ext, "top_domain_under_public_suffix")
        else "registered_domain"
    )
    registrable = str(getattr(ext, attr) or "")
    if not registrable:
        return None
    if registrable in SUBDOMAIN_TENANT_HOSTS:
        labels = [s for s in ext.subdomain.split(".") if s and s not in _TENANT_SKIP]
        return f"{labels[-1]}.{registrable}" if labels else None
    if registrable in NON_IDENTIFYING:
        return None
    return registrable


# --------------------------------------------------------------------- site scope
# A subsidiary often has no domain of its own and lists a section of the parent's
# site instead: ``microsoft.com/en-ca`` (Microsoft Canada), ``nintendo.com/hk/``,
# ``poland.arcelormittal.com``. Reducing these to the registrable domain makes
# siblings look like the same company. ``site_key`` keeps the regional/section
# scope: registrable domain + non-generic subdomain + first non-generic path segment.

_GENERIC_SUBDOMAINS: Final[frozenset[str]] = frozenset(
    {
        "www", "www1", "www2", "www3", "m", "mobile", "web", "home", "app", "apps",
        "jobs", "careers", "career", "boards", "about", "corporate", "corp", "company",
        "news", "newsroom", "press", "media", "ir", "investor", "investors", "blog",
        "shop", "store", "info", "global", "group", "en", "int", "international",
    }
)  # fmt: skip
_GENERIC_PATH: Final[frozenset[str]] = frozenset(
    {
        "careers", "career", "jobs", "job", "about", "about-us", "aboutus", "about_us",
        "company", "home", "index", "main", "web", "site", "contact", "en", "english",
        "global", "int", "international", "corporate", "default", "start", "welcome",
    }
)  # fmt: skip
# Pure language codes say which translation of the site you see, not which company:
# ``siemens.com/de`` and ``siemens.com/en`` are the same firm. Region-bearing codes
# (``en-ca``, ``fr-ma``) and country codes that are not common language codes stay.
_LANGUAGE_CODES: Final[frozenset[str]] = frozenset(
    {
        "en", "de", "fr", "es", "it", "nl", "pt", "ja", "zh", "ru", "sv", "pl", "ko",
        "ar", "tr", "da", "fi", "no", "nb", "cs", "hu", "el", "he", "ro", "uk", "id",
    }
)  # fmt: skip


def _path_scope(path: str) -> str | None:
    for segment in path.split("/"):
        seg = segment.strip().lower()
        if not seg:
            continue
        stem = seg.split(".", 1)[0]  # index.html, default.aspx
        if stem in _GENERIC_PATH or seg in _LANGUAGE_CODES:
            return None
        return seg
    return None


@lru_cache(maxsize=200_000)
def site_key(raw: str | None) -> str | None:
    """``canonical_domain`` plus the regional/section scope of the URL (or ``None``).

    >>> site_key("https://www.microsoft.com/en-ca")
    'microsoft.com/en-ca'
    >>> site_key("http://poland.arcelormittal.com/")
    'arcelormittal.com/poland'
    >>> site_key("https://www.acme.com/careers?utm_source=x")
    'acme.com'
    """
    registrable = canonical_domain(raw)
    if registrable is None or raw is None:
        return None
    value = raw.strip().strip("<>\"'")
    if "@" in value and "://" not in value and "/" not in value:
        return registrable
    if "://" not in value:
        value = "http://" + value.lstrip("/")
    try:
        parts = urlsplit(value)
        host = _to_ascii(parts.hostname or "")
    except ValueError:
        return registrable
    if not host:
        return registrable
    ext = _EXTRACT(host)
    attr = (
        "top_domain_under_public_suffix"
        if hasattr(ext, "top_domain_under_public_suffix")
        else "registered_domain"
    )
    if str(getattr(ext, attr) or "") != registrable:
        return registrable  # redirect wrapper or tenant host: no scope to keep
    scope = [s for s in ext.subdomain.split(".") if s and s not in _GENERIC_SUBDOMAINS]
    path = _path_scope(parts.path)
    if path:
        scope.append(path)
    return f"{registrable}/{'/'.join(scope)}" if scope else registrable


def registrable_of(site: str | None) -> str | None:
    """Inverse of the scope part of ``site_key``: ``microsoft.com/en-ca`` -> ``microsoft.com``."""
    return site.split("/", 1)[0] if site else None
