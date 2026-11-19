"""Rule-based classifiers for job titles (role family) and news headlines.

Deliberately transparent keyword rules: a GTM team must be able to read *why* a
posting counted as a GTM hire. Order matters: the first family whose pattern
matches wins, so more specific families come first (``data`` before ``eng``
so that "Data Engineer" is a data hire, ``gtm`` before ``leadership``...).
"""

from __future__ import annotations

import re
from typing import Final

_ROLE_RULES: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "gtm",
        re.compile(
            r"\b(account executive|ae\b|sdr|bdr|sales|revops|revenue operations|gtm|"
            r"go[- ]to[- ]market|growth marketing|demand gen|marketing|customer success|"
            r"partnerships?|solutions? (engineer|consultant)|sales engineer|business development)",
            re.I,
        ),
    ),
    (
        "data",
        re.compile(
            r"\b(data (engineer|scientist|analyst|platform)|analytics|machine learning|ml engineer|"
            r"\bai\b|llm|applied scientist|bi developer|data)",
            re.I,
        ),
    ),
    (
        "security",
        re.compile(r"\b(security|secops|soc analyst|appsec|infosec|abuse investigator)", re.I),
    ),
    (
        "eng",
        re.compile(
            r"\b(engineer|developer|sre|devops|platform|backend|frontend|full[- ]?stack|"
            r"software|architect|infrastructure|mobile|ios|android)",
            re.I,
        ),
    ),
    ("product", re.compile(r"\b(product manager|product owner|designer|ux|ui|research)", re.I)),
    (
        "leadership",
        re.compile(r"\b(chief|vp|vice president|head of|director|cxo|ceo|cto|cfo|coo)", re.I),
    ),
    (
        "ops",
        re.compile(
            r"\b(finance|accountant|legal|counsel|people|recruit|talent|hr\b|operations|office)",
            re.I,
        ),
    ),
]

ROLE_FAMILIES: Final[tuple[str, ...]] = (*(name for name, _ in _ROLE_RULES), "other")


def classify_role(title: str, department: str | None = None) -> str:
    """Return the role family for a job title (department used as a fallback hint)."""
    for text in (title, department or ""):
        for family, pattern in _ROLE_RULES:
            if pattern.search(text):
                return family
    return "other"


_NEWS_RULES: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        "funding",
        re.compile(
            r"\b(raises?|raised|funding|series [a-f]\b|seed round|investment|backed by|valuation)",
            re.I,
        ),
    ),
    (
        "leadership_change",
        re.compile(r"\b(appoints?|names?|hires?|joins as|new (ceo|cto|cfo|cro|vp))", re.I),
    ),
    (
        "expansion",
        re.compile(r"\b(expands?|expansion|opens? (new )?office|enters|launch(es)? in\b)", re.I),
    ),
    ("product_launch", re.compile(r"\b(launch(es|ed)?|introduces?|unveils?|announces?)", re.I)),
    ("partnership", re.compile(r"\b(partners? with|partnership|integration with)", re.I)),
    ("acquisition", re.compile(r"\b(acquires?|acquired|acquisition|merger)", re.I)),
]


def classify_news(headline: str) -> str:
    for label, pattern in _NEWS_RULES:
        if pattern.search(headline):
            return label
    return "general"
