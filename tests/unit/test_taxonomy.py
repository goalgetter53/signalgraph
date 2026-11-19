import pytest

from signalgraph.domain.taxonomy import ROLE_FAMILIES, classify_news, classify_role


@pytest.mark.parametrize(
    ("title", "dept", "family"),
    [
        ("Enterprise Account Executive", None, "gtm"),
        ("Sales Development Representative", None, "gtm"),
        ("RevOps Manager", None, "gtm"),
        ("Solutions Engineer", None, "gtm"),
        ("Senior Data Engineer", None, "data"),
        ("Machine Learning Engineer", None, "data"),
        ("Senior Backend Engineer", None, "eng"),
        ("Head of Security", None, "security"),
        ("Product Designer", None, "product"),
        ("VP Finance", None, "leadership"),
        ("Office Manager", None, "ops"),
        ("Team Member", "Customer Success", "gtm"),
        ("Barista", None, "other"),
    ],
)
def test_classify_role(title: str, dept: str | None, family: str) -> None:
    assert classify_role(title, dept) == family
    assert family in ROLE_FAMILIES


@pytest.mark.parametrize(
    ("headline", "label"),
    [
        ("Northwind raises $12M Series A", "funding"),
        ("Acme appoints first Chief Revenue Officer", "leadership_change"),
        ("Acme opens new office in Denver", "expansion"),
        ("Acme launches Insights", "product_launch"),
        ("Acme partners with Globex", "partnership"),
        ("Acme acquires Initech", "acquisition"),
        ("Our holiday schedule", "general"),
    ],
)
def test_classify_news(headline: str, label: str) -> None:
    assert classify_news(headline) == label
