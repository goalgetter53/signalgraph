"""Offline tests for the Wikidata benchmark parsing / record building (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from signalgraph.benchmarks.wikidata import (
    WikidataCompany,
    bindings,
    bootstrap_ci,
    build_rows,
    component_counts,
    is_latin,
    merge_bindings,
    pair_components,
    qid_from_uri,
)

E = "http://www.wikidata.org/entity/"
PREFERRED = "http://wikiba.se/ontology#PreferredRank"
NORMAL = "http://wikiba.se/ontology#NormalRank"
DEPRECATED = "http://wikiba.se/ontology#DeprecatedRank"


def _uri(v: str) -> dict[str, str]:
    return {"type": "uri", "value": v}


def _lit(v: str, lang: str | None = None) -> dict[str, str]:
    cell = {"type": "literal", "value": v}
    if lang:
        cell["xml:lang"] = lang
    return cell


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"head": {"vars": []}, "results": {"bindings": rows}}


def _fixture() -> dict[str, WikidataCompany]:
    companies: dict[str, WikidataCompany] = {}
    merge_bindings(
        companies,
        "label",
        bindings(
            _payload(
                [
                    {"c": _uri(E + "Q1"), "l": _lit("Acme", "en")},
                    {"c": _uri(E + "Q2"), "l": _lit("Acme India", "en")},
                    {"c": _uri(E + "Q3"), "l": _lit("トヨタ", "en")},
                ]
            )
        ),
    )
    merge_bindings(
        companies,
        "official",
        [
            {"c": _uri(E + "Q1"), "n": _lit("Acme Corporation", "en"), "rank": _uri(NORMAL)},
            {
                "c": _uri(E + "Q1"),
                "n": _lit("Acme Widgets Inc.", "en"),
                "rank": _uri(NORMAL),
                "end": _lit("2019-01-01T00:00:00Z"),
            },
            {"c": _uri(E + "Q1"), "n": _lit("Wrong Name", "en"), "rank": _uri(DEPRECATED)},
            {"c": _uri(E + "Q2"), "n": _lit("ACME", "en"), "rank": _uri(NORMAL)},
        ],
    )
    merge_bindings(
        companies,
        "website",
        [
            {"c": _uri(E + "Q1"), "w": _uri("https://zz-old.example"), "rank": _uri(NORMAL)},
            {"c": _uri(E + "Q1"), "w": _uri("https://acme.example"), "rank": _uri(PREFERRED)},
            {
                "c": _uri(E + "Q2"),
                "w": _uri("https://acme.example/in"),
                "rank": _uri(NORMAL),
                "end": _lit("2020-01-01T00:00:00Z"),
            },
        ],
    )
    merge_bindings(
        companies,
        "alias",
        [
            {"c": _uri(E + "Q1"), "a": _lit("ACME", "en")},  # same as label after casefold
            {"c": _uri(E + "Q1"), "a": _lit("Acme Co", "en")},
        ],
    )
    merge_bindings(companies, "country", [{"c": _uri(E + "Q2"), "iso": _lit("in")}])
    merge_bindings(companies, "parent", [{"c": _uri(E + "Q2"), "p": _uri(E + "Q1")}])
    return companies


def test_parsing_ranks_end_dates_and_fields() -> None:
    co = _fixture()
    assert qid_from_uri(E + "Q95") == "Q95"
    assert co["Q1"].official_names == {"Acme Corporation"}  # deprecated dropped
    assert co["Q1"].former_names == {"Acme Widgets Inc."}  # P582 end date -> former
    assert co["Q1"].website() == "https://acme.example"  # preferred rank wins
    assert co["Q2"].website() is None  # ended website ignored
    assert co["Q2"].countries == {"IN"}
    assert co["Q2"].parents == {"Q1"}
    assert bindings({"results": "garbage"}) == []
    with pytest.raises(ValueError, match="unknown binding kind"):
        merge_bindings(co, "nope", [{"c": _uri(E + "Q1")}])


def test_build_rows_views_dedup_and_script_filter() -> None:
    rows = build_rows(_fixture(), {"Q1": "renamed", "Q2": "subsidiary_or_parent"})
    by_id = {r.record_id: r for r in rows}
    assert set(by_id) == {
        "Q1-careers",
        "Q1-registry",
        "Q1-news",
        "Q1-watchlist",
        "Q2-careers",
        "Q2-registry",
    }  # Q3 has a non-Latin label -> skipped
    assert by_id["Q1-careers"].domain == "https://acme.example"
    assert by_id["Q1-registry"].domain is None
    assert by_id["Q1-news"].name == "Acme Co"
    assert by_id["Q1-watchlist"].name == "Acme Widgets Inc."
    assert by_id["Q2-registry"].country == "IN"
    assert by_id["Q2-careers"].parents == ("Q1",)
    assert by_id["Q2-careers"].seed == "subsidiary_or_parent"
    assert is_latin("Česká spořitelna")
    assert not is_latin("トヨタ")
    assert not is_latin("123")


def test_component_bootstrap_is_paired_and_bounded() -> None:
    truth = {(0, 1), (2, 3)}
    perfect, half = set(truth), {(0, 1), (3, 4)}
    comps = pair_components(5, [truth, perfect, half])
    assert comps[2] == comps[3] == comps[4] != comps[0]
    counts = {
        "perfect": component_counts(comps, perfect, truth),
        "half": component_counts(comps, half, truth),
    }
    assert sum(fp for _, fp, _ in counts["half"].values()) == 1
    ci = bootstrap_ci(counts, n_resamples=200, seed=1)
    assert ci["perfect"]["f1"] == [1.0, 1.0]
    lo, hi = ci["f1_diff_perfect_minus_half"]
    assert 0.0 <= lo <= hi <= 1.0
