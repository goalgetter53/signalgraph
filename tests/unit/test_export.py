import csv
import io

from signalgraph.services.export import CSV_FIELDS, to_clay_rows, to_csv, to_hubspot_batches

ROWS = [
    {
        "account_id": f"a{i}",
        "name": f"Co {i}",
        "domain": f"co{i}.com" if i % 2 else None,
        "country": "US",
        "score": 50.0 + i,
        "icp_fit": i % 3 != 0,
        "n_signals": i,
        "by_type": {"hiring": 1.5},
        "icp_reasons": [] if i % 3 else ["country XX not in ICP"],
    }
    for i in range(205)
]
TOP = {"a1": "Enterprise Account Executive"}


def test_csv_has_stable_header_and_rows() -> None:
    text = to_csv(ROWS[:3], TOP)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert tuple(rows[0].keys()) == CSV_FIELDS
    assert rows[1]["top_signal"] == "Enterprise Account Executive"
    assert rows[1]["hiring_points"] == "1.5"
    assert rows[0]["domain"] == ""


def test_clay_rows_are_flat_json_objects() -> None:
    rows = to_clay_rows(ROWS[:4], TOP)
    assert all(not isinstance(v, (dict, list)) for r in rows for v in r.values())
    assert rows[0]["icp_reasons"] == "country XX not in ICP"
    assert rows[0]["source"] == "signalgraph"


def test_hubspot_batches_chunk_at_100_and_flag_missing_domains() -> None:
    payload = to_hubspot_batches(ROWS, TOP)
    assert [len(b["inputs"]) for b in payload["batches"]] == [100, 100, 5]
    first = payload["batches"][0]["inputs"][1]
    assert first["idProperty"] == "signalgraph_account_id"
    assert first["id"] == "a1"
    assert all(isinstance(v, str) for v in first["properties"].values())
    assert "a0" in payload["without_domain"]
    assert "a1" not in payload["without_domain"]
