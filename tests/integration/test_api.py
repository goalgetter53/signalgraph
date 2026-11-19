from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signalgraph.api.app import create_app
from signalgraph.connectors import PoliteClient
from signalgraph.storage import Repository
from tests.integration.test_pipeline import _full_run


@pytest.fixture
async def db_path(tmp_path: Path, offline_client: PoliteClient) -> Path:
    path = tmp_path / "api.duckdb"
    repo = Repository(path)
    await _full_run(repo, offline_client)
    repo.close()
    return path


@pytest.fixture
def client(db_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(db_path)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["signals"] == 35
    assert "edgar:form_d" in body["watermarks"]


def test_accounts_ranked_and_filterable(client: TestClient) -> None:
    rows = client.get("/accounts").json()
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    icp = client.get("/accounts", params={"icp_only": True}).json()
    assert len(icp) == len(rows) - 1
    assert client.get("/accounts", params={"limit": 0}).status_code == 422


def test_account_detail_and_explain(client: TestClient) -> None:
    top = client.get("/accounts", params={"limit": 1}).json()[0]
    detail = client.get(f"/accounts/{top['account_id']}").json()
    assert detail["account"]["domain"] == "northwindrobotics.com"
    assert len(detail["signals"]) == top["n_signals"]
    explain = client.get(f"/accounts/{top['account_id']}/explain").json()
    assert explain["score"] == top["score"]
    assert sum(c["points"] for c in explain["contributions"]) == pytest.approx(
        explain["raw_points"], abs=1e-3
    )
    assert {"age_days", "decay", "base_weight", "magnitude_factor"} <= set(
        explain["contributions"][0]
    )


def test_signals_and_404s(client: TestClient) -> None:
    hiring = client.get("/signals", params={"signal_type": "hiring"}).json()
    assert hiring
    assert all(s["signal_type"] == "hiring" for s in hiring)
    assert client.get("/accounts/nope").status_code == 404
    assert client.get("/accounts/nope/explain").status_code == 404
