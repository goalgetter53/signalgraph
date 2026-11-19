import csv
import io
import json
from pathlib import Path

from click.testing import CliRunner

from signalgraph.cli import cli
from tests.conftest import FIXTURES


def test_cli_end_to_end(tmp_path: Path) -> None:
    db = str(tmp_path / "cli.duckdb")
    runner = CliRunner()

    def run(*args: str) -> str:
        result = runner.invoke(cli, ["--db", db, *args], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result.output

    first = json.loads(run("ingest", "--fixtures", str(FIXTURES), "--today", "2026-11-19"))
    assert first["inserted"] == 35
    again = json.loads(run("ingest", "--fixtures", str(FIXTURES), "--today", "2026-11-19"))
    assert again["inserted"] == 0
    assert json.loads(run("resolve"))["accounts"] == 8
    assert json.loads(run("score", "--as-of", "2026-11-19T00:00:00+00:00"))["icp_fit"] == 7

    rows = list(csv.DictReader(io.StringIO(run("export", "--format", "csv"))))
    assert rows[0]["domain"] == "northwindrobotics.com"
    clay = json.loads(run("export", "--format", "clay", "--icp-only"))
    assert len(clay) == 7
    out = tmp_path / "hubspot.json"
    run("export", "--format", "hubspot", "--out", str(out))
    assert json.loads(out.read_text())["batches"][0]["inputs"]

    history = json.loads(run("runs"))
    assert [r["kind"] for r in history["runs"]] == ["ingest", "ingest", "resolve", "score"]
