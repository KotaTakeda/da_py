import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "examples" / "example_registry.json"


def load_registry():
    return json.loads(REGISTRY_PATH.read_text())


def test_example_registry_references_existing_files():
    registry = load_registry()
    assert registry["version"] == 1
    assert registry["examples"]

    ids = set()
    for item in registry["examples"]:
        ids.add(item["id"])
        assert item["model"]
        assert item["filter"]
        assert item["status"] in {"representative", "representative_optional"}
        for key in ("script", "notebook", "docs"):
            assert (ROOT / item[key]).exists(), item[key]

    assert len(ids) == len(registry["examples"])


def test_docs_examples_mentions_all_registry_ids():
    docs = (ROOT / "docs" / "examples.md").read_text()
    for item in load_registry()["examples"]:
        assert item["id"] in docs


def test_representative_scripts_finish_and_report_rmse():
    registry = load_registry()
    skip_ids = {"l63_etpf"}
    for item in registry["examples"]:
        if item["id"] in skip_ids:
            continue
        result = subprocess.run(
            [sys.executable, item["script"], "--cycles", "2"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "final analysis RMSE:" in result.stdout


def test_optional_etpf_script_skips_without_pot_or_reports_rmse():
    result = subprocess.run(
        [sys.executable, "examples/scripts/l63_etpf.py", "--cycles", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "skipped: POT is not installed" in result.stdout or "final analysis RMSE:" in result.stdout
