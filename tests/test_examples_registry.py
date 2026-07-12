import importlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "examples" / "example_registry.json"


def registry_entry(example_id):
    for item in load_registry()["examples"]:
        if item["id"] == example_id:
            return item
    raise AssertionError(f"{example_id} not found in registry")


def script_defaults(module_name):
    """Return the argparse defaults of a representative script as a namespace."""
    sys.path.insert(0, str(ROOT / "examples" / "scripts"))
    argv = sys.argv
    try:
        module = importlib.import_module(module_name)
        sys.argv = [module_name]
        return module.parse_args()
    finally:
        sys.argv = argv
        sys.path.remove(str(ROOT / "examples" / "scripts"))


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
    docs = (ROOT / "docs" / "guides" / "examples.md").read_text()
    for item in load_registry()["examples"]:
        assert item["id"] in docs


def test_representative_scripts_finish_and_report_rmse():
    registry = load_registry()
    # l63_etpf needs the optional POT dependency; nse2d_etkf has a dedicated
    # convergence test below so it is not run again here at a stub cycle count.
    skip_ids = {"l63_etpf", "nse2d_etkf"}
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


def _parse_reported(stdout, label):
    match = re.search(rf"{label}:\s*([0-9.eE+-]+)", stdout)
    assert match, f"missing '{label}' in:\n{stdout}"
    return float(match.group(1))


def test_nse2d_default_assimilates_below_observation_noise():
    """The NSE2D representative default must actually assimilate.

    Guards against shipping a default whose analysis RMSE stays at the
    attractor scale (see docs/contributing/notebook_spec.md: a successful
    default reaches an RMSE clearly below the observation-noise scale).
    """
    result = subprocess.run(
        [sys.executable, "examples/scripts/nse2d_etkf.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    noise = _parse_reported(result.stdout, "observation noise scale")
    rmse = _parse_reported(result.stdout, "final analysis RMSE")
    assert rmse < noise, f"NSE2D default did not converge: RMSE {rmse} >= obs noise {noise}"


def test_nse2d_registry_matches_script_defaults():
    """The registry is the single source of truth; keep it in sync with the script."""
    entry = registry_entry("nse2d_etkf")["parameters"]
    args = script_defaults("nse2d_etkf")
    assert entry["grid"] == f"{args.nx}x{args.ny}"
    assert entry["ensemble_size"] == args.ensemble_size
    assert entry["cycles"] == args.cycles
    assert entry["viscosity"] == args.viscosity
    assert entry["drag"] == 0.0
    assert entry["inflation"] == args.inflation


def test_optional_etpf_script_skips_without_pot_or_reports_rmse():
    result = subprocess.run(
        [sys.executable, "examples/scripts/l63_etpf.py", "--cycles", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "skipped: POT is not installed" in result.stdout or "final analysis RMSE:" in result.stdout
