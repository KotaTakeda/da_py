import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_reported(stdout, label):
    match = re.search(rf"{re.escape(label)}:\s*([0-9.eE+-]+)", stdout)
    assert match, f"missing '{label}' in:\n{stdout}"
    return float(match.group(1))


def _alpha_grid(**kw):
    import argparse

    sys.path.insert(0, str(ROOT / "examples" / "scripts"))
    try:
        import l96_enkfn_tuning
    finally:
        sys.path.remove(str(ROOT / "examples" / "scripts"))
    return l96_enkfn_tuning.alpha_grid(argparse.Namespace(**kw))


def test_alpha_grid_stays_within_requested_range():
    # exact-multiple range: endpoints inclusive
    grid = _alpha_grid(alpha_min=1.00, alpha_max=1.20, alpha_step=0.02)
    assert grid[0] == 1.00 and grid[-1] == 1.20
    assert len(grid) == 11
    # non-multiple range must not overshoot alpha_max
    grid = _alpha_grid(alpha_min=1.0, alpha_max=1.06, alpha_step=0.1)
    assert grid.max() <= 1.06 + 1e-9
    grid = _alpha_grid(alpha_min=1.0, alpha_max=1.05, alpha_step=0.02)
    assert grid.max() <= 1.05 + 1e-9
    assert grid.tolist() == [1.0, 1.02, 1.04]


def test_benchmark_script_smoke_reports_both_methods():
    result = subprocess.run(
        [sys.executable, "examples/scripts/l96_enkfn.py", "--cycles", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for label in (
        "observation noise scale",
        "ETKF post-spinup mean RMSE",
        "EnKFN post-spinup mean RMSE",
        "final analysis RMSE",
    ):
        assert label in result.stdout


def test_tuning_script_smoke_writes_csv(tmp_path):
    csv_path = tmp_path / "tuning.csv"
    result = subprocess.run(
        [
            sys.executable, "examples/scripts/l96_enkfn_tuning.py",
            "--cycles", "20", "--num-seeds", "2", "--spinup-cycles", "5",
            "--alpha-min", "1.06", "--alpha-max", "1.12", "--alpha-step", "0.02",
            "--no-figure", "--csv-output", str(csv_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "selected alpha_*" in result.stdout
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "alpha,rmse_mean,rmse_std,num_seeds"
    assert len(lines) == 1 + 4  # header + four alpha rows


def test_benchmark_default_meets_success_criteria():
    """Representative default: EnKF-N matches tuned ETKF without tuning.

    Acceptance criteria from the issue: EnKF-N post-spin-up mean RMSE is below
    the observation-noise scale and no worse than 1.2x the tuned ETKF value.
    """
    result = subprocess.run(
        [sys.executable, "examples/scripts/l96_enkfn.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    noise = _parse_reported(result.stdout, "observation noise scale")
    etkf = _parse_reported(result.stdout, "ETKF post-spinup mean RMSE")
    enkfn = _parse_reported(result.stdout, "EnKFN post-spinup mean RMSE")
    assert enkfn < noise, f"EnKF-N RMSE {enkfn} not below obs-noise scale {noise}"
    assert enkfn <= 1.2 * etkf, f"EnKF-N RMSE {enkfn} exceeds 1.2x tuned ETKF {etkf}"
