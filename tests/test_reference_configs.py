"""Tests for the reference NSE benchmark configs (#27). No long simulations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "archive"))

from nse2d_reference_configs import REFERENCE_CONFIGS  # noqa: E402


def test_expected_configs_present():
    assert set(REFERENCE_CONFIGS) == {"kelly_32", "kelly_64", "inubushi_32"}


def test_kelly_32_low_mode_observation_dimension():
    # q_low = (2*5+1)^2 = 121: the rank bound relevant to issue #28.
    dims = REFERENCE_CONFIGS["kelly_32"].observation_dims()
    assert dims["low"] == 121
    assert dims["low"] + dims["high"] == dims["full"]


@pytest.mark.parametrize("name", ["kelly_32", "inubushi_32"])
def test_models_build_and_dims_are_consistent(name):
    cfg = REFERENCE_CONFIGS[name]
    model = cfg.build_model()
    assert model.shape == (cfg.ny, cfg.nx)
    dims = cfg.observation_dims()
    assert dims["low"] + dims["high"] == dims["full"] == model.spectral_state_dim
