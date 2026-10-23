import numpy as np
import pytest

from da.etkf import ETKF
from da.l96 import two_thirds_observation
from da.letkf import LETKF
from da.localization import calc_dist, gaspari_cohn


def advance(x, dt):
    return x + dt


def test_partial_observations_and_order_invariance():
    H, positions = two_thirds_observation(6)
    X = np.random.default_rng(2).normal(size=(8, 6)) + 3
    R = np.diag([0.3, 0.7, 1.0, 2.0])
    y = np.arange(4.0)
    order = np.array([3, 1, 0, 2])
    original = LETKF(advance, H, R, c=2.0)
    permuted = LETKF(
        advance, H[order], R[np.ix_(order, order)], c=2.0,
        observation_positions=positions[order],
    )
    for filt, obs in [(original, y), (permuted, y[order])]:
        filt.initialize(X)
        filt.update(obs)
        assert filt.X.shape == X.shape
    np.testing.assert_allclose(original.X, permuted.X, atol=1e-12)
    np.testing.assert_array_equal(original.observation_positions, positions)


@pytest.mark.parametrize("correlated", [False])
def test_partial_observations_global_limit_matches_etkf(correlated):
    H, _ = two_thirds_observation(6)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 6))
    A = rng.normal(size=(4, 4))
    R = A @ A.T + np.eye(4) if correlated else np.diag([0.5, 1, 2, 3])
    global_filter = ETKF(advance, H, R, alpha=1.1)
    local_filter = LETKF(advance, H, R, alpha=1.1, c=np.inf)
    for filt in [global_filter, local_filter]:
        filt.initialize(X)
        filt.update(np.arange(4.0))
    np.testing.assert_allclose(local_filter.X, global_filter.X, atol=1e-12)


@pytest.mark.parametrize("H", [np.array([[0.5, 0.5]]), lambda x: x[:1] ** 2])
def test_nonselection_operator_needs_explicit_positions(H):
    filt = LETKF(advance, H, np.eye(1))
    with pytest.raises(ValueError, match="observation_positions is required"):
        filt.initialize(np.ones((3, 2)))
    positioned = LETKF(advance, H, np.eye(1), observation_positions=[0.5])
    positioned.initialize(np.ones((3, 2)))
    positioned.update(np.ones(1))


@pytest.mark.parametrize("positions", [[0], [0, np.nan], [0, 4], [-1, 0]])
def test_invalid_positions_rejected(positions):
    filt = LETKF(advance, np.eye(4)[[0, 2]], np.eye(2), observation_positions=positions)
    with pytest.raises(ValueError, match="observation_positions"):
        filt.initialize(np.ones((5, 4)))


def test_periodic_distance_and_kernel_boundaries():
    assert calc_dist(0, 5, J=6) == calc_dist(5, 0, J=6) == 1
    assert calc_dist(0, 5.5, J=6) == 0.5
    assert gaspari_cohn(2.0, 2.0) == pytest.approx(5 / 24)
    assert gaspari_cohn(4.0, 2.0) == 0
