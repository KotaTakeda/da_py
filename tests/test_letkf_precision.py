import numpy as np
import pytest

from da.etkf import ETKF
from da.letkf import LETKF

def advance(x, dt):
    return x + dt


@pytest.mark.parametrize("correlated", [False, True])
def test_partial_observations_global_limit_matches_etkf(correlated):
    H = np.eye(4)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(9, 4))
    A = rng.normal(size=(4, 4))
    R = A @ A.T + np.eye(4) if correlated else np.diag([0.5, 1, 2, 3])
    global_filter = ETKF(advance, H, R, alpha=1.1)
    local_filter = LETKF(advance, H, R, alpha=1.1, c=np.inf)
    for filt in [global_filter, local_filter]:
        filt.initialize(X)
        filt.update(np.arange(4.0))
    np.testing.assert_allclose(local_filter.X, global_filter.X, atol=1e-12)


def test_correlated_local_precision_is_symmetric_psd():
    precision = np.eye(4)
    precision[0, 1] = precision[1, 0] = 0.9
    filt = LETKF(advance, np.eye(4), np.linalg.inv(precision), c=0.1)
    filt.initialize(np.ones((5, 4)))
    for i in range(4):
        local = filt._locR(i)
        np.testing.assert_allclose(local, local.T, atol=1e-14)
        assert np.linalg.eigvalsh(local).min() >= -1e-14


def test_diagonal_local_precision_matches_weighted_inverse_variances():
    filt = LETKF(advance, np.eye(4), np.diag([1, 2, 3, 4]), c=2)
    filt.initialize(np.ones((5, 4)))
    for i in range(4):
        np.testing.assert_allclose(filt._locR(i), np.diag(filt._rho(i) / [1, 2, 3, 4]))


@pytest.mark.parametrize("R", [np.array([[1, 0.2], [0, 1]]), np.diag([1, -1])])
def test_invalid_covariance_rejected(R):
    with pytest.raises(ValueError, match="symmetric positive definite"):
        LETKF(advance, np.eye(2), R)
