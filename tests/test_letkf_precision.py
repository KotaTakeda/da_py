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


@pytest.fixture(params=[6, 12])
def correlated_partial_case(request):
    nx = request.param
    positions = np.flatnonzero(np.arange(nx) % 3 != 2)
    ny = len(positions)
    rng = np.random.default_rng(2026 + nx)
    X = rng.normal(size=(15, nx)) + np.linspace(-1, 2, nx)
    H = np.eye(nx)[positions]
    A = rng.normal(size=(ny, ny))
    R = A @ A.T / ny + 0.5 * np.eye(ny)
    y = rng.normal(size=ny)
    assert ny < nx and 3 * ny == 2 * nx
    assert not np.allclose(R, np.diag(np.diag(R)))
    assert np.linalg.eigvalsh(R).min() > 0
    return X, H, R, y, positions


@pytest.mark.parametrize("c", [0.6, 1.75, np.inf])
@pytest.mark.parametrize("explicit_positions", [False, True])
def test_correlated_partial_analysis_is_observation_permutation_invariant(
    correlated_partial_case, c, explicit_positions
):
    X, H, R, y, positions = correlated_partial_case
    permutation = np.random.default_rng(17).permutation(len(y))
    options = {"observation_positions": positions} if explicit_positions else {}
    permuted_options = (
        {"observation_positions": positions[permutation]} if explicit_positions else {}
    )
    original = LETKF(advance, H, R, alpha=1.1, c=c, **options)
    reordered = LETKF(
        advance, H[permutation], R[np.ix_(permutation, permutation)],
        alpha=1.1, c=c, **permuted_options
    )
    original.initialize(X)
    reordered.initialize(X)
    for i in range(X.shape[1]):
        local = original._locR(i)
        np.testing.assert_allclose(local, local.T, rtol=0, atol=1e-14)
        assert np.linalg.eigvalsh(local).min() >= -1e-13
        np.testing.assert_allclose(
            reordered._locR(i), local[np.ix_(permutation, permutation)],
            rtol=1e-12, atol=1e-13
        )
    original.update(y)
    reordered.update(y[permutation])
    np.testing.assert_allclose(reordered.X, original.X, rtol=1e-11, atol=1e-12)


def test_correlated_partial_global_limit_matches_etkf(correlated_partial_case):
    X, H, R, y, positions = correlated_partial_case
    global_filter = ETKF(advance, H, R, alpha=1.1)
    local_filter = LETKF(
        advance, H, R, alpha=1.1, c=np.inf, observation_positions=positions
    )
    for filt in [global_filter, local_filter]:
        filt.initialize(X)
        filt.update(y)
    np.testing.assert_allclose(local_filter.X, global_filter.X, rtol=1e-11, atol=1e-12)
