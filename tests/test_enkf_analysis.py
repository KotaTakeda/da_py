"""Analysis-step invariants for the ensemble Kalman filters (ETKF, PO, LETKF).

Test-only (issue #10). The linear-Gaussian checks compare the ensemble
analysis against the exact Kalman formulas evaluated with the *sample*
forecast covariance, which the square-root (ETKF) update must reproduce to
numerical precision. Randomness is only used to build initial ensembles and
is always seeded.
"""

import numpy as np
import pytest

from da.etkf import ETKF
from da.letkf import LETKF
from da.po import PO


def _identity_model(x, dt):
    return x


def _seeded_ensemble(m, nx, seed=0, spread=1.0, center=None):
    rng = np.random.default_rng(seed)
    X = spread * rng.standard_normal((m, nx))
    if center is not None:
        X += center
    return X


def _kalman_analysis(X, H, R, y):
    """Exact Kalman mean/covariance from the sample statistics of X (m, Nx)."""
    m = X.shape[0]
    xf = X.mean(axis=0)
    dX = X - xf
    Pf = dX.T @ dX / (m - 1)
    K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + R)
    xa = xf + K @ (y - H @ xf)
    Pa = Pf - K @ H @ Pf
    return xa, Pa


def test_etkf_update_preserves_ensemble_shape():
    X0 = _seeded_ensemble(7, 3)
    H = np.eye(3)
    etkf = ETKF(_identity_model, H, 0.5 * np.eye(3))
    etkf.initialize(X0)
    etkf.update(np.zeros(3))
    assert etkf.X.shape == (7, 3)
    assert len(etkf.x) == 1 and etkf.x[0].shape == (3,)


def test_etkf_matches_kalman_mean_and_covariance():
    # Linear-Gaussian case with m-1 >= Nx: the ETKF analysis mean and sample
    # covariance must equal the exact Kalman update built from the sample Pf.
    nx, m = 2, 30
    X0 = _seeded_ensemble(m, nx, seed=1, spread=0.8, center=np.array([1.0, -2.0]))
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.25]])
    y = np.array([1.7])

    etkf = ETKF(_identity_model, H, R, alpha=1.0)
    etkf.initialize(X0)
    etkf.update(y)

    xa_ref, Pa_ref = _kalman_analysis(X0, H, R, y)
    xa = etkf.X.mean(axis=0)
    dXa = etkf.X - xa
    Pa = dXa.T @ dXa / (m - 1)

    np.testing.assert_allclose(xa, xa_ref, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(Pa, Pa_ref, rtol=1e-8, atol=1e-10)


def test_etkf_uninformative_observation_is_noop():
    # With enormous observation noise the analysis equals the forecast.
    X0 = _seeded_ensemble(10, 3, seed=2)
    etkf = ETKF(_identity_model, np.eye(3), 1e12 * np.eye(3), alpha=1.0)
    etkf.initialize(X0)
    etkf.update(np.array([5.0, -5.0, 5.0]))
    np.testing.assert_allclose(etkf.X, X0, rtol=1e-6, atol=1e-6)


def test_etkf_mean_unchanged_when_innovation_is_zero():
    X0 = _seeded_ensemble(12, 2, seed=3)
    H = np.eye(2)
    etkf = ETKF(_identity_model, H, np.eye(2))
    etkf.initialize(X0)
    xf = X0.mean(axis=0)
    etkf.update(H @ xf)  # y = H xf -> zero innovation
    np.testing.assert_allclose(etkf.X.mean(axis=0), xf, rtol=1e-10, atol=1e-12)


def test_etkf_callable_observation_matches_linear_matrix():
    # A callable H exercising the np.vectorize branch must reproduce the
    # linear-matrix result exactly.
    X0 = _seeded_ensemble(15, 3, seed=4)
    H = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R = 0.1 * np.eye(2)
    y = np.array([0.3, -0.4])

    linear = ETKF(_identity_model, H, R)
    linear.initialize(X0.copy())
    linear.update(y)

    nonlin = ETKF(_identity_model, lambda x: H @ x, R)
    nonlin.initialize(X0.copy())
    nonlin.update(y)

    np.testing.assert_allclose(nonlin.X, linear.X, rtol=1e-10, atol=1e-12)


def test_etkf_forecast_applies_model_and_advances_time():
    X0 = _seeded_ensemble(5, 2, seed=5)
    etkf = ETKF(lambda x, dt: x + dt, np.eye(2), np.eye(2))
    etkf.initialize(X0)
    etkf.forecast(0.5)
    np.testing.assert_allclose(etkf.X, X0 + 0.5)
    assert etkf.t == pytest.approx(0.5)
    np.testing.assert_allclose(etkf.x_f[0], (X0 + 0.5).mean(axis=0))


def test_po_update_preserves_shape_and_weights_toward_obs():
    np.random.seed(0)  # PO draws perturbed observations from global RNG
    X0 = _seeded_ensemble(20, 2, seed=6)
    H = np.eye(2)
    po = PO(_identity_model, H, 1e-8 * np.eye(2))
    po.initialize(X0)
    y = np.array([3.0, -1.0])
    po.update(y)
    assert po.X.shape == (20, 2)
    # Nearly exact observation of the full state pulls the mean onto y.
    np.testing.assert_allclose(po.X.mean(axis=0), y, atol=1e-3)


def test_po_uninformative_observation_is_noop():
    np.random.seed(1)
    X0 = _seeded_ensemble(10, 2, seed=7)
    po = PO(_identity_model, np.eye(2), 1e12 * np.eye(2))
    po.initialize(X0)
    po.update(np.array([100.0, -100.0]))
    np.testing.assert_allclose(po.X, X0, rtol=1e-4, atol=1e-4)


def test_po_additive_inflation_path_runs():
    np.random.seed(2)
    X0 = _seeded_ensemble(10, 3, seed=8)
    po = PO(_identity_model, np.eye(3), np.eye(3), alpha=0.1, additive_inflation=True)
    po.initialize(X0)
    po.update(np.zeros(3))
    assert po.X.shape == (10, 3)
    assert np.all(np.isfinite(po.X))


def test_letkf_update_shape_and_uninformative_limit():
    # LETKF on a small ring: shape is preserved and a huge-R observation
    # leaves the ensemble (essentially) unchanged.
    nx, m = 8, 6
    X0 = _seeded_ensemble(m, nx, seed=9)
    letkf = LETKF(_identity_model, np.eye(nx), 1e12 * np.eye(nx), c=3.0)
    letkf.initialize(X0)
    letkf.update(np.zeros(nx))
    assert letkf.X.shape == (m, nx)
    np.testing.assert_allclose(letkf.X, X0, rtol=1e-5, atol=1e-5)


def test_letkf_matches_etkf_mean_for_large_localization_radius():
    # With H = I and a localization radius much larger than the domain the
    # LETKF analysis mean should be close to the global ETKF mean.
    nx, m = 6, 12
    X0 = _seeded_ensemble(m, nx, seed=10)
    H = np.eye(nx)
    R = 0.5 * np.eye(nx)
    y = np.linspace(-1.0, 1.0, nx)

    etkf = ETKF(_identity_model, H, R)
    etkf.initialize(X0.copy())
    etkf.update(y)

    letkf = LETKF(_identity_model, H, R, c=1e6)
    letkf.initialize(X0.copy())
    letkf.update(y)

    np.testing.assert_allclose(
        letkf.X.mean(axis=0), etkf.X.mean(axis=0), rtol=1e-6, atol=1e-8
    )
