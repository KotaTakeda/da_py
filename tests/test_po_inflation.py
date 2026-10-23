import numpy as np
import pytest

from da.po import PO


@pytest.mark.parametrize("alpha", [1.0, 2.0])
def test_po_covariance_matches_inflated_kalman_formula(monkeypatch, alpha):
    # Exactly unit sample variance, zero means and zero sample cross-covariance.
    q = np.sqrt(3) / 2
    X = q * np.array([[-1.0], [-1.0], [1.0], [1.0]])
    noise = q * np.array([[-1.0], [1.0], [-1.0], [1.0]])
    monkeypatch.setattr(np.random, "multivariate_normal", lambda *args: noise.copy())
    filt = PO(lambda x, dt: x, np.eye(1), np.eye(1), alpha=alpha)
    filt.initialize(X)
    filt.update(np.array([1.0]))
    gain = alpha**2 / (alpha**2 + 1)
    np.testing.assert_allclose(filt.X.mean(axis=0), [gain], atol=1e-14)
    np.testing.assert_allclose(np.var(filt.X, axis=0, ddof=1), [(1-gain)*alpha**2])


def test_partial_observations_match_multivariate_kalman_moments(monkeypatch):
    # Orthogonal centered state/noise columns realize exactly uncorrelated noise.
    rng = np.random.default_rng(4)
    A = rng.normal(size=(8, 3))
    Q, _ = np.linalg.qr(A - A.mean(axis=0))
    X = np.sqrt(7) * Q[:, :2] + [2, -1]
    noise = np.sqrt(7) * Q[:, 2:]
    H = np.array([[1.0, 0.5]])
    R = np.eye(1)
    alpha = 2.0
    P = alpha**2 * np.cov(X.T)
    K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
    y = np.array([3.0])
    mean = X.mean(axis=0)
    monkeypatch.setattr(np.random, "multivariate_normal", lambda *args: noise.copy())
    filt = PO(lambda x, dt: x, H, R, alpha=alpha)
    filt.initialize(X)
    filt.update(y)
    np.testing.assert_allclose(filt.X.mean(axis=0), mean + K @ (y-H @ mean))
    np.testing.assert_allclose(np.cov(filt.X.T), P-K @ H @ P, atol=1e-13)


def test_nonlinear_observations_use_inflated_states(monkeypatch):
    X = np.array([[0.0], [1.0], [2.0]])
    inflated = X.mean(axis=0) + 2 * (X-X.mean(axis=0))
    predicted = inflated**2
    anomalies = inflated-inflated.mean(axis=0)
    obs_anomalies = predicted-predicted.mean(axis=0)
    K = (anomalies.T @ obs_anomalies) / (obs_anomalies.T @ obs_anomalies + 2)
    monkeypatch.setattr(np.random, "multivariate_normal", lambda *args: np.zeros_like(X))
    filt = PO(lambda x, dt: x, lambda x: x**2, np.eye(1), alpha=2)
    filt.initialize(X)
    filt.update(np.array([1.0]))
    np.testing.assert_allclose(filt.X, inflated + (1-predicted) @ K.T)
