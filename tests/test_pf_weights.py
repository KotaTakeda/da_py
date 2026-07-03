"""Weight-update tests for the particle filter (covers the batched path)."""

import numpy as np

from da.pf import ParticleFilter


def _pf_with_ensemble(X0, R, **kwargs):
    pf = ParticleFilter(M=None, h=lambda x: x, R=R, **kwargs)
    pf.initialize(X0)
    return pf


def test_weights_match_gaussian_likelihood():
    X0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    R = np.diag([0.5, 2.0])
    y = np.array([0.5, 0.5])

    pf = _pf_with_ensemble(X0.copy(), R, N_thr=0.0)  # N_thr=0 -> no resampling
    pf.update(y)

    Rinv = np.linalg.inv(R)
    nll = np.array([0.5 * (y - x) @ Rinv @ (y - x) for x in X0])
    expected = np.exp(-(nll - nll.min()))
    expected /= expected.sum()

    np.testing.assert_allclose(pf.W, expected, rtol=1e-12)
    assert np.isclose(pf.W.sum(), 1.0)
    # Analysis mean is the weighted ensemble mean.
    np.testing.assert_allclose(pf.x[-1], pf.W @ X0, rtol=1e-12)


def test_weights_uniform_for_equidistant_particles():
    X0 = np.array([[1.0], [-1.0]])
    pf = _pf_with_ensemble(X0.copy(), np.eye(1), N_thr=0.0)
    pf.update(np.array([0.0]))
    np.testing.assert_allclose(pf.W, [0.5, 0.5])
