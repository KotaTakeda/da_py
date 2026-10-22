"""Weight-update tests for the particle filter (covers the batched path)."""

import numpy as np
import pytest

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


def test_sequential_weights_equal_product_of_likelihoods():
    X0 = np.array([[-1.0], [0.0], [1.0]])
    pf = _pf_with_ensemble(X0, np.eye(1), N_thr=0.0)
    y = np.array([0.5])
    pf.update(y)
    expected = pf.W**2
    expected /= expected.sum()
    pf.update(y)
    np.testing.assert_allclose(pf.W, expected)
    np.testing.assert_allclose(pf.x[-1], expected @ X0)


def test_zero_prior_weight_stays_zero_with_extreme_likelihood():
    pf = _pf_with_ensemble(np.array([[0.0], [100.0], [101.0]]), np.eye(1), N_thr=0)
    pf.W = np.array([0.0, 0.25, 0.75])
    pf.update(np.array([0.0]))
    assert pf.W[0] == 0
    assert np.all(np.isfinite(pf.W))
    np.testing.assert_allclose(pf.W.sum(), 1.0)


@pytest.mark.parametrize("resample_option", ["systematic", "multinomial", "residual"])
def test_resampling_does_not_apply_likelihood_twice(resample_option):
    pf = _pf_with_ensemble(
        np.array([[-1.0], [0.0], [1.0]]), np.eye(1),
        resample_option=resample_option,
    )
    pf.update(np.array([0.5]))
    np.testing.assert_allclose(pf.W, np.full(3, 1 / 3))
    np.testing.assert_allclose(pf.x[-1], pf.X.mean(axis=0))
    saved = pf.Xa[-1].copy()
    pf.X += 10
    np.testing.assert_array_equal(pf.Xa[-1], saved)


def test_etpf_transport_preserves_posterior_mean_and_uniform_weights():
    pytest.importorskip("ot")
    from da.etpf import EnsembleTransformParticleFilter

    X0 = np.array([[-1.0], [0.0], [1.0]])
    pf = EnsembleTransformParticleFilter(None, lambda x: x, np.eye(1))
    pf.initialize(X0)
    likelihood = np.exp(-0.5 * (0.5 - X0[:, 0]) ** 2)
    expected_mean = (likelihood / likelihood.sum()) @ X0
    pf.update(np.array([0.5]))
    np.testing.assert_allclose(pf.W, np.full(3, 1 / 3))
    np.testing.assert_allclose(pf.x[-1], expected_mean)
