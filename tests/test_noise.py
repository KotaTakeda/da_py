import numpy as np
import pytest

from da.etkf import ETKF
from da.noise import GaussianModelNoise, sample_model_noise


def orthogonal_projection(Nx, rank, seed=0):
    """Test fixture: a random orthogonal projection P = V V^T of given rank.

    Validates the construction so that a broken fixture (non-symmetric or
    non-idempotent matrix used as a projection) fails loudly rather than
    silently weakening the Q = sigma^2 P tests.
    """
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((Nx, rank)))
    P = V @ V.T
    assert_projection(P)
    return P


def assert_projection(P):
    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"projection fixture must be square, got shape {P.shape}")
    if not np.allclose(P, P.T, atol=1.0e-12):
        raise ValueError("projection fixture must be symmetric")
    if not np.allclose(P @ P, P, atol=1.0e-12):
        raise ValueError("projection fixture must be idempotent (P @ P == P)")


def test_projection_fixture_rejects_non_idempotent_matrix():
    with pytest.raises(ValueError, match="idempotent"):
        assert_projection(np.array([[2.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):
        assert_projection(np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_zero_covariance_reproduces_deterministic_forecast_exactly():
    rng = np.random.default_rng(0)
    # sigma = 0: dense zero matrix and zero diagonal variances give exact zeros
    for Q in (np.zeros((4, 4)), np.zeros(4)):
        eta = GaussianModelNoise(Q).sample(rng, 7)
        assert eta.shape == (7, 4)
        assert np.all(eta == 0.0)


def test_sigma_zero_projection_covariance_gives_exact_zero():
    P = orthogonal_projection(6, 2)
    eta = sample_model_noise(np.random.default_rng(1), 0.0 * P, 5)
    assert np.all(eta == 0.0)


def test_samples_lie_in_range_of_projection():
    Nx, rank, sigma = 8, 3, 0.7
    P = orthogonal_projection(Nx, rank, seed=2)
    noise = GaussianModelNoise(sigma**2 * P)
    eta = noise.sample(np.random.default_rng(3), 500)
    # (I - P) eta = 0 for every sample, up to numerical tolerance; the floor
    # is the ~sqrt(eps) accuracy of eigh eigenvectors on the exactly
    # degenerate spectrum of sigma^2 P, i.e. ~1e-8 against an in-range
    # sample scale of sigma.
    residual = eta @ (np.eye(Nx) - P).T
    np.testing.assert_allclose(residual, 0.0, atol=1.0e-6 * sigma)


@pytest.mark.parametrize("rank", [8, 3])  # full-space P = I and lower-rank P
def test_empirical_mean_and_covariance_match_sigma2_P(rank):
    Nx, sigma, n = 8, 0.7, 200_000
    P = np.eye(Nx) if rank == Nx else orthogonal_projection(Nx, rank, seed=4)
    noise = GaussianModelNoise(sigma**2 * P)
    eta = noise.sample(np.random.default_rng(5), n)

    assert np.abs(eta.mean(axis=0)).max() < 5.0 * sigma / np.sqrt(n)
    cov = eta.T @ eta / n
    np.testing.assert_allclose(cov, sigma**2 * P, atol=0.02 * sigma**2)


def test_diagonal_variances_match():
    variances = np.array([0.0, 0.25, 1.0, 4.0])
    eta = GaussianModelNoise(variances).sample(np.random.default_rng(6), 100_000)
    np.testing.assert_allclose(eta.var(axis=0), variances, atol=0.05)
    assert np.all(eta[:, 0] == 0.0)


def test_samples_independent_across_members_and_cycles():
    Nx, sigma = 4, 1.0
    noise = GaussianModelNoise(sigma**2 * np.eye(Nx))
    rng = np.random.default_rng(7)
    cycles = np.stack([noise.sample(rng, 6) for _ in range(20_000)])  # (n, m, Nx)

    # members within a cycle are uncorrelated
    member_corr = np.mean(cycles[:, 0, :] * cycles[:, 1, :])
    assert abs(member_corr) < 0.02
    # consecutive cycles are uncorrelated (and not identical draws)
    cycle_corr = np.mean(cycles[:-1, 0, :] * cycles[1:, 0, :])
    assert abs(cycle_corr) < 0.02
    assert not np.array_equal(cycles[0], cycles[1])


def test_fixed_seed_is_reproducible():
    Q = 0.3 * orthogonal_projection(5, 2, seed=8)
    a = GaussianModelNoise(Q).sample(np.random.default_rng(42), 9)
    b = GaussianModelNoise(Q).sample(np.random.default_rng(42), 9)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, sample_model_noise(np.random.default_rng(42), Q, 9))


def test_invalid_covariances_raise_informative_errors():
    with pytest.raises(ValueError, match="square"):
        GaussianModelNoise(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        GaussianModelNoise(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianModelNoise(np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianModelNoise(np.array([[1.0, 0.0], [0.0, -1.0]]))
    with pytest.raises(ValueError, match="non-negative"):
        GaussianModelNoise(np.array([1.0, -0.1]))
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        GaussianModelNoise(np.eye(2)).sample(0, 3)


def _run_etkf(noise, rng, cycles=5):
    """Small linear-model ETKF loop; noise=None runs the deterministic filter."""
    H = np.eye(3)
    R = 0.5 * np.eye(3)
    filt = ETKF(lambda x, dt: 0.95 * x, H, R, alpha=1.02)
    X0 = np.random.default_rng(10).standard_normal((6, 3))
    filt.initialize(X0.copy())
    obs = np.random.default_rng(11).standard_normal((cycles, 3))
    for k in range(cycles):
        filt.forecast(0.1)
        if noise is not None:
            filt.X += noise.sample(rng, filt.m)
        filt.update(obs[k])
    return filt.X


def test_etkf_loop_with_zero_noise_matches_deterministic_run_exactly():
    deterministic = _run_etkf(None, None)
    zero_noise = _run_etkf(GaussianModelNoise(np.zeros(3)), np.random.default_rng(12))
    np.testing.assert_array_equal(deterministic, zero_noise)


def test_etkf_loop_with_model_noise_is_seed_reproducible():
    noise = GaussianModelNoise(0.1 * np.eye(3))
    a = _run_etkf(noise, np.random.default_rng(13))
    b = _run_etkf(noise, np.random.default_rng(13))
    np.testing.assert_array_equal(a, b)
    c = _run_etkf(noise, np.random.default_rng(14))
    assert not np.array_equal(a, c)
