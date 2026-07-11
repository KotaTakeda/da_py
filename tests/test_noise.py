import numpy as np
import pytest

from da.enkfn import EnKFN
from da.etkf import ETKF
from da.letkf import LETKF
from da.noise import GaussianModelNoise


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


# ---------------------------------------------------------------------------
# GaussianModelNoise sampler
# ---------------------------------------------------------------------------


def test_zero_covariance_gives_exact_zero_samples():
    rng = np.random.default_rng(0)
    # sigma = 0: dense zero matrix and zero diagonal variances give exact zeros
    for Q in (np.zeros((4, 4)), np.zeros(4)):
        eta = GaussianModelNoise(Q).sample(rng, 7)
        assert eta.shape == (7, 4)
        assert np.all(eta == 0.0)


def test_sigma_zero_projection_covariance_gives_exact_zero():
    P = orthogonal_projection(6, 2)
    eta = GaussianModelNoise(0.0 * P).sample(np.random.default_rng(1), 5)
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


def test_samples_independent_across_members_and_steps():
    Nx, sigma = 4, 1.0
    noise = GaussianModelNoise(sigma**2 * np.eye(Nx))
    rng = np.random.default_rng(7)
    steps = np.stack([noise.sample(rng, 6) for _ in range(20_000)])  # (n, m, Nx)

    # members within a step are uncorrelated
    member_corr = np.mean(steps[:, 0, :] * steps[:, 1, :])
    assert abs(member_corr) < 0.02
    # consecutive steps are uncorrelated (and not identical draws)
    step_corr = np.mean(steps[:-1, 0, :] * steps[1:, 0, :])
    assert abs(step_corr) < 0.02
    assert not np.array_equal(steps[0], steps[1])


def test_sampler_fixed_seed_is_reproducible():
    Q = 0.3 * orthogonal_projection(5, 2, seed=8)
    a = GaussianModelNoise(Q).sample(np.random.default_rng(42), 9)
    b = GaussianModelNoise(Q).sample(np.random.default_rng(42), 9)
    np.testing.assert_array_equal(a, b)


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
    # legacy RNG interfaces expose standard_normal but are still rejected
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        GaussianModelNoise(np.eye(2)).sample(np.random.RandomState(0), 3)
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        GaussianModelNoise(np.eye(2)).sample(np.random, 3)


# ---------------------------------------------------------------------------
# ETKF(Q=..., rng=...) constructor integration
# ---------------------------------------------------------------------------


def _run_etkf(Q, rng, cycles=5, n_obs=2):
    """Small linear-model ETKF loop; Q=None runs the deterministic filter."""
    H = np.eye(3)
    R = 0.5 * np.eye(3)
    filt = ETKF(lambda x, dt: 0.95 * x, H, R, alpha=1.02, Q=Q, rng=rng)
    X0 = np.random.default_rng(10).standard_normal((6, 3))
    filt.initialize(X0.copy())
    obs = np.random.default_rng(11).standard_normal((cycles, 3))
    for k in range(cycles):
        for _ in range(n_obs):
            filt.forecast(0.1)
        filt.update(obs[k])
    return filt.X


def test_etkf_without_Q_matches_zero_Q_exactly():
    deterministic = _run_etkf(None, None)
    zero_noise = _run_etkf(np.zeros(3), np.random.default_rng(12))
    np.testing.assert_array_equal(deterministic, zero_noise)


def test_etkf_with_Q_is_seed_reproducible():
    Q = 0.1 * np.eye(3)
    a = _run_etkf(Q, np.random.default_rng(13))
    b = _run_etkf(Q, np.random.default_rng(13))
    np.testing.assert_array_equal(a, b)
    c = _run_etkf(Q, np.random.default_rng(14))
    assert not np.array_equal(a, c)


def test_noise_is_applied_at_every_forecast_step():
    # identity model: any change in X between forecast calls is model noise
    filt = ETKF(lambda x, dt: x, np.eye(2), np.eye(2),
                Q=np.eye(2), rng=np.random.default_rng(15))
    filt.initialize(np.zeros((4, 2)))
    filt.forecast(0.1)
    after_first = filt.X.copy()
    assert not np.array_equal(after_first, np.zeros((4, 2)))
    filt.forecast(0.1)
    assert not np.array_equal(filt.X, after_first)


def test_forecast_diagnostics_include_model_noise():
    filt = ETKF(lambda x, dt: x, np.eye(2), np.eye(2), store_ensemble=True,
                Q=np.eye(2), rng=np.random.default_rng(16))
    filt.initialize(np.random.default_rng(17).standard_normal((4, 2)))
    filt.forecast(0.1)
    np.testing.assert_array_equal(filt.x_f[-1], filt.X.mean(axis=0))
    np.testing.assert_array_equal(filt.Xf[-1], filt.X)


def test_enkfn_and_letkf_accept_Q():
    H = np.eye(3)
    R = np.eye(3)
    X0 = np.random.default_rng(18).standard_normal((6, 3))
    y = np.array([0.1, -0.2, 0.3])
    for factory in (
        lambda rng: EnKFN(lambda x, dt: x, H, R, Q=0.1 * np.eye(3), rng=rng),
        lambda rng: LETKF(lambda x, dt: x, H, R, c=2.0, Q=0.1 * np.eye(3), rng=rng),
    ):
        runs = []
        for seed in (19, 19, 20):
            filt = factory(np.random.default_rng(seed))
            filt.initialize(X0.copy())
            filt.forecast(0.1)
            filt.update(y)
            runs.append(filt.X.copy())
        np.testing.assert_array_equal(runs[0], runs[1])  # same seed reproduces
        assert not np.array_equal(runs[0], runs[2])  # different seed differs


def test_constructor_validation_errors():
    M = lambda x, dt: x  # noqa: E731
    H = R = np.eye(2)
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        ETKF(M, H, R, Q=np.eye(2))  # Q without rng
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        ETKF(M, H, R, Q=np.eye(2), rng=np.random.RandomState(0))
    with pytest.raises(ValueError, match="rng has no effect without Q"):
        ETKF(M, H, R, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="state dimension"):
        filt = ETKF(M, H, R, Q=np.eye(3), rng=np.random.default_rng(0))
        filt.initialize(np.zeros((4, 2)))  # Nx=2 but Q is 3x3
