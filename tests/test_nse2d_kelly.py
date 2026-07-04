"""Smoke tests for the Kelly-style high-/full-mode observations and ETKF (#13)."""

import numpy as np
import pytest

from da.etkf import ETKF
from da.nse2d import NSE2DTorus, inubushi_caulfield_config


@pytest.fixture(scope="module")
def model():
    cfg = inubushi_caulfield_config(
        nx=16, ny=16, viscosity=1.0e-3, drag=1.0e-1, forcing_mode=2, length=2 * np.pi
    )
    return NSE2DTorus(cfg)


def test_low_and_high_mode_observations_partition_the_spectrum(model):
    kmax = 2
    low = model.independent_low_mode_observation(kmax=kmax)
    high = model.high_mode_observation(kmax=kmax)
    full = model.full_mode_observation()
    assert low.obs_dim + high.obs_dim == full.obs_dim == model.spectral_state_dim


def test_high_mode_observation_sees_only_high_modes(model):
    kmax = 2
    high = model.high_mode_observation(kmax=kmax)
    rng = np.random.default_rng(0)
    omega = rng.standard_normal(model.shape)
    # Observing the high-pass part equals observing the full field, and the
    # low-pass part is invisible.
    np.testing.assert_allclose(
        high.observe(model.project_high_modes(omega, kmax)),
        high.observe(omega),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        high.observe(model.project_low_modes(omega, kmax)), 0.0, atol=1e-12
    )


def test_full_mode_observation_matrix_shape(model):
    H = model.full_mode_observation(linear=True)
    assert H.shape == (model.spectral_state_dim, model.state_dim)


def test_diagonal_vorticity_forcing(model):
    # F_omega = -|k| * amplitude * cos(k . x) for f = grad^perp Phi with
    # |f| = amplitude: mean zero, correct peak magnitude, correct value at 0.
    amplitude, mode = 10.0, (2, 2)
    F = model.diagonal_vorticity_forcing(mode=mode, amplitude=amplitude)
    kx = 2 * np.pi * mode[0] / model.config.length
    ky = 2 * np.pi * mode[1] / model.config.length
    k_norm = np.hypot(kx, ky)
    assert F.shape == model.shape
    assert abs(float(F.mean())) < 1e-12
    np.testing.assert_allclose(F[0, 0], -amplitude * k_norm, rtol=1e-12)
    np.testing.assert_allclose(np.abs(F).max(), amplitude * k_norm, rtol=1e-10)


def test_low_mode_etkf_beats_free_evolution(model):
    # Tiny end-to-end twin experiment: assimilating low-mode observations
    # must track the truth better than free ensemble evolution.
    dt, J, n_cycles, kmax, gamma, m = 1.0e-2, 10, 5, 4, 0.01, 8
    rng = np.random.default_rng(3)

    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=2) + 0.1 * np.sin(xx + yy)
    for _ in range(200):
        omega = model.step(omega, dt)

    truth = [omega.reshape(-1).copy()]
    for _ in range(n_cycles):
        for _ in range(J):
            omega = model.step(omega, dt)
        truth.append(omega.reshape(-1).copy())

    X0 = truth[0][None, :] + 1.0 * rng.standard_normal((m, model.state_dim))
    M = model.as_forecast(internal_steps=J)
    dt_obs = dt * J

    obs = model.independent_low_mode_observation(kmax=kmax)
    H = obs.as_matrix()
    R = gamma**2 * np.eye(obs.obs_dim)
    y = np.stack([H @ x for x in truth[1:]])
    y = y + gamma * rng.standard_normal(y.shape)

    etkf = ETKF(M, H, R, alpha=1.05)
    etkf.initialize(X0.copy())
    X_free = X0.copy()
    for n in range(n_cycles):
        etkf.forecast(dt_obs)
        etkf.update(y[n])
        X_free = np.stack([M(member, dt_obs) for member in X_free])

    x_true = truth[-1]
    err_da = np.linalg.norm(etkf.X.mean(axis=0) - x_true)
    err_free = np.linalg.norm(X_free.mean(axis=0) - x_true)
    assert err_da < err_free
