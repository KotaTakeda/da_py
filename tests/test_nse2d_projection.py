"""Tests for the Fourier low-/high-mode projections and synchronization (#12)."""

import numpy as np
import pytest

from da.nse2d import NSE2DTorus, inubushi_caulfield_config


@pytest.fixture(scope="module")
def model():
    cfg = inubushi_caulfield_config(
        nx=16, ny=16, viscosity=1.0e-3, drag=1.0e-1, forcing_mode=2, length=2 * np.pi
    )
    return NSE2DTorus(cfg)


@pytest.fixture()
def field(model):
    rng = np.random.default_rng(0)
    return rng.standard_normal(model.shape)


def test_projection_is_idempotent(model, field):
    P = model.project_low_modes(field, kmax=3)
    np.testing.assert_allclose(model.project_low_modes(P, kmax=3), P, atol=1e-12)


def test_high_projection_is_idempotent(model, field):
    Q = model.project_high_modes(field, kmax=3)
    np.testing.assert_allclose(model.project_high_modes(Q, kmax=3), Q, atol=1e-12)


def test_projections_decompose_identity(model, field):
    P = model.project_low_modes(field, kmax=3)
    Q = model.project_high_modes(field, kmax=3)
    np.testing.assert_allclose(P + Q, field, atol=1e-12)


def test_projections_are_complementary(model, field):
    # P annihilates the high-pass part and Q annihilates the low-pass part.
    P = model.project_low_modes(field, kmax=3)
    Q = model.project_high_modes(field, kmax=3)
    np.testing.assert_allclose(model.project_high_modes(P, kmax=3), 0.0, atol=1e-12)
    np.testing.assert_allclose(model.project_low_modes(Q, kmax=3), 0.0, atol=1e-12)


def test_projection_selects_expected_modes(model):
    # sin(2x) is an |m|=2 mode and sin(6y) an |m|=6 mode: a kmax=3 low-pass
    # keeps the former exactly and removes the latter.
    xx, yy = model.grid()
    low, high = np.sin(2 * xx), np.sin(6 * yy)
    P = model.project_low_modes(low + high, kmax=3)
    np.testing.assert_allclose(P, low, atol=1e-10)
    np.testing.assert_allclose(
        model.project_high_modes(low + high, kmax=3), high, atol=1e-10
    )


def test_projection_mask_matches_observation_convention(model):
    # The projection uses the same square integer-mode cutoff as
    # LowModeObservation: observing the projected field equals observing the
    # original field.
    rng = np.random.default_rng(1)
    omega = rng.standard_normal(model.shape)
    obs = model.low_mode_observation(kmax=2)
    np.testing.assert_allclose(
        obs.observe(model.project_low_modes(omega, kmax=2)),
        obs.observe(omega),
        atol=1e-10,
    )


def test_direct_insertion_synchronizes_on_tiny_grid(model):
    # Mechanism smoke test on a tiny (non-chaotic) grid. The chaotic
    # small-vs-large-cutoff contrast is exercised by the 64^2 sweep in
    # examples/nse2d_synchronization.py; here we pin two crisp invariants:
    # (a) an insertion step zeroes the observed-mode error exactly, and
    # (b) the assimilated twin converges to the truth over time.
    dt, n_steps, ka = 1.0e-2, 600, 6

    xx, yy = model.grid()
    truth = model.kolmogorov_vorticity(mode=2) + 0.1 * np.sin(xx + yy)
    for _ in range(100):  # short spin-up
        truth = model.step(truth, dt)

    # Start from the observed part plus an O(1) unobserved-mode perturbation
    # (modes (6, 7) lie outside the ka=6 square cutoff), so there is a
    # substantial high-mode error for the assimilation to remove.
    tilde = model.project_low_modes(truth, ka) + np.sin(6 * xx) * np.sin(7 * yy)
    t = truth.copy()
    err_init = np.linalg.norm(tilde - t) / np.linalg.norm(t)
    for _ in range(n_steps):
        t = model.step(t, dt)
        tilde = model.step(tilde, dt)
        tilde = model.project_low_modes(t, ka) + model.project_high_modes(tilde, ka)

    # (a) observed modes match the truth exactly after an insertion step
    np.testing.assert_allclose(
        model.project_low_modes(tilde - t, ka), 0.0, atol=1e-12
    )
    # (b) the total relative error decayed by well over an order of magnitude
    err_final = np.linalg.norm(tilde - t) / np.linalg.norm(t)
    assert err_final < 0.1 * err_init
