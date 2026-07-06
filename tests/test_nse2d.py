import subprocess
import sys

import numpy as np
import pytest

from da.nse2d import NSE2DConfig, NSE2DTorus, inubushi_caulfield_config


def _model(nx=16, ny=16, viscosity=1.0e-2, drag=0.0, length=2 * np.pi):
    return NSE2DTorus(
        NSE2DConfig(
            nx=nx,
            ny=ny,
            viscosity=viscosity,
            drag=drag,
            length=length,
        )
    )


def _grid(model):
    x = np.linspace(0.0, model.config.length, model.nx, endpoint=False)
    y = np.linspace(0.0, model.config.length, model.ny, endpoint=False)
    return np.meshgrid(x, y)


def test_config_positional_arguments_keep_existing_meaning():
    cfg = NSE2DConfig(10, 12, 2.0e-2, 3.0)

    assert cfg.nx == 10
    assert cfg.ny == 12
    assert cfg.viscosity == 2.0e-2
    assert cfg.length == 3.0
    assert cfg.drag == 0.0


def test_spectral_derivative_consistency():
    model = _model()
    x, y = _grid(model)
    field = np.sin(3 * x) * np.cos(2 * y)

    np.testing.assert_allclose(
        model.derivative(field, "x"),
        3 * np.cos(3 * x) * np.cos(2 * y),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        model.derivative(field, "y"),
        -2 * np.sin(3 * x) * np.sin(2 * y),
        atol=1.0e-12,
    )


def test_inverse_laplacian_consistency_for_zero_mean_field():
    model = _model()
    x, y = _grid(model)
    psi = np.sin(2 * x) * np.cos(y)
    omega = 5 * psi

    np.testing.assert_allclose(model.streamfunction(omega), psi, atol=1.0e-12)


def test_velocity_reconstruction_is_incompressible():
    model = _model()
    x, y = _grid(model)
    omega = np.sin(2 * x) + np.cos(3 * y)
    u, v = model.velocity(omega)
    div = model.derivative(u, "x") + model.derivative(v, "y")

    np.testing.assert_allclose(div, 0.0, atol=1.0e-12)


def test_unforced_viscous_dynamics_decay_energy_and_enstrophy():
    model = _model(viscosity=5.0e-2)
    x, y = _grid(model)
    omega0 = np.sin(x) + 0.5 * np.cos(2 * y)
    omega1 = model.solve(omega0, dt=1.0e-3, n_steps=20)[-1]

    assert model.energy(omega1) < model.energy(omega0)
    assert model.enstrophy(omega1) < model.enstrophy(omega0)


def test_radial_spectra_sum_to_energy_and_enstrophy():
    model = _model(nx=16, ny=16)
    x, y = _grid(model)
    omega = np.sin(2 * x) * np.cos(3 * y) + 0.2 * np.cos(x - y)

    _, energy_spectrum = model.radial_spectrum(omega, quantity="energy")
    _, enstrophy_spectrum = model.radial_spectrum(omega, quantity="enstrophy")

    np.testing.assert_allclose(np.sum(energy_spectrum), model.energy(omega), atol=1e-14)
    np.testing.assert_allclose(
        np.sum(enstrophy_spectrum),
        model.enstrophy(omega),
        atol=1e-14,
    )


def test_spectral_tail_fraction_detects_low_mode_fields():
    model = _model(nx=16, ny=16)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(2 * y)

    assert model.spectral_tail_fraction(omega, cutoff=3, quantity="enstrophy") < 1e-28
    assert model.spectral_tail_fraction(omega, cutoff=3, quantity="energy") < 1e-28


def test_radial_spectrum_rejects_unknown_quantity():
    model = _model()
    with pytest.raises(ValueError, match="quantity must be"):
        model.radial_spectrum(np.zeros(model.shape), quantity="vorticity")


def test_zero_drag_preserves_rhs_behavior():
    x, y = _grid(_model())
    omega = np.sin(x) + np.cos(2 * y)
    no_drag = _model(nx=16, ny=16, viscosity=1.0e-2, drag=0.0)
    default_drag = NSE2DTorus(
        NSE2DConfig(nx=16, ny=16, viscosity=1.0e-2, length=2 * np.pi)
    )

    np.testing.assert_allclose(no_drag.rhs(omega), default_drag.rhs(omega))


def test_drag_decreases_enstrophy_in_unforced_run():
    model = _model(nx=16, ny=16, viscosity=0.0, drag=0.2)
    x, y = _grid(model)
    omega0 = np.sin(x) + 0.5 * np.cos(2 * y)
    omega1 = model.solve(omega0, dt=1.0e-2, n_steps=5)[-1]

    assert model.enstrophy(omega1) < model.enstrophy(omega0)


def test_kolmogorov_flow_is_steady_with_matching_forcing():
    length = 3.0
    base = _model(nx=16, ny=16, viscosity=1.0e-2, drag=0.3, length=length)
    forcing = base.kolmogorov_forcing(mode=2, amplitude=1.5)
    model = NSE2DTorus(
        NSE2DConfig(
            nx=16,
            ny=16,
            viscosity=1.0e-2,
            drag=0.3,
            length=length,
            forcing=forcing,
        )
    )
    omega = model.kolmogorov_vorticity(mode=2, amplitude=1.5)
    expected_u, expected_v = model.kolmogorov_velocity(mode=2, amplitude=1.5)
    u, v = model.velocity(omega)

    np.testing.assert_allclose(model.rhs(omega), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(u, expected_u, atol=1.0e-12)
    np.testing.assert_allclose(v, expected_v, atol=1.0e-12)


def test_kolmogorov_vorticity_forcing_shape_mean_and_values():
    model = _model(nx=16, ny=16, length=2 * np.pi)
    _, y = _grid(model)
    forcing = model.kolmogorov_vorticity_forcing(mode=4, amplitude=2.0)

    assert forcing.shape == model.shape
    np.testing.assert_allclose(np.mean(forcing), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(forcing, -8.0 * np.cos(4 * y), atol=1.0e-14)


def test_inubushi_caulfield_config_uses_forced_damped_reference_parameters():
    cfg = inubushi_caulfield_config(nx=16, ny=16)
    model = NSE2DTorus(cfg)

    assert cfg.viscosity == 1.0e-3
    assert cfg.drag == 1.0e-1
    assert cfg.forcing is not None
    np.testing.assert_allclose(
        cfg.forcing,
        model.kolmogorov_vorticity_forcing(mode=4),
    )


def test_kolmogorov_mode_must_be_positive():
    model = _model()

    for method in (
        model.kolmogorov_velocity,
        model.kolmogorov_vorticity,
        model.kolmogorov_forcing,
        model.kolmogorov_vorticity_forcing,
    ):
        with pytest.raises(ValueError, match="mode must be positive"):
            method(mode=0)


def test_palinstrophy_matches_analytic_gradient_norm():
    model = _model()
    x, y = _grid(model)
    omega = np.sin(2 * x) * np.cos(3 * y)
    expected = 0.5 * np.mean(
        (2 * np.cos(2 * x) * np.cos(3 * y)) ** 2
        + (-3 * np.sin(2 * x) * np.sin(3 * y)) ** 2
    )

    np.testing.assert_allclose(model.palinstrophy(omega), expected, atol=1.0e-12)


def test_forecast_adapter_shapes_for_single_state_and_batches():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = (np.sin(x) + np.cos(y)).astype(np.float32)
    flat = omega.reshape(-1)

    forecast = model.forecast_batch_fn(dt=1.0e-3, n_steps=2)
    out = forecast(flat)
    assert out.shape == (model.state_dim,)
    assert out.dtype == flat.dtype

    batch = np.stack([flat, 1.01 * flat], axis=0)
    batch_out = forecast(batch)
    assert batch_out.shape == batch.shape
    assert batch_out.dtype == batch.dtype


def test_as_forecast_accepts_da_model_signature():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(y)
    flat = omega.reshape(-1)

    forecast = model.as_forecast(internal_steps=2)
    out = forecast(flat, 2.0e-3)
    expected = model.forecast_fn(dt=1.0e-3, n_steps=2)(flat)

    assert out.shape == (model.state_dim,)
    np.testing.assert_allclose(out, expected)


def test_spectral_step_matches_real_space_step():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(2 * y)

    omega_next = model.step(omega, dt=1.0e-3)
    omega_hat_next = model.step_spectral(model.rfft(omega), dt=1.0e-3)

    np.testing.assert_allclose(model.irfft(omega_hat_next), omega_next)


def test_packed_spectral_state_roundtrip_and_forecast_match_real_space():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(2 * y)

    x_hat = model.to_spectral_state(omega)
    assert x_hat.shape == (model.spectral_state_dim,)
    assert model.spectral_state_dim == model.state_dim
    np.testing.assert_allclose(model.from_spectral_state(x_hat), omega)

    forecast = model.as_spectral_forecast(internal_steps=2)
    x_hat_next = forecast(x_hat, 2.0e-3)
    omega_next = model.forecast_fn(dt=1.0e-3, n_steps=2)(omega.reshape(-1))

    np.testing.assert_allclose(model.from_spectral_state(x_hat_next).reshape(-1), omega_next)


def test_observation_operators_are_deterministic_and_shape_consistent():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(2 * y)
    flat = omega.reshape(-1)

    low = model.low_mode_observation(kmax=2)
    y1 = low.observe(omega)
    y2 = low.apply_flat(flat)
    assert y1.shape == (low.obs_dim,)
    np.testing.assert_allclose(y1, y2)

    grid = model.grid_observation(stride=2)
    g1 = grid.observe(omega)
    g2 = grid.apply_flat(flat)
    assert g1.shape == (grid.obs_dim,)
    np.testing.assert_allclose(g1, g2)


def test_observation_matrices_match_matrix_free_apply():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(2 * y)
    flat = omega.reshape(-1)

    grid = model.grid_observation(stride=2)
    H_grid = grid.as_matrix()
    np.testing.assert_allclose(H_grid @ flat, grid.apply_flat(flat))
    np.testing.assert_allclose(model.grid_observation(stride=2, linear=True), H_grid)

    low = model.low_mode_observation(kmax=1)
    np.testing.assert_allclose(low.as_matrix() @ flat, low.apply_flat(flat), atol=1e-14)

    independent = model.independent_low_mode_observation(kmax=1)
    np.testing.assert_allclose(
        independent.as_matrix() @ flat,
        independent.apply_flat(flat),
        atol=1e-14,
    )

    spectral = model.spectral_low_mode_observation(kmax=1)
    x_hat = model.to_spectral_state(omega)
    H_spectral = spectral.as_matrix()
    np.testing.assert_allclose(H_spectral @ x_hat, spectral.apply_flat(x_hat))
    np.testing.assert_allclose(
        model.spectral_low_mode_observation(kmax=1, linear=True),
        H_spectral,
    )
    low_state = spectral.to_spectral_state(spectral.apply_flat(x_hat))
    np.testing.assert_allclose(H_spectral @ low_state, spectral.apply_flat(x_hat))
    assert spectral.observation_variances(sigma0=0.1).shape == (spectral.obs_dim,)


def test_independent_low_mode_observation_uses_minimal_real_coefficients():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(y)
    omega_hat = model.rfft(omega)

    obs = model.independent_low_mode_observation(kmax=1)
    y_real = obs.observe(omega)
    y_hat = obs.observe_spectral(omega_hat)

    assert obs.obs_dim == 9
    assert y_real.shape == (obs.obs_dim,)
    np.testing.assert_allclose(y_real, y_hat)


def test_grid_observation_output_does_not_alias_state():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x) + np.cos(y)
    omega_before = omega.copy()

    observed = model.grid_observation(stride=1).observe(omega)
    observed += 1.0

    np.testing.assert_allclose(omega, omega_before)


def test_short_trajectory_is_reproducible():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x + y)

    traj1 = model.solve(omega, dt=1.0e-3, n_steps=3)
    traj2 = model.solve(omega, dt=1.0e-3, n_steps=3)
    np.testing.assert_allclose(traj1, traj2, rtol=0.0, atol=0.0)


def test_dealias_pad_fft_roundtrip_preserves_resolved_coefficients():
    model = NSE2DTorus(NSE2DConfig(nx=8, ny=8, dealias="pad"))
    x, y = _grid(model)
    omega = np.sin(2 * x) * np.cos(3 * y) + 0.25 * np.cos(x - y)
    omega_hat = model.fft(omega)

    np.testing.assert_allclose(
        model._truncate_fft(model._pad_fft(omega_hat)),
        omega_hat,
        atol=1e-12,
    )


def test_dealias_pad_fft_aligns_modes_on_odd_grids():
    model = NSE2DTorus(NSE2DConfig(nx=7, ny=7, dealias="pad"))
    omega_hat = np.zeros(model.shape, dtype=complex)
    omega_hat[0, 0] = 1.0
    omega_hat[4, 2] = 2.0 + 3.0j  # integer mode (kx, ky) = (2, -3)

    padded = model._pad_fft(omega_hat)
    pad_kx = np.rint(
        np.fft.fftfreq(model._pad_shape[1]) * model._pad_shape[1]
    ).astype(int)
    pad_ky = np.rint(
        np.fft.fftfreq(model._pad_shape[0]) * model._pad_shape[0]
    ).astype(int)
    zero_y = int(np.flatnonzero(pad_ky == 0)[0])
    zero_x = int(np.flatnonzero(pad_kx == 0)[0])
    mode_y = int(np.flatnonzero(pad_ky == -3)[0])
    mode_x = int(np.flatnonzero(pad_kx == 2)[0])

    assert padded[zero_y, zero_x] == model._pad_scale
    assert padded[mode_y, mode_x] == (2.0 + 3.0j) * model._pad_scale
    np.testing.assert_allclose(
        model._truncate_fft(padded),
        omega_hat,
        atol=1e-12,
    )


def test_dealias_pad_fft_zeros_even_grid_nyquist_modes():
    model = NSE2DTorus(NSE2DConfig(nx=8, ny=8, dealias="pad"))
    omega_hat = np.zeros(model.shape, dtype=complex)
    omega_hat[0, 0] = 1.0
    omega_hat[1, 2] = 2.0 + 3.0j
    omega_hat[7, 6] = 2.0 - 3.0j
    omega_hat[0, model.nx // 2] = 4.0
    omega_hat[model.ny // 2, 0] = 5.0
    omega_hat[model.ny // 2, model.nx // 2] = 6.0

    padded = model._pad_fft(omega_hat)
    padded_field = np.fft.ifft2(padded)
    truncated = model._truncate_fft(padded)

    assert np.max(np.abs(padded_field.imag)) < 1e-14
    assert truncated[0, model.nx // 2] == 0.0
    assert truncated[model.ny // 2, 0] == 0.0
    assert truncated[model.ny // 2, model.nx // 2] == 0.0
    assert truncated[0, 0] == omega_hat[0, 0]
    assert truncated[1, 2] == omega_hat[1, 2]
    assert truncated[7, 6] == omega_hat[7, 6]


def test_dealias_pad_short_trajectory_is_finite():
    model = NSE2DTorus(NSE2DConfig(nx=8, ny=8, viscosity=1.0e-2, dealias="pad"))
    x, y = _grid(model)
    omega = np.sin(x + y) + 0.2 * np.cos(2 * x - y)

    traj = model.solve(omega, dt=1.0e-3, n_steps=3)

    assert traj.shape == (4, model.ny, model.nx)
    assert np.all(np.isfinite(traj))


def test_dealias_option_is_validated():
    with pytest.raises(ValueError, match="dealias must be"):
        NSE2DConfig(dealias="unknown")


def test_nse2d_etkf_example_smoke():
    result = subprocess.run(
        [sys.executable, "examples/scripts/nse2d_etkf.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "final analysis RMSE:" in result.stdout
    assert "linear observation matrix: True" in result.stdout
