import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus


def _model(nx=16, ny=16, viscosity=1.0e-2):
    return NSE2DTorus(NSE2DConfig(nx=nx, ny=ny, viscosity=viscosity))


def _grid(model):
    x = np.linspace(0.0, model.config.length, model.nx, endpoint=False)
    y = np.linspace(0.0, model.config.length, model.ny, endpoint=False)
    return np.meshgrid(x, y)


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

    forecast = model.as_forecast(n_steps=2)
    out = forecast(flat, 2.0e-3)
    expected = model.forecast_fn(dt=1.0e-3, n_steps=2)(flat)

    assert out.shape == (model.state_dim,)
    np.testing.assert_allclose(out, expected)


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


def test_short_trajectory_is_reproducible():
    model = _model(nx=8, ny=8)
    x, y = _grid(model)
    omega = np.sin(x + y)

    traj1 = model.solve(omega, dt=1.0e-3, n_steps=3)
    traj2 = model.solve(omega, dt=1.0e-3, n_steps=3)
    np.testing.assert_allclose(traj1, traj2, rtol=0.0, atol=0.0)
