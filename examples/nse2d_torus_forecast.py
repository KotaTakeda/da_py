import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus


def initial_vorticity(model):
    x = np.linspace(0.0, model.config.length, model.nx, endpoint=False)
    y = np.linspace(0.0, model.config.length, model.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    omega = np.sin(xx) * np.cos(yy) + 0.25 * np.cos(2 * yy)
    return omega, xx, yy


def main():
    cfg = NSE2DConfig(nx=32, ny=32, viscosity=1.0e-3, length=2 * np.pi)
    model = NSE2DTorus(cfg)
    omega0, xx, yy = initial_vorticity(model)

    dt = 1.0e-2
    traj = model.solve(omega0, dt=dt, n_steps=20, store_every=5)
    print("trajectory shape:", traj.shape)
    print("initial energy/enstrophy:", model.energy(omega0), model.enstrophy(omega0))
    print("final energy/enstrophy:", model.energy(traj[-1]), model.enstrophy(traj[-1]))

    low_obs = model.low_mode_observation(kmax=2)
    grid_obs = model.grid_observation(stride=4)
    print("low-mode observation dimension:", low_obs.obs_dim)
    print("low-mode observation norm:", np.linalg.norm(low_obs.observe(traj[-1])))
    print("grid observation dimension:", grid_obs.obs_dim)
    print("grid observation mean:", np.mean(grid_obs.observe(traj[-1])))

    forecast = model.as_forecast(internal_steps=4)
    omega_next_flat = forecast(omega0.reshape(-1), dt)
    print("single flattened forecast shape:", omega_next_flat.shape)

    ensemble = np.stack(
        [
            omega0.reshape(-1),
            (omega0 + 1.0e-3 * np.sin(3 * xx)).reshape(-1),
            (omega0 + 1.0e-3 * np.cos(2 * yy)).reshape(-1),
        ],
        axis=0,
    )
    forecast_batch = model.forecast_batch_fn(dt=dt / 2, n_steps=2)
    ensemble_next = forecast_batch(ensemble)
    print("forecast ensemble shape:", ensemble_next.shape)


if __name__ == "__main__":
    main()
