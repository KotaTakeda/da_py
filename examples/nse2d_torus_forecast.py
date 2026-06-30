import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus


def main():
    cfg = NSE2DConfig(nx=32, ny=32, viscosity=1.0e-3, length=2 * np.pi)
    model = NSE2DTorus(cfg)

    x = np.linspace(0.0, cfg.length, cfg.nx, endpoint=False)
    y = np.linspace(0.0, cfg.length, cfg.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    omega0 = np.sin(xx) * np.cos(yy) + 0.25 * np.cos(2 * yy)

    traj = model.solve(omega0, dt=1.0e-2, n_steps=20, store_every=5)
    print("trajectory shape:", traj.shape)
    print("initial energy/enstrophy:", model.energy(omega0), model.enstrophy(omega0))
    print("final energy/enstrophy:", model.energy(traj[-1]), model.enstrophy(traj[-1]))

    obs = model.low_mode_observation(kmax=2)
    y_obs = obs.observe(traj[-1])
    print("low-mode observation dimension:", obs.obs_dim)
    print("low-mode observation norm:", np.linalg.norm(y_obs))

    forecast = model.forecast_batch_fn(dt=1.0e-2, n_steps=2, flatten=True)
    ensemble = np.stack(
        [
            omega0.reshape(-1),
            (omega0 + 1.0e-3 * np.sin(3 * xx)).reshape(-1),
            (omega0 + 1.0e-3 * np.cos(2 * yy)).reshape(-1),
        ],
        axis=0,
    )
    ensemble_next = forecast(ensemble)
    print("forecast ensemble shape:", ensemble_next.shape)


if __name__ == "__main__":
    main()
