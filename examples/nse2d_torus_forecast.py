from pathlib import Path

import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus


def initial_vorticity(model):
    x = np.linspace(0.0, model.config.length, model.nx, endpoint=False)
    y = np.linspace(0.0, model.config.length, model.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    omega = np.sin(xx) * np.cos(yy) + 0.25 * np.cos(2 * yy)
    return omega, xx, yy


def plot_vorticity_panels(traj, times, length):
    import matplotlib.pyplot as plt

    vlim = float(np.percentile(np.abs(traj), 99.0))
    fig, axes = plt.subplots(
        1, len(traj), figsize=(3.0 * len(traj), 3.0), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    im = None
    for ax, omega, time in zip(axes, traj, times, strict=False):
        im = ax.imshow(
            omega,
            extent=[0.0, length, 0.0, length],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="bicubic",
        )
        ax.set_title(f"t={time:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    return fig


def plot_diagnostics(times, energies, enstrophies):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    ax.plot(times, energies, marker="o", label="energy")
    ax.plot(times, enstrophies, marker="o", label="enstrophy")
    ax.set_xlabel("time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def main():
    cfg = NSE2DConfig(nx=32, ny=32, viscosity=1.0e-3, length=2 * np.pi)
    model = NSE2DTorus(cfg)
    omega0, xx, yy = initial_vorticity(model)

    dt = 1.0e-2
    traj = model.solve(omega0, dt=dt, n_steps=20, store_every=5)
    times = dt * 5 * np.arange(len(traj))
    energies = np.array([model.energy(omega) for omega in traj])
    enstrophies = np.array([model.enstrophy(omega) for omega in traj])
    print("trajectory shape:", traj.shape)
    print("initial energy/enstrophy:", energies[0], enstrophies[0])
    print("final energy/enstrophy:", energies[-1], enstrophies[-1])

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

    try:
        output_dir = Path("data/nse2d_torus_forecast")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_vorticity_panels(traj, times, cfg.length)
        fig.savefig(output_dir / "vorticity_panels.png", dpi=150)
        fig.clf()
        fig = plot_diagnostics(times, energies, enstrophies)
        fig.savefig(output_dir / "diagnostics.png", dpi=150)
        fig.clf()
        print("saved figures to:", output_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
