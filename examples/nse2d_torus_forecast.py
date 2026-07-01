import argparse
import os
from pathlib import Path

import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus


def initial_vorticity(model, mode):
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=mode)
    omega += 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    return omega, xx, yy


def vorticity_cmap():
    try:
        import seaborn as sns

        return sns.color_palette("icefire", as_cmap=True)
    except ModuleNotFoundError:
        return "coolwarm"


def verify_png(path):
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return

    with Image.open(path) as img:
        img.verify()


def save_figure_safely(fig, path, *, dpi=150):
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.png")
    try:
        fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        verify_png(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)


def build_kolmogorov_model(args):
    cfg_kwargs = {
        "nx": args.nx,
        "ny": args.ny,
        "viscosity": args.viscosity,
        "length": 2 * np.pi,
    }
    base_model = NSE2DTorus(NSE2DConfig(**cfg_kwargs))
    forcing = base_model.kolmogorov_forcing(mode=args.mode)
    return NSE2DTorus(NSE2DConfig(**cfg_kwargs, forcing=forcing))


def solve_trajectory(model, omega0, args):
    store_every = max(1, args.n_steps // max(1, args.n_panels - 1))
    traj = model.solve(omega0, dt=args.dt, n_steps=args.n_steps, store_every=store_every)
    traj = traj[: args.n_panels]
    times = args.dt * store_every * np.arange(len(traj))
    return traj, times


def plot_vorticity_panels(traj, times, length, *, ncols=4):
    import matplotlib.pyplot as plt

    if len(traj) == 0:
        raise ValueError("traj must contain at least one state")
    vlim = float(np.percentile(np.abs(traj), 99.0))
    nrows = int(np.ceil(len(traj) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.1 * ncols, 3.15 * nrows),
    )
    axes = np.atleast_1d(axes).reshape(-1)
    cmap = vorticity_cmap()

    im = None
    for ax, omega, time in zip(axes, traj, times, strict=False):
        im = ax.imshow(
            omega,
            extent=[0.0, length, 0.0, length],
            origin="lower",
            cmap=cmap,
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_title(f"t={time:.3f}", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    for ax in axes[len(traj) :]:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.93,
        bottom=0.13,
        wspace=0.30,
        hspace=0.50,
    )
    fig.colorbar(
        im,
        ax=axes[: len(traj)].tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.06,
    )
    return fig


def diagnostics(model, traj):
    energies = np.array([model.energy(omega) for omega in traj])
    enstrophies = np.array([model.enstrophy(omega) for omega in traj])
    return energies, enstrophies


def plot_diagnostics(times, energies, enstrophies):
    import matplotlib.pyplot as plt

    if energies[0] == 0.0 or enstrophies[0] == 0.0:
        raise ValueError("initial energy and enstrophy must be nonzero")
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(times, energies / energies[0], marker="o", label="energy")
    ax.plot(times, enstrophies / enstrophies[0], marker="o", label="enstrophy")
    ax.set_xlabel("time")
    ax.set_ylabel("normalized value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.94, bottom=0.14)
    return fig


def save_plots(output_dir, model, traj, times, energies, enstrophies):
    output_dir = Path(output_dir)
    fig = plot_vorticity_panels(traj, times, model.config.length)
    save_figure_safely(fig, output_dir / "vorticity_panels.png")
    fig = plot_diagnostics(times, energies, enstrophies)
    save_figure_safely(fig, output_dir / "diagnostics.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a forced 2D Navier-Stokes trajectory on a torus.",
    )
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--n-panels", type=int, default=11)
    parser.add_argument("--mode", type=int, default=4)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_kolmogorov_model(args)
    omega0, xx, yy = initial_vorticity(model, mode=args.mode)

    traj, times = solve_trajectory(model, omega0, args)
    energies, enstrophies = diagnostics(model, traj)
    print("trajectory shape:", traj.shape)
    print("viscosity:", args.viscosity)
    print("final time:", float(times[-1]))
    print("initial energy/enstrophy:", energies[0], enstrophies[0])
    print("final energy/enstrophy:", energies[-1], enstrophies[-1])

    low_obs = model.low_mode_observation(kmax=2)
    grid_obs = model.grid_observation(stride=4)
    print("low-mode observation dimension:", low_obs.obs_dim)
    print("low-mode observation norm:", np.linalg.norm(low_obs.observe(traj[-1])))
    print("grid observation dimension:", grid_obs.obs_dim)
    print("grid observation mean:", np.mean(grid_obs.observe(traj[-1])))

    forecast = model.as_forecast(internal_steps=4)
    omega_next_flat = forecast(omega0.reshape(-1), args.dt)
    print("single flattened forecast shape:", omega_next_flat.shape)

    ensemble = np.stack(
        [
            omega0.reshape(-1),
            (omega0 + 1.0e-3 * np.sin(3 * xx)).reshape(-1),
            (omega0 + 1.0e-3 * np.cos(2 * yy)).reshape(-1),
        ],
        axis=0,
    )
    forecast_batch = model.forecast_batch_fn(dt=args.dt / 2, n_steps=2)
    ensemble_next = forecast_batch(ensemble)
    print("forecast ensemble shape:", ensemble_next.shape)

    try:
        output_dir = Path("data/nse2d_torus_forecast")
        save_plots(output_dir, model, traj, times, energies, enstrophies)
        print("saved figures to:", output_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
