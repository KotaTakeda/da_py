import argparse
from pathlib import Path

import numpy as np

from da.nse2d import NSE2DTorus, inubushi_caulfield_config


def initial_vorticity(model, mode):
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=mode)
    omega += 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    return omega, xx, yy


def save_figure_safely(fig, path, *, dpi=150):
    """Backward-compatible alias for the shared atomic PNG writer.

    The original bespoke implementation (temp file + PNG verify + atomic
    ``os.replace``) now lives in ``da.viz.save_png`` /
    ``research_figures.save_figure``; keep this thin wrapper so any external
    caller of this script keeps working.
    """
    from da import viz

    return viz.save_png(fig, path, dpi=dpi)


def build_kolmogorov_model(args):
    cfg = inubushi_caulfield_config(
        nx=args.nx,
        ny=args.ny,
        viscosity=args.viscosity,
        drag=args.drag,
        forcing_mode=args.mode,
        length=2 * np.pi,
    )
    return NSE2DTorus(cfg)


def solve_trajectory(model, omega0, args):
    store_every = max(1, args.n_steps // max(1, args.n_panels - 1))
    traj = model.solve(omega0, dt=args.dt, n_steps=args.n_steps, store_every=store_every)
    traj = traj[: args.n_panels]
    times = args.dt * store_every * np.arange(len(traj))
    return traj, times


def plot_vorticity_panels(
    traj,
    times,
    length,
    *,
    cmap_name="icefire",
    interpolation="nearest",
    ncols=4,
):
    """Vorticity snapshots as a labelled panel grid with a shared colorbar.

    Built on the generic figure layer: :func:`da.viz.multi_panel` for the grid,
    :func:`da.viz.image_plot` per panel, and :func:`da.viz.shared_colorbar` /
    :func:`da.viz.panel_labels` for the shared scale and ``(a)``, ``(b)`` labels.
    A single symmetric limit ``[-vlim, vlim]`` (99th percentile of ``|traj|``,
    as before) is shared across panels so the colorbar is meaningful.
    """
    from da import viz

    if len(traj) == 0:
        raise ValueError("traj must contain at least one state")
    vlim = float(np.percentile(np.abs(traj), 99.0))
    nrows = int(np.ceil(len(traj) / ncols))
    cmap = viz.vorticity_cmap(cmap_name)
    extent = [0.0, length, 0.0, length]

    with viz.style_context():
        fig, axes = viz.multi_panel(nrows, ncols)
        im = None
        for ax, omega, time in zip(axes, traj, times, strict=False):
            _, im = viz.image_plot(
                omega,
                ax=ax,
                extent=extent,
                cmap=cmap,
                vmin=-vlim,
                vmax=vlim,
                interpolation=interpolation,
            )
            ax.set_title(f"t={time:.3f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        used = axes[: len(traj)]
        viz.hide_unused(axes, len(traj))
        viz.panel_labels(used)
        viz.shared_colorbar(fig, im, used, label="vorticity")
    return fig


def diagnostics(model, traj):
    energies = np.array([model.energy(omega) for omega in traj])
    enstrophies = np.array([model.enstrophy(omega) for omega in traj])
    palinstrophies = np.array([model.palinstrophy(omega) for omega in traj])
    return energies, enstrophies, palinstrophies


def plot_diagnostics(times, energies, enstrophies, palinstrophies):
    """Normalized energy/enstrophy/palinstrophy curves via the shared line plot."""
    from da import viz

    if energies[0] == 0.0 or enstrophies[0] == 0.0 or palinstrophies[0] == 0.0:
        raise ValueError("initial diagnostics must be nonzero")
    with viz.style_context():
        fig, ax = viz.single_panel(width=5.0, height=3.5)
        viz.line_plot(times, energies / energies[0], ax=ax, marker="o", label="energy")
        viz.line_plot(
            times, enstrophies / enstrophies[0], ax=ax, marker="o", label="enstrophy"
        )
        viz.line_plot(
            times,
            palinstrophies / palinstrophies[0],
            ax=ax,
            marker="o",
            label="palinstrophy",
        )
        ax.set_xlabel("time")
        ax.set_ylabel("normalized value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    return fig


def save_plots(
    output_dir,
    model,
    traj,
    times,
    energies,
    enstrophies,
    palinstrophies,
    *,
    cmap_name,
    interpolation,
):
    output_dir = Path(output_dir)
    fig = plot_vorticity_panels(
        traj,
        times,
        model.config.length,
        cmap_name=cmap_name,
        interpolation=interpolation,
    )
    save_figure_safely(fig, output_dir / "vorticity_panels.png")
    fig = plot_diagnostics(times, energies, enstrophies, palinstrophies)
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
    parser.add_argument("--drag", type=float, default=1.0e-1)
    parser.add_argument("--cmap", default="icefire")
    parser.add_argument(
        "--interpolation",
        choices=["nearest", "none", "bilinear"],
        default="nearest",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_kolmogorov_model(args)
    omega0, xx, yy = initial_vorticity(model, mode=args.mode)

    traj, times = solve_trajectory(model, omega0, args)
    energies, enstrophies, palinstrophies = diagnostics(model, traj)
    print("trajectory shape:", traj.shape)
    print("viscosity:", args.viscosity)
    print("drag:", args.drag)
    print("final time:", float(times[-1]))
    print("initial energy/enstrophy:", energies[0], enstrophies[0])
    print("final energy/enstrophy:", energies[-1], enstrophies[-1])
    print("initial/final palinstrophy:", palinstrophies[0], palinstrophies[-1])

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
        save_plots(
            output_dir,
            model,
            traj,
            times,
            energies,
            enstrophies,
            palinstrophies,
            cmap_name=args.cmap,
            interpolation=args.interpolation,
        )
        print("saved figures to:", output_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
