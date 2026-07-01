from pathlib import Path

import numpy as np

from da.etkf import ETKF
from da.nse2d import NSE2DConfig, NSE2DTorus


def initial_vorticity(model):
    x = np.linspace(0.0, model.config.length, model.nx, endpoint=False)
    y = np.linspace(0.0, model.config.length, model.ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    omega = np.sin(xx) * np.cos(yy) + 0.25 * np.cos(2 * yy)
    return omega


def rmse(x, x_ref):
    return float(np.sqrt(np.mean((x - x_ref) ** 2)))


def plot_rmse(times, analysis_rmse, obs_rmse):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    ax.plot(times, analysis_rmse, marker="o", label="analysis")
    ax.plot(times, obs_rmse, marker="o", label="observation")
    ax.set_xlabel("time")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_final_vorticity(model, truth, analysis):
    import matplotlib.pyplot as plt

    error = analysis - truth
    fields = [truth, analysis, error]
    titles = ["truth", "analysis mean", "analysis error"]
    vorticity_limit = float(np.max(np.abs([truth, analysis])))
    error_limit = float(np.max(np.abs(error)))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    for ax, field, title in zip(axes, fields, titles, strict=False):
        limit = error_limit if title == "analysis error" else vorticity_limit
        im = ax.imshow(
            field,
            extent=[0.0, model.config.length, 0.0, model.config.length],
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bicubic",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def main():
    rng = np.random.default_rng(7)
    cfg = NSE2DConfig(nx=16, ny=16, viscosity=1.0e-3, length=2 * np.pi)
    model = NSE2DTorus(cfg)

    dt = 1.0e-2
    internal_steps = 4
    n_cycles = 12
    ensemble_size = 12
    sigma_obs = 0.1
    sigma_init = 0.2
    use_linear_observation_matrix = True

    omega0 = initial_vorticity(model)
    truth = model.solve(
        omega0,
        dt=dt / internal_steps,
        n_steps=n_cycles * internal_steps,
        store_every=internal_steps,
    )
    truth_flat = truth.reshape(len(truth), model.state_dim)
    times = dt * np.arange(len(truth))

    grid_obs = model.grid_observation(stride=4)
    H = (
        model.grid_observation(stride=4, linear=True)
        if use_linear_observation_matrix
        else grid_obs.apply_flat
    )
    R = sigma_obs**2 * np.eye(grid_obs.obs_dim)
    y = np.stack([grid_obs.apply_flat(x) for x in truth_flat])
    y += rng.multivariate_normal(np.zeros(grid_obs.obs_dim), R, size=len(y))

    X0 = truth_flat[0] + sigma_init * rng.standard_normal(
        size=(ensemble_size, model.state_dim)
    )
    etkf = ETKF(model.as_forecast(internal_steps=internal_steps), H, R, alpha=1.02)
    etkf.initialize(X0)

    analysis_rmse = [rmse(etkf.X.mean(axis=0), truth_flat[0])]
    obs_rmse = [rmse(y[0], grid_obs.apply_flat(truth_flat[0]))]
    for n in range(1, len(truth_flat)):
        etkf.forecast(dt)
        etkf.update(y[n])
        analysis_rmse.append(rmse(etkf.X.mean(axis=0), truth_flat[n]))
        obs_rmse.append(rmse(y[n], grid_obs.apply_flat(truth_flat[n])))

    fourier_obs = model.independent_low_mode_observation(kmax=2)
    fourier_y = fourier_obs.observe_spectral(model.rfft(truth[-1]))

    print("state dimension:", model.state_dim)
    print("observation dimension:", grid_obs.obs_dim)
    print("ensemble size:", ensemble_size)
    print("linear observation matrix:", isinstance(H, np.ndarray))
    print("final analysis RMSE:", analysis_rmse[-1])
    print("time-mean analysis RMSE:", float(np.mean(analysis_rmse)))
    print("time-mean observation RMSE:", float(np.mean(obs_rmse)))
    print("independent low-mode observation dimension:", fourier_obs.obs_dim)
    print("independent low-mode observation norm:", np.linalg.norm(fourier_y))

    try:
        output_dir = Path("data/nse2d_etkf")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = plot_rmse(times, analysis_rmse, obs_rmse)
        fig.savefig(output_dir / "rmse.png", dpi=150)
        fig.clf()
        analysis = etkf.X.mean(axis=0).reshape(model.shape)
        fig = plot_final_vorticity(model, truth[-1], analysis)
        fig.savefig(output_dir / "final_vorticity.png", dpi=150)
        fig.clf()
        print("saved figures to:", output_dir)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
