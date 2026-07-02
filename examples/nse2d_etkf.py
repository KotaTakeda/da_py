"""API smoke test for NSE2D spectral-state ETKF assimilation."""

import argparse
from pathlib import Path

import numpy as np

from da.etkf import ETKF
from da.nse2d import NSE2DTorus, inubushi_caulfield_config


def plot_analysis_rmse(analysis_rmse, path):
    """Save the per-cycle analysis RMSE curve via the shared figure layer."""
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.0, height=3.2)
        viz.line_plot(
            np.arange(len(analysis_rmse)), analysis_rmse, ax=ax, marker="o", label="ETKF"
        )
        ax.set_xlabel("assimilation cycle")
        ax.set_ylabel("analysis RMSE")
        ax.legend()
    return viz.save_png(fig, path)


def initial_vorticity(model):
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=4)
    omega += 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    return omega


def rmse(x, x_ref):
    return float(np.sqrt(np.mean((x - x_ref) ** 2)))


def solve_packed_spectral(model, x_hat0, dt, n_steps):
    x_hat = np.asarray(x_hat0, dtype=float).copy()
    out = [x_hat.copy()]
    for _ in range(n_steps):
        x_hat = model.step_spectral_state(x_hat, dt)
        out.append(x_hat.copy())
    return np.stack(out, axis=0)


def build_kolmogorov_model(args):
    return NSE2DTorus(
        inubushi_caulfield_config(
            nx=args.nx,
            ny=args.ny,
            viscosity=1.0e-3,
            drag=1.0e-1,
            forcing_mode=4,
            length=2 * np.pi,
        )
    )


def make_observations(model, truth_hat, rng, kmax_obs):
    obs = model.spectral_low_mode_observation(kmax=kmax_obs)
    H = obs.as_matrix()
    obs_var = obs.observation_variances(sigma0=0.5, decay_power=1.0)
    R = np.diag(obs_var)
    y = np.stack([H @ x_hat for x_hat in truth_hat])
    y += rng.multivariate_normal(np.zeros(obs.obs_dim), R, size=len(y))
    return obs, H, R, y


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small API smoke test for NSE2D + spectral ETKF.",
    )
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--kmax-obs", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="data/nse2d_etkf",
        help="directory for the analysis-RMSE figure (skipped if matplotlib missing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(7)

    model = build_kolmogorov_model(args)
    dt = 1.0e-2
    omega0 = initial_vorticity(model)
    truth_hat = solve_packed_spectral(
        model,
        model.to_spectral_state(omega0),
        dt=dt,
        n_steps=args.n_cycles,
    )
    truth = np.stack([model.from_spectral_state(x_hat) for x_hat in truth_hat])

    obs, H, R, y = make_observations(model, truth_hat, rng, args.kmax_obs)

    X0 = truth_hat[0] + 0.2 * rng.standard_normal(
        size=(args.ensemble_size, model.spectral_state_dim)
    )
    etkf = ETKF(model.as_spectral_forecast(internal_steps=2), H, R, alpha=1.02)
    etkf.initialize(X0)

    analysis_rmse = [rmse(model.from_spectral_state(etkf.X.mean(axis=0)), truth[0])]
    for n in range(1, len(truth_hat)):
        etkf.forecast(dt)
        etkf.update(y[n])
        analysis = model.from_spectral_state(etkf.X.mean(axis=0))
        analysis_rmse.append(rmse(analysis, truth[n]))

    print("NSE2D ETKF API smoke test")
    print("state space: packed rfft2 vorticity")
    print("spectral state dimension:", model.spectral_state_dim)
    print("observation dimension:", obs.obs_dim)
    print("linear observation matrix:", isinstance(H, np.ndarray))
    print("ensemble size:", args.ensemble_size)
    print("cycles:", args.n_cycles)
    print("final analysis RMSE:", analysis_rmse[-1])

    try:
        path = plot_analysis_rmse(analysis_rmse, Path(args.output_dir) / "analysis_rmse.png")
        print("saved figure to:", path)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
