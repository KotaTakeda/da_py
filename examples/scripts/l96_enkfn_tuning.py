"""Fixed-inflation ETKF tuning for the Lorenz-96 EnKF-N benchmark.

Sweeps a grid of fixed multiplicative inflation factors ``alpha`` for the ETKF
on the ``Nx = 60``, 2/3-observed Lorenz-96 configuration, evaluates each value
over several random seeds, and selects the inflation ``alpha_*`` that minimises
the seed-averaged post-spin-up time-averaged RMSE.

The selected ``alpha_*`` is the fixed inflation used for the tuned ETKF in the
paired benchmark ``examples/scripts/l96_enkfn.py``. Only the ETKF is tuned here;
EnKF-N estimates its inflation adaptively and requires no sweep.

Outputs a CSV summary (``alpha``, seed-mean RMSE, seed-std RMSE) and an
``alpha`` vs RMSE diagnostic figure with per-alpha uncertainty bars.
"""

import argparse
import csv
import os

import numpy as np

from da.etkf import ETKF
from da.l96 import two_thirds_observation

from _common import (
    add_common_args,
    attractor_ensemble,
    l96_step,
    obs_noise_scale,
    post_spinup_mean,
    rmse,
    truth_and_observations,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=400, seed=0)
    parser.add_argument("--dimension", type=int, default=60)
    parser.add_argument("--ensemble-size", type=int, default=30)
    parser.add_argument("--obs-noise-variance", type=float, default=1.0)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--alpha-min", type=float, default=1.00)
    parser.add_argument("--alpha-max", type=float, default=1.20)
    parser.add_argument("--alpha-step", type=float, default=0.02)
    parser.add_argument("--spinup-cycles", type=int, default=None)
    parser.add_argument("--csv-output", default="examples/output/l96_enkfn_tuning.csv")
    parser.add_argument("--figure-output", default="examples/output/l96_enkfn_tuning.png")
    parser.add_argument("--no-figure", action="store_true")
    return parser


def alpha_grid(args):
    if args.alpha_step <= 0:
        raise ValueError("--alpha-step must be positive")
    if args.alpha_max < args.alpha_min:
        raise ValueError("--alpha-max must be >= --alpha-min")
    # Use floor (with a small tolerance for float drift) so the grid stays
    # within [alpha_min, alpha_max] and never overshoots alpha_max when the
    # range is not an exact multiple of the step.
    span = args.alpha_max - args.alpha_min
    n = int(np.floor(span / args.alpha_step + 1e-9)) + 1
    return np.round(args.alpha_min + args.alpha_step * np.arange(n), 6)


def etkf_post_spinup_rmse(args, H, R, x0, alpha, seed, spinup):
    """Post-spin-up time-averaged analysis RMSE of the ETKF for one seed."""
    seed_args = argparse.Namespace(
        seed=seed, cycles=args.cycles, dt=args.dt, obs_interval=args.obs_interval
    )
    truth, obs, rng = truth_and_observations(l96_step, x0, H, R, seed_args)
    X0 = attractor_ensemble(l96_step, rng, x0, args.dt, args.ensemble_size)
    filt = ETKF(l96_step, H, R, alpha=alpha)
    filt.initialize(X0)

    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))
    return post_spinup_mean(rmses, spinup)


def save_csv(path, alphas, means, stds, num_seeds):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["alpha", "rmse_mean", "rmse_std", "num_seeds"])
        for alpha, mean, std in zip(alphas, means, stds):
            writer.writerow([f"{alpha:.4f}", f"{mean:.6f}", f"{std:.6f}", num_seeds])


def save_figure(path, alphas, means, stds, alpha_star, sigma_obs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=3, color="tab:blue",
                label="ETKF seed-mean RMSE")
    ax.axhline(sigma_obs, color="gray", ls="--", label=r"$\sigma_{\mathrm{obs}}$")
    ax.axvline(alpha_star, color="tab:red", ls=":", label=rf"$\alpha_*={alpha_star:.2f}$")
    ax.set_xlabel(r"fixed inflation $\alpha$")
    ax.set_ylabel("post-spin-up mean RMSE")
    ax.set_title("L96 (Nx=60, 2/3-observed) ETKF inflation sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args().parse_args()
    spinup = args.spinup_cycles if args.spinup_cycles is not None else args.cycles // 2

    H, observed = two_thirds_observation(args.dimension)
    R = args.obs_noise_variance * np.eye(H.shape[0])
    x0 = 8.0 * np.ones(args.dimension)
    x0[0] += 0.01
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    alphas = alpha_grid(args)

    print("L96 EnKF-N benchmark: ETKF fixed-inflation tuning")
    print(f"dimension: {args.dimension}  observed: {H.shape[0]}  ensemble size: {args.ensemble_size}")
    print(f"cycles: {args.cycles}  spin-up cycles: {spinup}  seeds: {seeds}")
    print(f"observation noise scale: {obs_noise_scale(R):.6f}")

    means, stds = [], []
    for alpha in alphas:
        per_seed = [etkf_post_spinup_rmse(args, H, R, x0, alpha, s, spinup) for s in seeds]
        mean, std = float(np.mean(per_seed)), float(np.std(per_seed))
        means.append(mean)
        stds.append(std)
        print(f"alpha={alpha:.2f}  rmse_mean={mean:.6f}  rmse_std={std:.6f}")

    means_arr = np.asarray(means)
    best = int(np.argmin(means_arr))
    alpha_star = float(alphas[best])
    print(f"selected alpha_*: {alpha_star:.2f}  (seed-mean RMSE {means_arr[best]:.6f})")

    save_csv(args.csv_output, alphas, means, stds, len(seeds))
    print(f"wrote CSV: {args.csv_output}")
    if not args.no_figure:
        save_figure(args.figure_output, alphas, means, stds, alpha_star, obs_noise_scale(R))
        print(f"wrote figure: {args.figure_output}")


if __name__ == "__main__":
    main()
