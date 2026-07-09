"""Lorenz-96 EnKF-N vs tuned-ETKF benchmark.

Compares, on the ``Nx = 60`` 2/3-observed Lorenz-96 configuration:

- an ETKF with a preselected fixed multiplicative inflation ``alpha_*`` (tuned
  offline by ``examples/scripts/l96_enkfn_tuning.py``);
- an EnKF-N whose anomaly inflation ``l1`` is estimated adaptively and
  independently at each assimilation cycle.

The benchmark demonstrates that EnKF-N reaches accuracy comparable to the
carefully tuned fixed-inflation ETKF without any prior inflation tuning. Both
filters share the same truth, observations, and initial ensemble.

Success criteria for the representative configuration:

- EnKF-N post-spin-up mean RMSE no worse than 1.2x the tuned ETKF value;
- EnKF-N post-spin-up mean RMSE below the observation-noise scale (1 for R = I).
"""

import argparse
import csv
import os

import numpy as np

from da.enkfn import EnKFN
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
    parser.add_argument("--inflation", type=float, default=1.08,
                        help="preselected fixed inflation alpha_* for the tuned ETKF")
    parser.add_argument("--spinup-cycles", type=int, default=None)
    parser.add_argument("--series-csv", default=None,
                        help="optional path to write the per-cycle RMSE/l1 time series as CSV")
    parser.add_argument("--figure-output", default=None,
                        help="optional path to write the RMSE + l1 diagnostic figure")
    return parser


def run_etkf(args, H, R, truth, obs, X0):
    filt = ETKF(l96_step, H, R, alpha=args.inflation)
    filt.initialize(X0.copy())
    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))
    return np.asarray(rmses)


def run_enkfn(args, H, R, truth, obs, X0):
    filt = EnKFN(l96_step, H, R)
    filt.initialize(X0.copy())
    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))
    l1 = np.array([d["l1"] for d in filt.inflation_diagnostics])
    return np.asarray(rmses), l1


def save_series_csv(path, times, etkf_rmse, enkfn_rmse, enkfn_l1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["time", "etkf_rmse", "enkfn_rmse", "enkfn_l1"])
        for k, t in enumerate(times):
            l1 = "" if k == 0 else f"{enkfn_l1[k - 1]:.6f}"
            writer.writerow([f"{t:.4f}", f"{etkf_rmse[k]:.6f}", f"{enkfn_rmse[k]:.6f}", l1])


def save_figure(path, times, etkf_rmse, enkfn_rmse, enkfn_l1, sigma_obs, alpha_star):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(times, etkf_rmse, color="tab:blue", lw=1,
                 label=rf"tuned ETKF ($\alpha_*={alpha_star:.2f}$)")
    axes[0].plot(times, enkfn_rmse, color="tab:orange", lw=1, label="EnKF-N")
    axes[0].axhline(sigma_obs, color="gray", ls="--", label=r"$\sigma_{\mathrm{obs}}$")
    axes[0].set_ylabel("analysis RMSE")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(times[1:], enkfn_l1, color="tab:orange", lw=1)
    axes[1].axhline(1.0, color="gray", ls=":")
    axes[1].set_ylabel(r"EnKF-N inflation $l_1$")
    axes[1].set_xlabel("time")
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
    truth, obs, rng = truth_and_observations(l96_step, x0, H, R, args)
    X0 = attractor_ensemble(l96_step, rng, x0, args.dt, args.ensemble_size)

    etkf_rmse = run_etkf(args, H, R, truth, obs, X0)
    enkfn_rmse, enkfn_l1 = run_enkfn(args, H, R, truth, obs, X0)
    times = np.arange(len(truth)) * args.dt * args.obs_interval

    sigma_obs = obs_noise_scale(R)
    etkf_mean = post_spinup_mean(etkf_rmse, spinup)
    enkfn_mean = post_spinup_mean(enkfn_rmse, spinup)
    ratio = enkfn_mean / etkf_mean

    print("L96 EnKF-N vs tuned-ETKF benchmark")
    print(f"dimension: {args.dimension}  observed: {H.shape[0]}  ensemble size: {args.ensemble_size}")
    print(f"cycles: {args.cycles}  spin-up cycles: {spinup}  seed: {args.seed}")
    print(f"tuned ETKF inflation alpha_*: {args.inflation:.2f}")
    print(f"observation noise scale: {sigma_obs:.6f}")
    print(f"ETKF post-spinup mean RMSE: {etkf_mean:.6f}")
    print(f"EnKFN post-spinup mean RMSE: {enkfn_mean:.6f}")
    print(f"EnKFN/ETKF RMSE ratio: {ratio:.6f}")
    print(f"EnKFN mean l1 (post-spinup): {np.mean(enkfn_l1[spinup:]):.6f}")
    print(f"final analysis RMSE: {enkfn_rmse[-1]:.6f}")

    if args.series_csv:
        save_series_csv(args.series_csv, times, etkf_rmse, enkfn_rmse, enkfn_l1)
        print(f"wrote CSV: {args.series_csv}")
    if args.figure_output:
        save_figure(args.figure_output, times, etkf_rmse, enkfn_rmse, enkfn_l1,
                    sigma_obs, args.inflation)
        print(f"wrote figure: {args.figure_output}")


if __name__ == "__main__":
    main()
