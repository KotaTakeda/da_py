"""Lorenz-63 ETPF benchmark.

This script exits successfully when POT is not installed, because ETPF is an
optional da_py feature.
"""

import argparse

import numpy as np

from _common import add_common_args, attractor_ensemble, l63_step, print_result, rmse, truth_and_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=20)
    parser.add_argument("--particles", type=int, default=40)
    parser.add_argument("--obs-noise-variance", type=float, default=2.0)
    parser.add_argument("--add-inflation", type=float, default=0.02)
    return parser.parse_args()


def main():
    try:
        from da.etpf import EnsembleTransformParticleFilter
    except ModuleNotFoundError as exc:
        if exc.name != "ot":
            raise
        print("L63 ETPF benchmark")
        print("skipped: POT is not installed; install da_py[etpf] to run this example")
        return

    args = parse_args()
    H = np.eye(3)
    R = args.obs_noise_variance * np.eye(3)
    truth, obs, rng = truth_and_observations(l63_step, np.array([1.0, 1.0, 1.0]), H, R, args)
    X0 = attractor_ensemble(l63_step, rng, np.array([1.0, 1.0, 1.0]), args.dt, args.particles)
    filt = EnsembleTransformParticleFilter(l63_step, lambda x: H @ x, R, add_inflation=args.add_inflation, N_thr=0.5)
    filt.initialize(X0)

    rmses = [rmse(filt.W @ filt.X, truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.W @ filt.X, truth[k]))

    print_result("L63 ETPF benchmark", rmses, R=R, cycles=args.cycles, particles=args.particles)


if __name__ == "__main__":
    main()
