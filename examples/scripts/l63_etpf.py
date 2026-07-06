"""Lorenz-63 ETPF benchmark.

This script exits successfully when POT is not installed, because ETPF is an
optional da_py feature.
"""

import argparse

import numpy as np

from _common import add_common_args, ensemble_around, l63_step, print_result, rmse, truth_and_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=20)
    parser.add_argument("--particles", type=int, default=40)
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
    R = 2.0 * np.eye(3)
    truth, obs, rng = truth_and_observations(l63_step, np.array([1.0, 1.0, 1.0]), H, R, args)
    X0 = ensemble_around(rng, truth[0] + np.array([1.0, -1.0, 0.5]), args.particles, 0.8)
    filt = EnsembleTransformParticleFilter(l63_step, lambda x: H @ x, R, add_inflation=0.02, N_thr=0.5)
    filt.initialize(X0)

    rmses = [rmse(filt.W @ filt.X, truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.W @ filt.X, truth[k]))

    print_result("L63 ETPF benchmark", rmses, cycles=args.cycles, particles=args.particles)


if __name__ == "__main__":
    main()
