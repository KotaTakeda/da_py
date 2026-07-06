"""Lorenz-63 ETKF benchmark."""

import argparse

import numpy as np

from da.etkf import ETKF

from _common import add_common_args, ensemble_around, l63_step, print_result, rmse, truth_and_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=20)
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--obs-noise-variance", type=float, default=2.0)
    parser.add_argument("--inflation", type=float, default=1.02)
    return parser.parse_args()


def main():
    args = parse_args()
    H = np.eye(3)
    R = args.obs_noise_variance * np.eye(3)
    truth, obs, rng = truth_and_observations(l63_step, np.array([1.0, 1.0, 1.0]), H, R, args)
    X0 = ensemble_around(rng, truth[0] + np.array([1.0, -1.0, 0.5]), args.ensemble_size, 0.5)
    filt = ETKF(l63_step, H, R, alpha=args.inflation)
    filt.initialize(X0)

    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))

    print_result("L63 ETKF benchmark", rmses, R=R, cycles=args.cycles, ensemble_size=args.ensemble_size)


if __name__ == "__main__":
    main()
