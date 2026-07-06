"""Lorenz-96 ETKF benchmark."""

import argparse

import numpy as np

from da.etkf import ETKF

from _common import add_common_args, ensemble_around, l96_step, print_result, rmse, truth_and_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=15)
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--obs-noise-variance", type=float, default=1.0)
    parser.add_argument("--inflation", type=float, default=1.05)
    return parser.parse_args()


def main():
    args = parse_args()
    H = np.eye(args.dimension)
    R = args.obs_noise_variance * np.eye(args.dimension)
    x0 = 8.0 * np.ones(args.dimension)
    x0[0] += 0.01
    truth, obs, rng = truth_and_observations(l96_step, x0, H, R, args)
    X0 = ensemble_around(rng, truth[0], args.ensemble_size, 0.5)
    filt = ETKF(l96_step, H, R, alpha=args.inflation)
    filt.initialize(X0)

    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))

    print_result("L96 ETKF benchmark", rmses, R=R, cycles=args.cycles, ensemble_size=args.ensemble_size)


if __name__ == "__main__":
    main()
