"""Lorenz-96 LETKF benchmark."""

import argparse

import numpy as np

from da.letkf import LETKF

from _common import add_common_args, ensemble_around, l96_step, print_result, rmse, truth_and_observations


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=15)
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--ensemble-size", type=int, default=12)
    parser.add_argument("--localization-radius", type=float, default=6.0)
    return parser.parse_args()


def main():
    args = parse_args()
    H = np.eye(args.dimension)
    R = np.eye(args.dimension)
    x0 = 8.0 * np.ones(args.dimension)
    x0[0] += 0.01
    truth, obs, rng = truth_and_observations(l96_step, x0, H, R, args)
    X0 = ensemble_around(rng, truth[0], args.ensemble_size, 0.5)
    filt = LETKF(l96_step, H, R, alpha=1.05, c=args.localization_radius)
    filt.initialize(X0)

    rmses = [rmse(filt.X.mean(axis=0), truth[0])]
    for k in range(1, len(truth)):
        for _ in range(args.obs_interval):
            filt.forecast(args.dt)
        filt.update(obs[k])
        rmses.append(rmse(filt.X.mean(axis=0), truth[k]))

    print_result("L96 LETKF benchmark", rmses, cycles=args.cycles, ensemble_size=args.ensemble_size)


if __name__ == "__main__":
    main()
