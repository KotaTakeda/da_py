"""Lorenz-63 3DVar and ExKF benchmark."""

import argparse

import numpy as np

from da.exkf import ExKF
from da.var3d import Var3D

from _common import add_common_args, advance, l63_step, print_result, rmse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, cycles=20)
    parser.add_argument("--obs-noise-variance", type=float, default=2.0)
    parser.add_argument("--background-variance", type=float, default=4.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    H = np.eye(3)
    R = args.obs_noise_variance * np.eye(3)
    B = args.background_variance * np.eye(3)
    truth = np.array([1.0, 1.0, 1.0])
    x0 = truth + np.array([2.0, -1.0, 1.5])

    var3d = Var3D(lambda x, dt: advance(l63_step, x, dt, args.obs_interval), H, R)
    var3d.initialize(x0, B)
    exkf = ExKF(lambda x, dt: advance(l63_step, x, dt, args.obs_interval), H, R)
    exkf.initialize(x0, B)

    var_rmses = [rmse(x0, truth)]
    exkf_rmses = [rmse(x0, truth)]
    for _ in range(args.cycles):
        truth = advance(l63_step, truth, args.dt, args.obs_interval)
        y = H @ truth + rng.multivariate_normal(np.zeros(3), R)
        var3d.forecast(args.dt)
        var3d.update(y)
        exkf.forecast(args.dt)
        exkf.update(y)
        var_rmses.append(rmse(var3d.x_a, truth))
        exkf_rmses.append(rmse(exkf.x_a, truth))

    print_result("L63 3DVar benchmark", var_rmses, R=R, cycles=args.cycles)
    print_result("L63 ExKF benchmark", exkf_rmses, R=R, cycles=args.cycles)


if __name__ == "__main__":
    main()
