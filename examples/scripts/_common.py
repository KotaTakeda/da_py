"""Shared utilities for lightweight example scripts."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from da.l63 import lorenz63
from da.l96 import lorenz96
from da.scheme import rk4


def add_common_args(parser: argparse.ArgumentParser, *, cycles=20, seed=7):
    parser.add_argument("--cycles", type=int, default=cycles)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--obs-interval", type=int, default=5)
    return parser


def l63_step(x, dt):
    return rk4(lorenz63, 0.0, np.asarray(x), (), dt)


def l96_step(x, dt, forcing=8.0):
    return rk4(lorenz96, 0.0, np.asarray(x), (forcing,), dt)


def advance(step: Callable[[np.ndarray, float], np.ndarray], x, dt, n_steps):
    state = np.asarray(x, dtype=float).copy()
    for _ in range(n_steps):
        state = step(state, dt)
    return state


def truth_and_observations(step, x0, H, R, args):
    rng = np.random.default_rng(args.seed)
    x = np.asarray(x0, dtype=float).copy()
    truth = [x.copy()]
    obs = [H @ x + rng.multivariate_normal(np.zeros(H.shape[0]), R)]
    for _ in range(args.cycles):
        x = advance(step, x, args.dt, args.obs_interval)
        truth.append(x.copy())
        obs.append(H @ x + rng.multivariate_normal(np.zeros(H.shape[0]), R))
    return np.asarray(truth), np.asarray(obs), rng


def ensemble_around(rng, center, size, spread):
    return np.asarray(center) + spread * rng.standard_normal((size, len(center)))


def rmse(x, truth):
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(truth)) ** 2)))


def print_result(name, rmses, **metadata):
    print(name)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print(f"final analysis RMSE: {rmses[-1]:.6f}")
