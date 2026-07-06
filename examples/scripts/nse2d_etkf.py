"""NSE2D spectral-state ETKF benchmark."""

import argparse

import numpy as np

from da.etkf import ETKF
from da.loss import loss_rms
from da.nse2d import NSE2DTorus, inubushi_caulfield_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--kmax-obs", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def initial_vorticity(model):
    xx, yy = model.grid()
    return model.kolmogorov_vorticity(mode=4) + 0.1 * np.sin(xx + yy)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = NSE2DTorus(
        inubushi_caulfield_config(
            nx=args.nx,
            ny=args.ny,
            viscosity=1.0e-3,
            drag=1.0e-1,
            forcing_mode=4,
            length=2 * np.pi,
        )
    )
    dt = 1.0e-2
    truth_hat = [model.to_spectral_state(initial_vorticity(model))]
    for _ in range(args.cycles):
        truth_hat.append(model.step_spectral_state(truth_hat[-1], dt))
    truth_hat = np.asarray(truth_hat)

    obs = model.spectral_low_mode_observation(kmax=args.kmax_obs)
    H = obs.as_matrix()
    R = np.diag(obs.observation_variances(sigma0=0.5, decay_power=1.0))
    y = np.stack([H @ x for x in truth_hat])
    y += rng.multivariate_normal(np.zeros(obs.obs_dim), R, size=len(y))

    X0 = truth_hat[0] + 0.2 * rng.standard_normal((args.ensemble_size, model.spectral_state_dim))
    filt = ETKF(model.as_spectral_forecast(internal_steps=2), H, R, alpha=1.02)
    filt.initialize(X0)

    rmses = [float(loss_rms(filt.X.mean(axis=0), truth_hat[0]))]
    for k in range(1, len(truth_hat)):
        filt.forecast(dt)
        filt.update(y[k])
        rmses.append(float(loss_rms(filt.X.mean(axis=0), truth_hat[k])))

    print("NSE2D ETKF benchmark")
    print(f"spectral state dimension: {model.spectral_state_dim}")
    print(f"observation dimension: {obs.obs_dim}")
    print(f"linear observation matrix: {isinstance(H, np.ndarray)}")
    print(f"ensemble size: {args.ensemble_size}")
    print(f"cycles: {args.cycles}")
    print(f"final analysis RMSE: {rmses[-1]:.6f}")


if __name__ == "__main__":
    main()
