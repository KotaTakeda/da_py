"""Kelly-style ETKF benchmark for partially observed 2D NSE (issue #13).

Qualitative reproduction of the numerical-experiment section of Kelly,
Law & Stuart (2014): ETKF on the 2D Navier-Stokes torus with

- ``free``  -- no observations (free ensemble evolution baseline, Fig. 1);
- ``full``  -- all resolved Fourier modes observed;
- ``low``   -- low-mode partial observation ``H = P_lambda`` (Fig. 4);
- ``high``  -- high-mode-only observation ``H = Q_lambda = I - P_lambda`` (Fig. 5).

The key qualitative target: low-mode observations with inflation give
accurate filtering, high-mode-only observations do not.

This benchmark is **qualitative and ETKF-based**, not an exact reproduction
of Kelly et al.'s perturbed-observation EnKF: it reuses the validated
Ekman-drag Kolmogorov-flow configuration and the low/high-mode split from the
synchronization benchmark (#11/#12) rather than Kelly's exact parameters.

Inflation convention: ``ETKF(alpha=...)`` in da_py is an *anomaly* inflation
factor (``A -> alpha*A``, hence ``Pf -> alpha^2*Pf``, with ``alpha >= 1``).
Kelly et al.'s additive variance-inflation parameter ``alpha^2 = 0.0025``
does NOT map onto this convention directly; the multiplicative default here
(``--alpha 1.2``) was chosen empirically.

Known limitations (documented, not fixed here):

- Without localization the ETKF analysis can only correct errors inside the
  span of the ensemble anomalies, so with a small ensemble the low-mode case
  gives *bounded but not fully accurate* filtering — the defaults keep ``m``
  comparable to the number of observed coefficients. Localization is the
  standard remedy and is an explicit non-goal of this issue.
- At the CI scale (32^2, moderate Reynolds number) the resolved "high" band
  ``|k| > kmax`` still contains the actively turbulent scales, so observing
  it nearly perfectly constrains the low modes through ensemble
  cross-covariances and the high-mode-only case tracks about as well as the
  low-mode case. Kelly et al.'s Fig. 5 failure of high-mode-only observation
  requires a larger separation between the observed cutoff and the resolved
  dissipative tail — i.e. higher resolution / lower viscosity than the CI
  preset (see ``--preset repro`` as a starting point).

Observations are the non-redundant rfft2 coefficients (normalized by the
grid size), observed every ``--obs-interval`` model steps with iid
N(0, gamma^2) noise. Diagnostics per assimilation cycle: relative error
``relerr = ||mean - truth||_2 / ||truth||_2``, RMSE, ensemble spread, and the
observed-/unobserved-mode error split via the #12 projections. Outputs go to
``data/nse2d_kelly_etkf/``.

Run the CI-sized default (32^2), or ``--preset repro`` for a larger run.
"""

import argparse
from pathlib import Path

import numpy as np

from da.etkf import ETKF
from da.loss import loss_rms
from da.nse2d import NSE2DTorus, inubushi_caulfield_config

CASES = ("free", "full", "low", "high")


def build_model(args):
    cfg = inubushi_caulfield_config(
        nx=args.nx,
        ny=args.ny,
        viscosity=args.viscosity,
        drag=args.drag,
        forcing_mode=args.forcing_mode,
        length=2 * np.pi,
    )
    return NSE2DTorus(cfg)


def spun_up_truth(model, args):
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=args.forcing_mode)
    omega = omega + 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    for _ in range(args.spin_up):
        omega = model.step(omega, args.dt)
    return omega


def truth_trajectory(model, omega0, args):
    """Truth states at the assimilation times (every obs-interval steps)."""
    states = [omega0.reshape(-1).copy()]
    omega = omega0.copy()
    for _ in range(args.n_cycles):
        for _ in range(args.obs_interval):
            omega = model.step(omega, args.dt)
        states.append(omega.reshape(-1).copy())
    return np.stack(states)


def make_observation(model, case, args):
    if case == "free":
        return None
    if case == "full":
        return model.full_mode_observation()
    if case == "low":
        return model.independent_low_mode_observation(kmax=args.kmax_obs)
    if case == "high":
        return model.high_mode_observation(kmax=args.kmax_obs)
    raise ValueError(f"unknown case {case!r}")


def run_case(model, case, truth, X0, rng, args):
    """Run one observation scenario; returns per-cycle diagnostics."""
    obs = make_observation(model, case, args)
    dt_obs = args.dt * args.obs_interval
    M = model.as_forecast(internal_steps=args.obs_interval)

    etkf = None
    if obs is not None:
        H = obs.as_matrix()
        R = args.gamma**2 * np.eye(obs.obs_dim)
        y = np.stack([H @ x for x in truth[1:]])
        y = y + args.gamma * rng.standard_normal(y.shape)
        etkf = ETKF(M, H, R, alpha=args.alpha)
        etkf.initialize(X0.copy())
    X = X0.copy()  # free-evolution ensemble (H = 0: no update step at all)

    def diagnostics(X, x_true):
        mean = X.mean(axis=0)
        err = mean - x_true
        err_field = err.reshape(model.shape)
        spread = float(np.sqrt(np.mean(X.var(axis=0, ddof=1))))
        return {
            "relerr": float(np.linalg.norm(err) / np.linalg.norm(x_true)),
            "rmse": float(loss_rms(mean, x_true)),
            "err_obs": float(
                np.linalg.norm(model.project_low_modes(err_field, args.kmax_obs))
            ),
            "err_unobs": float(
                np.linalg.norm(model.project_high_modes(err_field, args.kmax_obs))
            ),
            "spread": spread,
        }

    ensemble = etkf.X if etkf is not None else X
    series = {key: [val] for key, val in diagnostics(ensemble, truth[0]).items()}
    for n in range(1, args.n_cycles + 1):
        if etkf is not None:
            etkf.forecast(dt_obs)
            etkf.update(y[n - 1])
            ensemble = etkf.X
        else:
            X = np.stack([M(member, dt_obs) for member in X])
            ensemble = X
        for key, val in diagnostics(ensemble, truth[n]).items():
            series[key].append(val)
    return {key: np.asarray(vals) for key, vals in series.items()}


def plot_relerr(results, args, path):
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.5, height=3.5)
        cycles = np.arange(args.n_cycles + 1)
        for case in CASES:
            ax.semilogy(cycles, results[case]["relerr"], label=case)
        ax.set_xlabel("assimilation cycle")
        ax.set_ylabel(
            r"$\mathrm{relerr} = \|\bar{x}^a - x^{\mathrm{true}}\|_2 "
            r"/ \|x^{\mathrm{true}}\|_2$"
        )
        ax.set_title(
            rf"ETKF, $|k_\lambda| \leq {args.kmax_obs}$, "
            rf"$\gamma={args.gamma}$, $\alpha={args.alpha}$"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def plot_mode_split(results, args, path):
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.5, height=3.5)
        cycles = np.arange(args.n_cycles + 1)
        for case in ("low", "high"):
            ax.semilogy(cycles, results[case]["err_obs"], label=f"{case}: low-mode err")
            ax.semilogy(
                cycles,
                results[case]["err_unobs"],
                linestyle="--",
                label=f"{case}: high-mode err",
            )
        ax.set_xlabel("assimilation cycle")
        ax.set_ylabel("error norm")
        ax.set_title(r"observed vs unobserved subspace errors")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kelly-style ETKF benchmark for partially observed 2D NSE.",
    )
    parser.add_argument("--preset", choices=["ci", "repro"], default="ci")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--spin-up", type=int, default=500)
    parser.add_argument("--n-cycles", type=int, default=None)
    parser.add_argument("--obs-interval", type=int, default=10, help="J model steps")
    parser.add_argument("--kmax-obs", type=int, default=4, help="|k_lambda| cutoff")
    parser.add_argument("--gamma", type=float, default=0.01, help="obs noise std")
    parser.add_argument("--alpha", type=float, default=1.2,
                        help="da_py anomaly inflation factor (>=1)")
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--init-spread", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--drag", type=float, default=1.0e-1)
    parser.add_argument("--forcing-mode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="data/nse2d_kelly_etkf")
    args = parser.parse_args()

    # Without localization the ETKF corrects only within the span of the
    # ensemble anomalies, so m must be comparable to the number of observed
    # low-mode coefficients (~(2*kmax+1)^2) for the low-mode case to track.
    presets = {
        "ci": {"nx": 32, "n_cycles": 30, "ensemble_size": 48},
        "repro": {"nx": 64, "n_cycles": 100, "ensemble_size": 100},
    }
    chosen = presets[args.preset]
    args.nx = args.nx or chosen["nx"]
    args.ny = args.ny or args.nx
    args.n_cycles = args.n_cycles or chosen["n_cycles"]
    args.ensemble_size = args.ensemble_size or chosen["ensemble_size"]
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = build_model(args)

    omega0 = spun_up_truth(model, args)
    truth = truth_trajectory(model, omega0, args)

    X0 = truth[0][None, :] + args.init_spread * rng.standard_normal(
        (args.ensemble_size, model.state_dim)
    )

    print("Kelly-style ETKF benchmark (qualitative, ETKF-based)")
    print("grid:", model.shape, "dt:", args.dt, "J:", args.obs_interval,
          "cycles:", args.n_cycles)
    print("|k_lambda| <=", args.kmax_obs, "gamma:", args.gamma,
          "m:", args.ensemble_size)
    print("inflation: da_py anomaly factor alpha =", args.alpha,
          "(Pf -> alpha^2 Pf; NOT Kelly's additive alpha^2)")

    results = {}
    for case in CASES:
        res = run_case(model, case, truth, X0, rng, args)
        results[case] = res
        print(
            f"{case:>4}: relerr final {res['relerr'][-1]:.3e}, "
            f"time-mean RMSE {res['rmse'].mean():.3e}, "
            f"final RMSE {res['rmse'][-1]:.3e}, "
            f"unobserved-mode err final {res['err_unobs'][-1]:.3e}, "
            f"spread final {res['spread'][-1]:.3e}"
        )

    ok_low = results["low"]["relerr"][-1] < results["free"]["relerr"][-1]
    print("low-mode beats free evolution:", ok_low)
    print(
        "low vs high-mode-only final relerr:",
        f"{results['low']['relerr'][-1]:.3e}",
        "vs",
        f"{results['high']['relerr'][-1]:.3e}",
        "(Kelly's Fig.4/Fig.5 contrast needs larger scale separation; see docstring)",
    )

    try:
        out = Path(args.output_dir)
        p1 = plot_relerr(results, args, out / "relerr_cases.png")
        p2 = plot_mode_split(results, args, out / "mode_split.png")
        np.savez(
            out / "diagnostics.npz",
            **{f"{case}_{key}": vals for case, res in results.items()
               for key, vals in res.items()},
        )
        print("saved:", p1, ",", p2, "and diagnostics.npz")
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
