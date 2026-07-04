"""Kelly-style EnKF/ETKF benchmark for partially observed 2D NSE (#13, #25).

Qualitative reproduction of the numerical-experiment section of Kelly,
Law & Stuart (2014) on the 2D Navier-Stokes torus with

- ``free``  -- no observations (free ensemble evolution baseline, Fig. 1);
- ``full``  -- all resolved Fourier modes observed;
- ``low``   -- low-mode partial observation ``H = P_lambda`` (Fig. 4);
- ``high``  -- high-mode-only observation ``H = Q_lambda = I - P_lambda`` (Fig. 5).

The key qualitative target: low-mode observations with inflation give
accurate filtering, high-mode-only observations do not. With the default
``kelly`` preset this contrast is reproduced (see issue #25 for the settings
study): low reaches relerr ~2e-2 while high stays O(1).

Three settings matter (issue #25):

- **Filter/inflation**: the default is the perturbed-observation EnKF
  ``PO(additive_inflation=True, alpha=--additive-alpha)``, whose additive
  inflation ``Pf -> Pf + alpha*I`` can correct errors *outside* the span of
  the ensemble anomalies — the mechanism behind Kelly et al.'s accuracy
  result. The multiplicative-anomaly ``ETKF(alpha=--alpha)`` stays available
  via ``--filter etkf`` but collapses/diverges in this regime because its
  inflation acts only inside the anomaly span. Kelly et al.'s
  ``alpha^2 = 0.0025`` is additive on their normalization and does not
  transfer literally; ``--additive-alpha`` must be calibrated to the state
  variance (0.5 works at the default preset, vorticity variance ~1e2).
- **Flow regime**: the ``kelly`` preset uses ``nu=0.01``, no Ekman drag,
  Kolmogorov forcing ``k_f=5`` with amplitude 10, ``dt=0.005`` and
  observations every ``J=20`` steps -- close to Kelly et al.'s setup and much
  less turbulent than the Inubushi-Caulfield configuration (kept as
  ``--preset inubushi``), where the resolved "high" band still contains the
  active scales and the Fig. 4/5 contrast disappears.
- **Initial ensemble**: sampled from a long on-attractor trajectory
  (climatological snapshots every ``--init-decorrelate`` steps), not truth
  plus white noise, so the experiment starts from O(1) climatological error
  as in Kelly et al.'s free-evolution baseline.

Observations are the non-redundant rfft2 coefficients (normalized by the
grid size), observed every ``--obs-interval`` model steps with iid
N(0, gamma^2) noise. Diagnostics per assimilation cycle: relative error
``relerr = ||mean - truth||_2 / ||truth||_2``, RMSE, ensemble spread, and the
observed-/unobserved-mode error split via the #12 projections. Outputs go to
``data/nse2d_kelly_etkf/``.

This benchmark remains **qualitative**: it is not an exact reproduction of
Kelly et al.'s parameters (see the open tasks in issue #25).
"""

import argparse
from pathlib import Path

import numpy as np

from da.etkf import ETKF
from da.loss import loss_rms
from da.nse2d import NSE2DConfig, NSE2DTorus, inubushi_caulfield_config
from da.po import PO

CASES = ("free", "full", "low", "high")


def build_model(args):
    if args.forcing == "diagonal":
        # Kelly-style forcing f = grad^perp Phi, Phi ~ cos(k_f . x) with
        # k_f = (mode, mode); amplitude is the velocity-forcing magnitude.
        base = NSE2DConfig(
            nx=args.nx,
            ny=args.ny,
            viscosity=args.viscosity,
            drag=args.drag,
            length=2 * np.pi,
        )
        helper = NSE2DTorus(base)
        cfg = NSE2DConfig(
            nx=args.nx,
            ny=args.ny,
            viscosity=args.viscosity,
            drag=args.drag,
            length=2 * np.pi,
            forcing=helper.diagonal_vorticity_forcing(
                mode=(args.forcing_mode, args.forcing_mode),
                amplitude=args.forcing_amplitude,
            ),
        )
    else:
        cfg = inubushi_caulfield_config(
            nx=args.nx,
            ny=args.ny,
            viscosity=args.viscosity,
            drag=args.drag,
            forcing_mode=args.forcing_mode,
            forcing_amplitude=args.forcing_amplitude,
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


def climatological_ensemble(model, omega_spun, args):
    """Sample the initial ensemble from a long on-attractor trajectory.

    Starting from the spun-up state, the free run is continued and one member
    is taken every ``--init-decorrelate`` steps, so the ensemble is a
    climatological sample of attractor states rather than truth plus white
    noise (which puts most of the initial perturbation off the attractor).
    The state reached after one further decorrelation gap is returned as the
    truth initial condition, so every member is at least one gap away from it.
    """
    members = []
    omega = omega_spun.copy()
    for _ in range(args.ensemble_size):
        for _ in range(args.init_decorrelate):
            omega = model.step(omega, args.dt)
        members.append(omega.reshape(-1).copy())
    for _ in range(args.init_decorrelate):
        omega = model.step(omega, args.dt)
    return np.stack(members), omega


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
    # substeps > 1 integrates the ensemble with a finer step (dt / substeps)
    # than the truth, which keeps the forecast stable through the large
    # transient adjustments right after an analysis step.
    M = model.as_forecast(internal_steps=args.obs_interval * args.substeps)

    da_filter = None
    if obs is not None:
        H = obs.as_matrix()
        R = args.gamma**2 * np.eye(obs.obs_dim)
        y = np.stack([H @ x for x in truth[1:]])
        y = y + args.gamma * rng.standard_normal(y.shape)
        if args.filter == "po":
            # Perturbed-observation EnKF with Kelly-style additive inflation
            # Pf -> Pf + additive_alpha * I (acts outside the anomaly span).
            da_filter = PO(M, H, R, alpha=args.additive_alpha, additive_inflation=True)
        else:
            # Square-root ETKF with multiplicative anomaly inflation A -> alpha*A.
            da_filter = ETKF(M, H, R, alpha=args.alpha)
        da_filter.initialize(X0.copy())
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

    ensemble = da_filter.X if da_filter is not None else X
    series = {key: [val] for key, val in diagnostics(ensemble, truth[0]).items()}
    for n in range(1, args.n_cycles + 1):
        if da_filter is not None:
            da_filter.forecast(dt_obs)
            da_filter.update(y[n - 1])
            ensemble = da_filter.X
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
        if args.filter == "po":
            label = rf"PO EnKF, additive $\alpha={args.additive_alpha}$"
        else:
            label = rf"ETKF, anomaly $\alpha={args.alpha}$"
        ax.set_title(
            label + rf", $|k_\lambda| \leq {args.kmax_obs}$, $\gamma={args.gamma}$"
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
    parser.add_argument(
        "--preset",
        choices=["kelly", "inubushi", "repro"],
        default="kelly",
        help="kelly: validated Fig.4/5 configuration (issue #25); inubushi: "
        "the strongly turbulent #11/#12 flow (contrast disappears); repro: "
        "larger kelly run outside CI",
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--spin-up", type=int, default=None)
    parser.add_argument("--n-cycles", type=int, default=None)
    parser.add_argument("--obs-interval", type=int, default=20, help="J model steps")
    parser.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="ensemble forecast substeps per model step (stability after updates)",
    )
    parser.add_argument("--kmax-obs", type=int, default=None, help="|k_lambda| cutoff")
    parser.add_argument("--gamma", type=float, default=0.01, help="obs noise std")
    parser.add_argument(
        "--filter",
        choices=["po", "etkf"],
        default="po",
        help="po: perturbed-observation EnKF with additive inflation "
        "(Kelly-style, default); etkf: multiplicative-anomaly ETKF",
    )
    parser.add_argument(
        "--additive-alpha",
        type=float,
        default=0.5,
        help="additive inflation Pf -> Pf + alpha*I for --filter po "
        "(calibrate to the state variance; Kelly's 0.0025 is on their scale)",
    )
    parser.add_argument("--alpha", type=float, default=1.2,
                        help="anomaly inflation factor (>=1) for --filter etkf")
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument(
        "--init",
        choices=["climatology", "perturb"],
        default="climatology",
        help="initial ensemble: attractor samples from a long free run "
        "(climatology) or truth plus white noise (perturb)",
    )
    parser.add_argument(
        "--init-decorrelate",
        type=int,
        default=200,
        help="model steps between climatological ensemble samples",
    )
    parser.add_argument("--init-spread", type=float, default=1.0,
                        help="white-noise std for --init perturb")
    parser.add_argument("--viscosity", type=float, default=None)
    parser.add_argument("--drag", type=float, default=None)
    parser.add_argument(
        "--forcing",
        choices=["kolmogorov", "diagonal"],
        default="kolmogorov",
        help="kolmogorov: single-mode sin(k y) forcing; diagonal: Kelly's "
        "grad^perp cos(k_f . x) forcing with k_f = (mode, mode)",
    )
    parser.add_argument("--forcing-mode", type=int, default=None)
    parser.add_argument("--forcing-amplitude", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="data/nse2d_kelly_etkf")
    args = parser.parse_args()

    # The kelly preset is the validated Fig.4/Fig.5 configuration from issue
    # #25: mild turbulence (nu=0.01, no Ekman drag, k_f=5 forcing at amplitude
    # 10), dt=0.005, J=20, |k_lambda|=5, climatological initial ensemble, and
    # the additive-inflation PO filter. The inubushi preset keeps the strongly
    # turbulent #11/#12 flow where the low/high contrast disappears.
    presets = {
        "kelly": {
            "nx": 32, "n_cycles": 40, "ensemble_size": 48, "dt": 5.0e-3,
            "spin_up": 1000, "kmax_obs": 5, "viscosity": 1.0e-2, "drag": 0.0,
            "forcing_mode": 5, "forcing_amplitude": 10.0,
        },
        "inubushi": {
            "nx": 32, "n_cycles": 30, "ensemble_size": 48, "dt": 1.0e-2,
            "spin_up": 500, "kmax_obs": 4, "viscosity": 1.0e-3, "drag": 1.0e-1,
            "forcing_mode": 4, "forcing_amplitude": 1.0,
        },
        "repro": {
            "nx": 64, "n_cycles": 100, "ensemble_size": 100, "dt": 5.0e-3,
            "spin_up": 2000, "kmax_obs": 5, "viscosity": 1.0e-2, "drag": 0.0,
            "forcing_mode": 5, "forcing_amplitude": 10.0,
        },
    }
    chosen = presets[args.preset]
    for key, value in chosen.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.ny = args.ny or args.nx
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = build_model(args)

    omega_spun = spun_up_truth(model, args)
    if args.init == "climatology":
        X0, omega0 = climatological_ensemble(model, omega_spun, args)
    else:
        omega0 = omega_spun
        X0 = omega0.reshape(-1)[None, :] + args.init_spread * rng.standard_normal(
            (args.ensemble_size, model.state_dim)
        )
    truth = truth_trajectory(model, omega0, args)

    print("Kelly-style EnKF benchmark (qualitative; see issue #25)")
    print("preset:", args.preset, "grid:", model.shape, "dt:", args.dt,
          "J:", args.obs_interval, "cycles:", args.n_cycles)
    print("|k_lambda| <=", args.kmax_obs, "gamma:", args.gamma,
          "m:", args.ensemble_size, "init:", args.init)
    if args.filter == "po":
        print("filter: PO EnKF, additive inflation Pf -> Pf +",
              args.additive_alpha, "* I (Kelly-style)")
    else:
        print("filter: ETKF, anomaly inflation alpha =", args.alpha,
              "(Pf -> alpha^2 Pf; acts only inside the anomaly span)")

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
        "(kelly preset + po filter reproduces the Fig.4/Fig.5 contrast; #25)",
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
