"""Low-mode synchronization (continuous data assimilation) benchmark for 2D NSE.

Solver/observation validation stage that comes *before* ETKF experiments
(issue #12): a twin experiment in the style of Inubushi & Caulfield checks
whether observing only the low Fourier modes of the reference vorticity is
enough for an assimilated copy to synchronize, i.e. whether the solver
transfers information from observed (low) to unobserved (high) modes.

Two assimilation modes are provided:

- ``insertion`` — direct insertion / discrete CDA: every ``obs-every`` steps
  the observed low modes of the assimilated state are replaced by the truth,
  ``tilde <- P_ka truth + Q_ka tilde``;
- ``nudging``  — finite-relaxation CDA: after every model step the low modes
  are relaxed toward the truth, ``tilde <- tilde - mu dt P_ka (tilde-truth)``
  (a split-step discretization of the -mu P (tilde - u) nudging term).

Diagnostics per cutoff ``ka``: the relative error
``E_omega = ||tilde-omega||_2 / ||omega||_2`` plus its observed/unobserved
splits ``E_P = ||P_ka e||_2`` and ``E_Q = ||Q_ka e||_2``, and the enstrophy /
enstrophy-error pair. A sweep over ``ka`` shows the qualitative
synchronization threshold: small cutoffs fail to synchronize, larger ones
drive the unobserved-mode error toward zero.
"""

import argparse
from pathlib import Path

import numpy as np

from da.nse2d import NSE2DTorus, inubushi_caulfield_config


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
    """Return an on-attractor truth state after a deterministic spin-up."""
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=args.forcing_mode)
    omega = omega + 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    for _ in range(args.spin_up):
        omega = model.step(omega, args.dt)
    return omega


def run_synchronization(model, truth0, args, ka):
    """Twin experiment for one cutoff; returns diagnostics time series."""
    truth = truth0.copy()
    # Assimilated state starts from the observed part only: high modes zeroed.
    tilde = model.project_low_modes(truth0, ka)

    n_rec = args.n_steps // args.store_every + 1
    times = np.zeros(n_rec)
    E_omega = np.zeros(n_rec)
    E_P = np.zeros(n_rec)
    E_Q = np.zeros(n_rec)
    enstrophy = np.zeros(n_rec)
    enstrophy_err = np.zeros(n_rec)

    def record(idx, t):
        err = tilde - truth
        times[idx] = t
        E_omega[idx] = np.linalg.norm(err) / np.linalg.norm(truth)
        E_P[idx] = np.linalg.norm(model.project_low_modes(err, ka))
        E_Q[idx] = np.linalg.norm(model.project_high_modes(err, ka))
        enstrophy[idx] = model.enstrophy(truth)
        enstrophy_err[idx] = 0.5 * float(np.mean(err**2))

    record(0, 0.0)
    idx = 1
    for step in range(1, args.n_steps + 1):
        truth = model.step(truth, args.dt)
        tilde = model.step(tilde, args.dt)
        if args.mode == "insertion":
            if step % args.obs_every == 0:
                tilde = model.project_low_modes(truth, ka) + model.project_high_modes(
                    tilde, ka
                )
        else:  # nudging
            tilde = tilde - args.mu * args.dt * model.project_low_modes(
                tilde - truth, ka
            )
        if step % args.store_every == 0:
            record(idx, step * args.dt)
            idx += 1

    return {
        "times": times,
        "E_omega": E_omega,
        "E_P": E_P,
        "E_Q": E_Q,
        "enstrophy": enstrophy,
        "enstrophy_err": enstrophy_err,
    }


def free_run_error(model, truth0, args, ka):
    """Reference: same initial condition but no assimilation at all."""
    truth = truth0.copy()
    tilde = model.project_low_modes(truth0, ka)
    for _ in range(args.n_steps):
        truth = model.step(truth, args.dt)
        tilde = model.step(tilde, args.dt)
    err = tilde - truth
    return float(np.linalg.norm(err) / np.linalg.norm(truth))


def plot_sweep(results, args, path):
    """Semilog relative-error curves for every cutoff in the sweep."""
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.5, height=3.5)
        for ka, res in sorted(results.items()):
            ax.semilogy(res["times"], res["E_omega"], label=rf"$k_a={ka}$")
        ax.set_xlabel(r"time $t$")
        ax.set_ylabel(r"$E_\omega = \|\tilde\omega-\omega\|_2 / \|\omega\|_2$")
        ax.set_title(f"{args.mode}, forcing mode $k_f={args.forcing_mode}$")
        ax.legend()
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def plot_mode_split(res, ka, args, path):
    """Observed- vs unobserved-mode error for one cutoff."""
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.5, height=3.5)
        ax.semilogy(res["times"], res["E_P"], label=r"$E_P=\|P_{k_a}e\|_2$ (observed)")
        ax.semilogy(
            res["times"], res["E_Q"], label=r"$E_Q=\|Q_{k_a}e\|_2$ (unobserved)"
        )
        ax.set_xlabel(r"time $t$")
        ax.set_ylabel("error norm")
        ax.set_title(rf"{args.mode}, $k_a={ka}$")
        ax.legend()
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-mode synchronization/nudging benchmark for 2D NSE.",
    )
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--n-steps", type=int, default=1500)
    parser.add_argument("--spin-up", type=int, default=500)
    parser.add_argument("--store-every", type=int, default=25)
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--drag", type=float, default=1.0e-1)
    parser.add_argument("--forcing-mode", type=int, default=4)
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="low-pass cutoffs k_a to sweep",
    )
    parser.add_argument(
        "--mode",
        choices=["insertion", "nudging"],
        default="insertion",
        help="direct insertion (discrete CDA) or finite-mu nudging",
    )
    parser.add_argument("--mu", type=float, default=10.0, help="nudging rate")
    parser.add_argument(
        "--obs-every",
        type=int,
        default=1,
        help="steps between direct insertions",
    )
    parser.add_argument("--output-dir", default="data/nse2d_synchronization")
    return parser.parse_args()


def main():
    args = parse_args()
    model = build_model(args)
    truth0 = spun_up_truth(model, args)

    print("NSE2D low-mode synchronization benchmark")
    print("mode:", args.mode, "(mu =", args.mu, ")" if args.mode == "nudging" else "")
    print("grid:", model.shape, "dt:", args.dt, "steps:", args.n_steps)
    print("forcing mode k_f:", args.forcing_mode, "cutoff sweep:", args.cutoffs)

    results = {}
    for ka in args.cutoffs:
        res = run_synchronization(model, truth0, args, ka)
        results[ka] = res
        print(
            f"k_a={ka}: E_omega initial {res['E_omega'][0]:.3e} "
            f"-> final {res['E_omega'][-1]:.3e}, "
            f"E_Q final {res['E_Q'][-1]:.3e}, "
            f"enstrophy err final {res['enstrophy_err'][-1]:.3e}"
        )

    ka_ref = max(args.cutoffs)
    free = free_run_error(model, truth0, args, ka_ref)
    print(f"free-run (no assimilation, k_a={ka_ref} IC) final E_omega: {free:.3e}")

    try:
        out = Path(args.output_dir)
        p1 = plot_sweep(results, args, out / f"sync_error_sweep_{args.mode}.png")
        p2 = plot_mode_split(
            results[ka_ref], ka_ref, args, out / f"sync_mode_split_{args.mode}.png"
        )
        print("saved figures:", p1, "and", p2)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figure export")


if __name__ == "__main__":
    main()
