"""Large-ensemble multiplicative-inflation ETKF under partial observation (#28).

Tests the multiplicative-anomaly ETKF (``ETKF(alpha=...)``, ``A -> alpha*A``,
NO additive inflation) in the regime the rank argument of #25 identifies as
theoretically relevant: ensemble sizes above the observed dimension,

    m - 1 >= rank(H) = q_low,

where for the ``kelly_32`` reference configuration the low-mode observation
has ``q_low = (2*5+1)^2 = 121``. Below that bound ``H Pf H^T`` cannot be full
rank on the observed space and multiplicative inflation alone cannot maintain
an observed covariance lower bound; this script sweeps ``m`` across the bound
(and ``alpha``) to distinguish

1. ``m <= rank(H)``  -- elementary rank obstruction, expected inaccurate;
2. ``m >  rank(H)``  -- theoretically admissible regime;
3. large ``m`` but still inaccurate -- failure due to dynamics/collapse/
   inflation choice rather than rank.

Configurations come from ``nse2d_reference_configs`` (#27): ``kelly_32`` (the
#25 protocol: climatological initial ensemble, J=20, gamma=0.01, kmax=5) and
``inubushi_32`` (stronger turbulence, same protocol style). This issue does
not claim Kelly reproduction unless the low-mode error reaches the PO
additive-inflation scale (~2e-2); results are recorded to CSV/NPZ either way.

Outputs under ``--output-dir``: ``sweep.csv`` (one row per run), ``sweep.npz``
(relerr curves), ``relerr_curves.png``, ``final_vs_alpha.png``.

A quick check runs via ``--smoke`` (reduced grid/cycles/ensemble).
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from da.etkf import ETKF
from da.loss import loss_rms
from nse2d_reference_configs import REFERENCE_CONFIGS


def spun_up_state(model, cfg, extra_perturbation=True):
    xx, yy = model.grid()
    omega = model.kolmogorov_vorticity(mode=cfg.forcing_mode)
    if extra_perturbation:
        omega = omega + 0.1 * np.sin(xx + yy) + 0.1 * np.cos(2 * xx - yy)
    for _ in range(cfg.spin_up):
        omega = model.step(omega, cfg.dt)
    return omega


def climatological_members(model, omega_spun, cfg, n_members):
    """n_members decorrelated attractor snapshots + a truth IC one gap later."""
    members = []
    omega = omega_spun.copy()
    for _ in range(n_members):
        for _ in range(cfg.init_decorrelate):
            omega = model.step(omega, cfg.dt)
        members.append(omega.reshape(-1).copy())
    for _ in range(cfg.init_decorrelate):
        omega = model.step(omega, cfg.dt)
    return np.stack(members), omega


def truth_trajectory(model, omega0, cfg, n_cycles):
    states = [omega0.reshape(-1).copy()]
    omega = omega0.copy()
    for _ in range(n_cycles):
        for _ in range(cfg.obs_interval):
            omega = model.step(omega, cfg.dt)
        states.append(omega.reshape(-1).copy())
    return np.stack(states)


def make_observation(model, case, cfg):
    if case == "low":
        return model.independent_low_mode_observation(kmax=cfg.kmax_obs)
    if case == "high":
        return model.high_mode_observation(kmax=cfg.kmax_obs)
    if case == "full":
        return model.full_mode_observation()
    raise ValueError(f"unknown case {case!r}")


def run_etkf(model, cfg, case, truth, X0, alpha, rng, n_cycles):
    """One multiplicative-inflation ETKF run; returns per-cycle diagnostics."""
    obs = make_observation(model, case, cfg)
    H = obs.as_matrix()
    R = cfg.gamma**2 * np.eye(obs.obs_dim)
    y = np.stack([H @ x for x in truth[1:]])
    y = y + cfg.gamma * rng.standard_normal(y.shape)

    dt_obs = cfg.dt * cfg.obs_interval
    M = model.as_forecast(internal_steps=cfg.obs_interval)
    etkf = ETKF(M, H, R, alpha=alpha)
    etkf.initialize(X0.copy())

    def diag(X, x_true):
        mean = X.mean(axis=0)
        err = (mean - x_true).reshape(model.shape)
        return {
            "relerr": float(np.linalg.norm(err) / np.linalg.norm(x_true)),
            "rmse": float(loss_rms(mean, x_true)),
            "err_obs": float(np.linalg.norm(model.project_low_modes(err, cfg.kmax_obs))),
            "err_unobs": float(
                np.linalg.norm(model.project_high_modes(err, cfg.kmax_obs))
            ),
            "spread": float(np.sqrt(np.mean(X.var(axis=0, ddof=1)))),
        }

    series = {key: [val] for key, val in diag(etkf.X, truth[0]).items()}
    for n in range(1, n_cycles + 1):
        etkf.forecast(dt_obs)
        etkf.update(y[n - 1])
        for key, val in diag(etkf.X, truth[n]).items():
            series[key].append(val)
    return {key: np.asarray(vals) for key, vals in series.items()}


def plot_curves(rows, curves, args, path):
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.8, height=3.6)
        for row in rows:
            key = f"m{row['m']}_a{row['alpha']}"
            ax.semilogy(
                curves[key],
                label=rf"$m={row['m']}$, $\alpha={row['alpha']}$"
                + (" (rank ok)" if row["m_gt_rank"] else ""),
            )
        ax.set_xlabel("assimilation cycle")
        ax.set_ylabel("relative error")
        ax.set_title(f"{args.config}, case={args.case}, multiplicative ETKF")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def plot_final_vs_alpha(rows, args, path):
    from da import viz

    with viz.style_context():
        fig, ax = viz.single_panel(width=5.0, height=3.4)
        for m in sorted({row["m"] for row in rows}):
            pts = sorted(
                (row["alpha"], row["final_relerr"]) for row in rows if row["m"] == m
            )
            ax.semilogy(
                [a for a, _ in pts], [e for _, e in pts], marker="o", label=rf"$m={m}$"
            )
        ax.set_xlabel(r"multiplicative inflation $\alpha$")
        ax.set_ylabel("final relative error")
        ax.set_title(f"{args.config}, case={args.case}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    return viz.save_png(fig, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-ensemble multiplicative-inflation ETKF sweep (#28).",
    )
    parser.add_argument(
        "--config", choices=sorted(REFERENCE_CONFIGS), default="kelly_32"
    )
    parser.add_argument("--case", choices=["low", "high", "full"], default="low")
    parser.add_argument(
        "--m-list",
        type=int,
        nargs="+",
        default=[48, 122, 160],
        help="ensemble sizes; include values above q_low=121 for kelly_32",
    )
    parser.add_argument(
        "--alpha-list",
        type=float,
        nargs="+",
        default=[1.0, 1.05, 1.1, 1.2, 1.3, 1.6],
        help="multiplicative anomaly inflation values",
    )
    parser.add_argument("--n-cycles", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny fast configuration (reduced spin-up/cycles/ensembles)",
    )
    parser.add_argument("--output-dir", default="data/nse2d_etkf_large_ensemble")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = REFERENCE_CONFIGS[args.config]
    if args.smoke:
        cfg = replace(cfg, spin_up=100, init_decorrelate=20)
        args.m_list = [8]
        args.alpha_list = [1.1]
        args.n_cycles = 3

    model = cfg.build_model()
    obs_dim = make_observation(model, args.case, cfg).obs_dim
    print(f"config: {cfg.name}, case: {args.case}, obs_dim (rank H): {obs_dim}")
    for m in args.m_list:
        print(f"  m={m}: m > obs_dim -> {m > obs_dim} (m-1 >= obs_dim -> {m - 1 >= obs_dim})")
    print(f"alpha sweep: {args.alpha_list}, cycles: {args.n_cycles}, seed: {args.seed}")

    rng = np.random.default_rng(args.seed)
    omega_spun = spun_up_state(model, cfg)
    # One climatology pool sized for the largest ensemble; smaller runs use
    # its leading members so all runs share the same truth trajectory.
    pool, omega0 = climatological_members(model, omega_spun, cfg, max(args.m_list))
    truth = truth_trajectory(model, omega0, cfg, args.n_cycles)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for m in args.m_list:
        for alpha in args.alpha_list:
            res = run_etkf(
                model, cfg, args.case, truth, pool[:m], alpha, rng, args.n_cycles
            )
            row = {
                "config": cfg.name,
                "case": args.case,
                "obs_dim": obs_dim,
                "m": m,
                "m_gt_rank": m > obs_dim,
                "alpha": alpha,
                "final_relerr": res["relerr"][-1],
                "time_mean_rmse": float(res["rmse"].mean()),
                "final_rmse": res["rmse"][-1],
                "final_spread": res["spread"][-1],
                "final_err_obs": res["err_obs"][-1],
                "final_err_unobs": res["err_unobs"][-1],
                "seed": args.seed,
            }
            rows.append(row)
            curves[f"m{m}_a{alpha}"] = res["relerr"]
            print(
                f"m={m:>4} alpha={alpha:<5}: final relerr {row['final_relerr']:.3e}, "
                f"spread {row['final_spread']:.3e}, "
                f"err_obs {row['final_err_obs']:.3e}"
            )

    csv_path = out / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(out / "sweep.npz", **curves)
    print("saved:", csv_path, "and sweep.npz")

    try:
        p1 = plot_curves(rows, curves, args, out / "relerr_curves.png")
        p2 = plot_final_vs_alpha(rows, args, out / "final_vs_alpha.png")
        print("saved figures:", p1, ",", p2)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("matplotlib is not installed; skipping figures")


if __name__ == "__main__":
    main()
