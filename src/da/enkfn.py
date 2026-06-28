import numpy as np
from scipy.optimize import minimize_scalar

from da.etkf import ETKF


def _pad0(a, n):
    out = np.zeros(n, dtype=float)
    out[: min(len(a), n)] = a[: min(len(a), n)]
    return out


def _hyperprior_coeffs(s, m, xN=1.0, g=0.0):
    """DAPPER-style EnKF-N hyperprior coefficients."""
    if m <= 1:
        raise ValueError("EnKF-N requires at least two ensemble members")
    if xN <= 0:
        raise ValueError("xN must be positive")

    m1 = m - 1
    eN = (m + 1) / m
    cL = (m + g) / m1

    prior_mode = eN / cL
    diagonal = _pad0(s**2, m) + m1
    i_kh = np.mean(diagonal**-1) * m1
    mode_correction = np.sqrt(prior_mode**i_kh)

    eN = eN / mode_correction * xN
    cL = cL * mode_correction * xN
    return eN, cL


def estimate_l1_enkfn_dual(
    dY,
    dy,
    R,
    xN=1.0,
    g=0.0,
    *,
    initial_l1=1.0,
    xtol=1e-8,
    max_iter=100,
):
    r"""Estimate the EnKF-N anomaly inflation factor in DAPPER's dual form.

    Parameters follow the ETKF internals of this package: ``dY`` has shape
    ``(Ny, m)``, ``dy`` has shape ``(Ny,)``, and ``R`` has shape ``(Ny, Ny)``.
    The returned ``l1`` inflates anomalies, so covariance inflation is
    ``lambda_cov = l1**2``.
    """
    dY = np.asarray(dY, dtype=float)
    dy = np.asarray(dy, dtype=float)
    R = np.asarray(R, dtype=float)

    if dY.ndim != 2:
        raise ValueError("dY must have shape (Ny, m)")
    ny, m = dY.shape
    if dy.shape != (ny,):
        raise ValueError("dy must have shape (Ny,)")
    if R.shape != (ny, ny):
        raise ValueError("R must have shape (Ny, Ny)")

    m1 = m - 1
    if m1 <= 0:
        raise ValueError("EnKF-N requires at least two ensemble members")

    L = np.linalg.cholesky(R)
    Z = np.linalg.solve(L, dY).T
    dy_white = np.linalg.solve(L, dy)

    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    du = vt @ dy_white
    rk = len(s)
    eN, cL = _hyperprior_coeffs(s, m, xN=xN, g=g)

    s2 = s**2
    s4 = s2**2

    def dgn(l1):
        return (l1 * s) ** 2 + m1

    def objective(l1):
        return np.sum(du[:rk] ** 2 / dgn(l1)) + eN / l1**2 + cL * np.log(l1**2)

    def grad(l1):
        return (
            -2 * l1 * np.sum(s2 * du[:rk] ** 2 / dgn(l1) ** 2)
            - 2 * eN / l1**3
            + 2 * cL / l1
        )

    def hess(l1):
        return (
            8 * l1**2 * np.sum(s4 * du[:rk] ** 2 / dgn(l1) ** 3)
            + 6 * eN / l1**4
            - 2 * cL / l1**2
        )

    l1 = float(initial_l1)
    converged = False
    method = "newton"
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        gp = grad(l1)
        hp = hess(l1)
        if abs(gp) <= xtol:
            converged = True
            break
        if not np.isfinite(hp) or hp <= 0:
            break
        step = gp / hp
        candidate = l1 - step
        if not np.isfinite(candidate) or candidate <= 0:
            break
        if objective(candidate) > objective(l1):
            break
        l1 = candidate
        if abs(step) <= xtol * max(1.0, abs(l1)):
            converged = True
            break

    if not converged:
        method = "bounded"
        result = minimize_scalar(
            objective,
            bounds=(1.0e-8, 1.0e8),
            method="bounded",
            options={"xatol": xtol, "maxiter": max_iter},
        )
        l1 = float(result.x)
        converged = bool(result.success)
        n_iter = int(result.nit)

    lambda_cov = l1**2
    info = {
        "lambda_cov": lambda_cov,
        "objective": float(objective(l1)),
        "gradient": float(grad(l1)),
        "converged": converged,
        "n_iter": n_iter,
        "method": method,
        "singular_values": s.copy(),
        "du": du.copy(),
        "eN": float(eN),
        "cL": float(cL),
    }
    return l1, info


class EnKFN(ETKF):
    def __init__(
        self,
        M,
        H,
        R,
        alpha=1.0,
        xN=1.0,
        g=0.0,
        store_ensemble=False,
        store_diagnostics=True,
    ):
        super().__init__(M, H, R, alpha=alpha, store_ensemble=store_ensemble)
        self.xN = xN
        self.g = g
        self.store_diagnostics = store_diagnostics
        self.inflation_diagnostics = []

    def _update_T(self, y_obs):
        Xf = self.X.T
        xf = Xf.mean(axis=1)

        dXf = Xf - xf[:, None]
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)
        dy = y_obs - self._apply_H(xf)

        l1, info = estimate_l1_enkfn_dual(dYf, dy, self.R, xN=self.xN, g=self.g)
        effective_alpha = self.alpha * l1

        previous_alpha = self.alpha
        self.alpha = effective_alpha
        try:
            Xa = xf[:, None] + self._transform_T(dy, dYf, dXf)
        finally:
            self.alpha = previous_alpha

        self.X = Xa.T
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())
        if self.store_diagnostics:
            info = dict(info)
            info["l1"] = l1
            info["effective_alpha"] = effective_alpha
            self.inflation_diagnostics.append(info)
