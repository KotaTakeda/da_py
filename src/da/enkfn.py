import numpy as np
from scipy.optimize import brentq, minimize_scalar, newton

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
    method="newton",
    initial_l1=1.0,
    xtol=1e-8,
    max_iter=100,
):
    r"""Estimate the EnKF-N anomaly inflation factor in DAPPER's dual form.

    Parameters follow the ETKF internals of this package: ``dY`` has shape
    ``(Ny, m)``, ``dy`` has shape ``(Ny,)``, and ``R`` has shape ``(Ny, Ny)``.
    The scalar optimized here is ``l1`` itself, the anomaly inflation factor.
    The corresponding covariance inflation and effective prior precision are
    ``lambda_cov = l1**2`` and ``zeta = (m - 1) / lambda_cov``.
    """
    R = np.asarray(R, dtype=float)
    R_cholesky = np.linalg.cholesky(R)
    return _estimate_l1_enkfn_dual_from_cholesky(
        dY,
        dy,
        R.shape,
        R_cholesky,
        xN=xN,
        g=g,
        method=method,
        initial_l1=initial_l1,
        xtol=xtol,
        max_iter=max_iter,
    )


def _estimate_l1_enkfn_dual_from_cholesky(
    dY,
    dy,
    R_shape,
    R_cholesky,
    xN=1.0,
    g=0.0,
    *,
    method="newton",
    initial_l1=1.0,
    xtol=1e-8,
    max_iter=100,
):
    dY = np.asarray(dY, dtype=float)
    dy = np.asarray(dy, dtype=float)
    R_cholesky = np.asarray(R_cholesky, dtype=float)

    if dY.ndim != 2:
        raise ValueError("dY must have shape (Ny, m)")
    ny, m = dY.shape
    if dy.shape != (ny,):
        raise ValueError("dy must have shape (Ny,)")
    if R_shape != (ny, ny):
        raise ValueError("R must have shape (Ny, Ny)")
    if R_cholesky.shape != (ny, ny):
        raise ValueError("R_cholesky must have shape (Ny, Ny)")
    if method not in {"newton", "brentq", "bounded"}:
        raise ValueError("method must be one of 'newton', 'brentq', or 'bounded'")

    m1 = m - 1
    if m1 <= 0:
        raise ValueError("EnKF-N requires at least two ensemble members")

    Z = np.linalg.solve(R_cholesky, dY).T
    dy_white = np.linalg.solve(R_cholesky, dy)

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

    def solve_brentq():
        lower = 1.0e-8
        upper = 1.0
        while grad(upper) < 0 and upper < 1.0e8:
            upper *= 10.0
        if grad(upper) < 0:
            return None
        l1, result = brentq(
            grad,
            lower,
            upper,
            xtol=xtol,
            maxiter=max_iter,
            full_output=True,
        )
        return float(l1), bool(result.converged), int(result.iterations)

    def solve_bounded():
        result = minimize_scalar(
            objective,
            bounds=(1.0e-8, 1.0e8),
            method="bounded",
            options={"xatol": xtol, "maxiter": max_iter},
        )
        return float(result.x), bool(result.success), int(result.nit)

    used_method = method
    if method == "newton":
        try:
            root, result = newton(
                grad,
                initial_l1,
                fprime=hess,
                tol=xtol,
                maxiter=max_iter,
                full_output=True,
            )
            l1 = float(root)
            converged = bool(result.converged) and np.isfinite(l1) and l1 > 0
            n_iter = int(result.iterations)
        except (RuntimeError, OverflowError, ZeroDivisionError, ValueError):
            converged = False
            l1 = np.nan
            n_iter = max_iter
        if not converged:
            used_method = "brentq"
            solution = solve_brentq()
            if solution is None:
                used_method = "bounded"
                l1, converged, n_iter = solve_bounded()
            else:
                l1, converged, n_iter = solution
        else:
            l1_bounded, converged_bounded, n_iter_bounded = solve_bounded()
            if objective(l1_bounded) < objective(l1):
                used_method = "bounded"
                l1 = l1_bounded
                converged = converged_bounded
                n_iter = n_iter_bounded
    elif method == "brentq":
        solution = solve_brentq()
        if solution is None:
            used_method = "bounded"
            l1, converged, n_iter = solve_bounded()
        else:
            l1, converged, n_iter = solution
    else:
        l1, converged, n_iter = solve_bounded()

    if not np.isfinite(l1) or l1 <= 0:
        used_method = "bounded"
        l1, converged, n_iter = solve_bounded()

    lambda_cov = l1**2
    zeta = m1 / lambda_cov
    info = {
        "l1": l1,
        "zeta": zeta,
        "lambda_cov": lambda_cov,
        "objective": float(objective(l1)),
        "gradient": float(grad(l1)),
        "converged": converged,
        "n_iter": n_iter,
        "requested_method": method,
        "method": used_method,
        "singular_values": s.copy(),
        "du": du.copy(),
        "eN": float(eN),
        "cL": float(cL),
    }
    return l1, info


class EnKFN(ETKF):
    """ETKF wrapper with EnKF-N adaptive anomaly inflation.

    Unlike ETKF, EnKFN does not accept a fixed multiplicative ``alpha`` in
    addition to the adaptive factor. The estimated ``l1`` is the total anomaly
    inflation applied in the ETKF transform, and ``lambda_cov = l1**2`` is the
    corresponding covariance inflation.
    """

    def __init__(
        self,
        M,
        H,
        R,
        alpha=1.0,
        xN=1.0,
        g=0.0,
        method="newton",
        store_ensemble=False,
        store_diagnostics=True,
    ):
        if alpha != 1.0:
            raise ValueError("EnKFN estimates total anomaly inflation; use alpha=1.0")
        super().__init__(M, H, R, alpha=alpha, store_ensemble=store_ensemble)
        self.R_cholesky = np.linalg.cholesky(self.R)
        self.xN = xN
        self.g = g
        self.inflation_method = method
        self.store_diagnostics = store_diagnostics
        self.inflation_diagnostics = []

    def initialize(self, X_0):
        super().initialize(X_0)
        self.inflation_diagnostics = []

    def _update_T(self, y_obs):
        Xf = self.X.T
        xf = Xf.mean(axis=1)

        dXf = Xf - xf[:, None]
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)
        dy = y_obs - self._apply_H(xf)

        l1, info = _estimate_l1_enkfn_dual_from_cholesky(
            dYf,
            dy,
            self.R.shape,
            self.R_cholesky,
            xN=self.xN,
            g=self.g,
            method=self.inflation_method,
        )

        previous_alpha = self.alpha
        self.alpha = l1
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
            info["effective_alpha"] = l1
            self.inflation_diagnostics.append(info)
