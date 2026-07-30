"""Original nonquadratic ETKF-N analysis of Bocquet (2011)."""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from da.etkf import ETKF


def _objective_gradient_hessian(w, dY, dy, Rinv):
    """Return Bocquet (2011), Eqs. (35), (37)--(39), in ensemble space."""
    m = w.size
    residual = dy - dY @ w
    scale = 1.0 + 1.0 / m + w @ w
    objective = 0.5 * residual @ Rinv @ residual + 0.5 * m * np.log(scale)
    gradient = -(dY.T @ Rinv @ residual) + m * w / scale
    hessian = (
        dY.T @ Rinv @ dY
        + m * (scale * np.eye(m) - 2.0 * np.outer(w, w)) / scale**2
    )
    return float(objective), gradient, hessian


class ETKFN2011(ETKF):
    """Finite-size ETKF-N using Bocquet's original non-Gaussian posterior.

    This implementation minimizes Eq. (35) of Bocquet (2011) from ``w=0``
    in the zero-sum ensemble subspace and constructs the analysis anomalies
    from the inverse Hessian at the optimum.  It is distinct from
    :class:`da.enkfn.EnKFN`, which estimates a scalar inflation using the later
    Gaussian-scale-mixture dual formulation.
    """

    def __init__(
        self,
        M,
        H,
        R,
        alpha=1.0,
        store_ensemble=False,
        store_diagnostics=True,
        optimizer_options=None,
        Q=None,
        rng=None,
    ):
        if alpha != 1.0:
            raise ValueError("ETKFN2011 includes finite-size adaptation; use alpha=1.0")
        super().__init__(
            M, H, R, alpha=alpha, store_ensemble=store_ensemble, Q=Q, rng=rng
        )
        if not self.linear_obs:
            raise TypeError("ETKFN2011 currently requires a linear observation matrix H")
        self.store_diagnostics = store_diagnostics
        self.optimizer_options = {"gtol": 1.0e-10, **dict(optimizer_options or {})}
        self.analysis_diagnostics = []

    def initialize(self, X_0):
        super().initialize(X_0)
        self.analysis_diagnostics = []

    def _update_T(self, y_obs):
        Xf = self.X.T
        xf = Xf.mean(axis=1)
        dXf = Xf - xf[:, None]
        dYf = self.H @ dXf
        dy = np.asarray(y_obs, dtype=float) - self.H @ xf

        # Qg spans {w : 1^T w = 0}; optimizing there fixes the ensemble gauge.
        Qg = null_space(np.ones((1, self.m)))

        def terms_reduced(u):
            value, gradient, hessian = _objective_gradient_hessian(
                Qg @ u, dYf, dy, self.Rinv
            )
            return value, Qg.T @ gradient, Qg.T @ hessian @ Qg

        result = minimize(
            lambda u: terms_reduced(u)[0],
            np.zeros(self.m - 1),
            jac=lambda u: terms_reduced(u)[1],
            hess=lambda u: terms_reduced(u)[2],
            method="trust-exact",
            options=self.optimizer_options,
        )
        optimizer_method = "trust-exact"
        if not result.success:
            fallback_options = {
                key: value
                for key, value in self.optimizer_options.items()
                if key in {"gtol", "maxiter"}
            }
            result = minimize(
                lambda u: terms_reduced(u)[0],
                np.asarray(result.x, dtype=float),
                jac=lambda u: terms_reduced(u)[1],
                method="BFGS",
                options=fallback_options,
            )
            optimizer_method = "BFGS"
        if not result.success:
            raise RuntimeError(
                "ETKFN2011 analysis optimization failed before convergence: "
                f"{result.message}"
            )
        u = np.asarray(result.x, dtype=float)
        w = Qg @ u
        objective, gradient, hessian = _objective_gradient_hessian(
            w, dYf, dy, self.Rinv
        )
        hessian_reduced = 0.5 * (Qg.T @ hessian @ Qg + Qg.T @ hessian.T @ Qg)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian_reduced)
        if eigenvalues[0] <= 0:
            raise np.linalg.LinAlgError(
                "ETKFN2011 analysis Hessian is not positive definite at the optimum"
            )

        hessian_inverse = (eigenvectors * eigenvalues**-1.0) @ eigenvectors.T
        Wa = (
            np.sqrt(self.m - 1)
            * Qg
            @ ((eigenvectors * eigenvalues**-0.5) @ eigenvectors.T)
            @ Qg.T
        )
        xa = xf + dXf @ w
        Xa = xa[:, None] + dXf @ Wa
        self.X = Xa.T
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())

        if self.store_diagnostics:
            self.analysis_diagnostics.append(
                {
                    "objective": objective,
                    "gradient_norm": float(np.linalg.norm(Qg.T @ gradient)),
                    "n_iter": int(result.nit),
                    "converged": bool(result.success),
                    "message": str(result.message),
                    "optimizer_method": optimizer_method,
                    "weights": w.copy(),
                    "gauge_residual": float(np.sum(w)),
                    "hessian_eigenvalues": eigenvalues.copy(),
                    "analysis_covariance": dXf @ Qg @ hessian_inverse @ Qg.T @ dXf.T,
                }
            )
