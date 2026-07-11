"""Additive Gaussian model noise for ensemble forecasts.

Sampling engine behind the ``Q=`` argument of the ensemble filters
(``ETKF``, ``EnKFN``, ``LETKF``), which implement the stochastic forecast
model

    x <- M(x, dt) + eta,    eta ~ i.i.d. N(0, Q),

with one independent perturbation per ensemble member at every
``forecast(dt)`` step — the same per-step timing as ``ExKF``'s ``Q`` in the
covariance propagation. The deterministic analysis transforms are untouched.

    filt = ETKF(M, H, R, alpha=1.02, Q=Q, rng=np.random.default_rng(seed))

This is additive *stochastic* inflation (Gaussian model noise). It is distinct
from the multiplicative anomaly inflation ``alpha`` of the ensemble filters
(``A -> alpha A``), from the deterministic additive covariance regularization
of ``PO`` (``Pf -> Pf + alpha I``), and from the observation perturbations
drawn by stochastic EnKF variants; see ``docs/model_noise.md``.
"""

import numpy as np

# Relative tolerances for validating a dense covariance: symmetry is checked
# against the largest entry, and eigenvalues down to -PSD_TOL * max|eig| are
# treated as zero (rank-deficient Q such as sigma^2 * projection is allowed).
_SYMMETRY_RTOL = 1.0e-8
_PSD_RTOL = 1.0e-10


class GaussianModelNoise:
    """Sampler for zero-mean Gaussian model noise with covariance ``Q``.

    ``Q`` may be a dense symmetric positive-semidefinite matrix of shape
    ``(Nx, Nx)`` — rank-deficient covariances are supported — or a 1-D array
    of per-component variances of shape ``(Nx,)`` for diagonal noise. The
    factorization is computed once at construction and reused across steps.

    For fully custom noise (e.g. a user-supplied sampler), no wrapper is
    needed: add the samples to ``filt.X`` directly in the driver loop.
    """

    def __init__(self, Q):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim == 1:
            if Q.shape[0] == 0:
                raise ValueError("Q must have at least one component")
            if np.any(Q < 0):
                raise ValueError(
                    "diagonal Q must have non-negative variances; "
                    f"min entry is {Q.min()}"
                )
            self.Nx = Q.shape[0]
            self._diag_std = np.sqrt(Q)
            self._factor = None
        elif Q.ndim == 2:
            if Q.shape[0] != Q.shape[1]:
                raise ValueError(f"dense Q must be square, got shape {Q.shape}")
            scale = np.abs(Q).max() if Q.size else 0.0
            # rtol=0: only the scale-relative atol decides, otherwise numpy's
            # default rtol=1e-5 would accept triangles differing by ~1e-5 of
            # the entry magnitude on large matrices.
            if not np.allclose(Q, Q.T, rtol=0.0, atol=_SYMMETRY_RTOL * max(scale, 1.0)):
                raise ValueError("dense Q must be symmetric")
            # Symmetrize so eigh (which reads one triangle) sees the same
            # matrix regardless of which triangle carries the tolerated
            # rounding asymmetry.
            Q = 0.5 * (Q + Q.T)
            eigvals, eigvecs = np.linalg.eigh(Q)
            eig_scale = np.abs(eigvals).max() if eigvals.size else 0.0
            if eigvals.min() < -_PSD_RTOL * max(eig_scale, 1.0):
                raise ValueError(
                    "dense Q must be positive semidefinite; "
                    f"smallest eigenvalue is {eigvals.min()}"
                )
            eigvals = np.clip(eigvals, 0.0, None)
            self.Nx = Q.shape[0]
            self._diag_std = None
            # factor @ z ~ N(0, Q): factor factor^T = V diag(w) V^T = Q
            self._factor = eigvecs * np.sqrt(eigvals)
        else:
            raise ValueError(
                f"Q must be 1-D (diagonal variances) or 2-D (dense), got ndim={Q.ndim}"
            )

    def sample(self, rng, size):
        """Draw ``size`` independent perturbations, one row per member.

        Args:
        - rng: ``numpy.random.Generator`` (project RNG convention).
        - size: number of ensemble members.

        Returns an array of shape ``(size, Nx)`` whose rows are i.i.d.
        ``N(0, Q)``.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError(
                "rng must be a numpy.random.Generator "
                "(e.g. numpy.random.default_rng(seed)); legacy RandomState and "
                "the global numpy.random module are not accepted"
            )
        z = rng.standard_normal((size, self.Nx))
        if self._diag_std is not None:
            return z * self._diag_std
        return z @ self._factor.T
