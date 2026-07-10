"""Additive Gaussian model noise for ensemble forecasts.

Implements the stochastic forecast model

    x_{n+1}^{f,(k)} = M(x_n^{a,(k)}) + eta_{n+1}^{(k)},
    eta_{n+1}^{(k)} ~ i.i.d. N(0, Q),

by sampling one independent perturbation per ensemble member. Sampling is
kept out of the deterministic analysis transforms: the driver loop applies it
between the last forecast step of an assimilation cycle and the analysis
update, which is the only place the cycle boundary is known,

    noise = GaussianModelNoise(Q)
    for each cycle:
        for _ in range(n_obs):
            filt.forecast(dt)
        filt.X += noise.sample(rng, m)
        filt.update(y_obs)

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
    factorization is computed once at construction and reused across cycles.

    For fully custom noise (e.g. a user-supplied sampler), no wrapper is
    needed: add the samples to the ensemble directly in the driver loop.
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
            if not np.allclose(Q, Q.T, atol=_SYMMETRY_RTOL * max(scale, 1.0)):
                raise ValueError("dense Q must be symmetric")
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


def sample_model_noise(rng, Q, size):
    """Draw ``size`` i.i.d. ``N(0, Q)`` perturbations, one row per member.

    Convenience wrapper around :class:`GaussianModelNoise` for one-off use;
    construct the class once instead when sampling every cycle, so the
    factorization of ``Q`` is reused.
    """
    return GaussianModelNoise(Q).sample(rng, size)
