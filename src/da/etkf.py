import numpy as np
from numpy.linalg import inv

# ==========================================
# EnsembleTransformKalmanFilter(ETKF)
# Refs.
# - C. H. Bishop, B. J. Etherton, and S. J. Majumdar, “Adaptive Sampling with the Ensemble Transform Kalman Filter. Part I: Theoretical Aspects,” Monthly Weather Review, vol. 129, no. 3, pp. 420–436, Mar. 2001.
# - M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill, and J. S. Whitaker, “Ensemble Square Root Filters,” Monthly Weather Review, vol. 131, no. 7, pp. 1485–1490, 2003.
# ==========================================


def _batch_observation(H):
    """Wrap a per-state observation callable to accept ensemble batches.

    Same contract as the previous ``np.vectorize(H, signature="(Nx)->(Ny)")``
    wrapper — a 1D state maps to a 1D observation and a stacked (m, Nx) batch
    maps to (m, Ny) by applying ``H`` per member — but without np.vectorize's
    per-call dispatch overhead.
    """

    def apply(X):
        X = np.asarray(X)
        if X.ndim == 1:
            return np.asarray(H(X))
        return np.stack([np.asarray(H(x)) for x in X])

    return apply


class ETKF:
    def __init__(
        self,
        M,
        H,
        R,
        alpha=1.0,
        store_ensemble=False,
    ):
        """
        Args:
        - M: (x, dt) -> x, model dynamics
        - H: observation operator
        - R: (Ny, Ny), covariance of observation noise
        - alpha: (float>=1), anomaly inflation factor, A -> alpha*A and Pf -> alpha^2*Pf
        - store_ensemble: bool, whether to store ensemble members at each step
        """
        self.M = M
        self.H = H
        self.linear_obs = isinstance(H, np.ndarray)
        if not self.linear_obs:
            self.H = _batch_observation(H)

        self.R = R
        self.Rinv = inv(self.R)

        self.alpha = alpha  # anomaly inflation factor

        self.store_ensemble = store_ensemble

    # 初期アンサンブル
    def initialize(self, X_0):
        m, Nx = X_0.shape  # ensemble shape
        self.Nx = Nx
        self.m = m
        self.t = 0.0
        self.X = X_0.copy()
        self.I = np.eye(m)

        # 初期化
        self.x = []  # 記録用
        self.x_f = []
        if self.store_ensemble:
            self.X0 = X_0.copy()
            self.Xf = []
            self.Xa = []

    # 予報/時間発展
    def forecast(self, dt):
        """dt: 予測時間"""
        # アンサンブルで予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, dt)

        self.t += dt
        self.x_f.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xf.append(self.X.copy())

    def perturb_forecast(self, eta):
        """Add per-member perturbations to the forecast ensemble.

        Intended for additive Gaussian model noise (see ``da.noise``),
        applied once per assimilation cycle between the last ``forecast``
        step and ``update``. The most recent recorded forecast diagnostics
        (``x_f`` and, with ``store_ensemble=True``, ``Xf``) are refreshed so
        they stay consistent with the perturbed ensemble the analysis
        consumes.

        Args:
        - eta: (m, Nx), one perturbation per ensemble member.
        """
        eta = np.asarray(eta, dtype=float)
        if eta.shape != self.X.shape:
            raise ValueError(
                f"eta must match the ensemble shape {self.X.shape}, "
                f"got {eta.shape}"
            )
        self.X = self.X + eta
        if self.x_f:
            self.x_f[-1] = self.X.mean(axis=0)
        if self.store_ensemble and self.Xf:
            self.Xf[-1] = self.X.copy()

    # 更新/解析
    def update(self, y_obs):
        self._update_T(y_obs)

    def _update_T(self, y_obs):
        Xf = self.X.T  # (Nx, m)
        xf = Xf.mean(axis=1)
        # H = self.H

        # transformの準備
        dXf = Xf - xf[:, None]  # (Nx, m)
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)
        dy = y_obs - self._apply_H(xf)

        # transform
        Xa = xf[:, None] + self._transform_T(dy, dYf, dXf)  # (Nx, m)

        self.X = Xa.T  # (m, Nx)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())

    # 本体
    def _transform_T(self, dy, dY, dXf):
        """
        Transform the ensemble perturbations to the analysis ensemble perturbations with the mean update.
        Args:
        - dy: (Ny,)
        - dY: (Nx, Ny)
        - dXf: (Nx, m)
        Returns:
        - dx + dXa = dXf @ T': (Nx, m)
        """
        m = self.m
        cov_inflation = self.alpha**2

        # alpha inflates anomalies: A -> alpha*A, hence Pf -> alpha^2*Pf.
        # In ensemble space this is S = ((m-1)/alpha^2)I + dY^T R^{-1} dY.
        dYtRinv = dY.T @ self.Rinv  # (Ny, m)
        S = ((m - 1) / cov_inflation) * self.I + dYtRinv @ dY  # (m, m)
        S = 0.5 * (S + S.T)  # for numerical stability

        # Symmetric eigenvalue decomposition
        s, U = np.linalg.eigh(S)  # S = U diag(s) U^T
        eps = 1e-60
        s = np.clip(s, eps, None)

        # Symmetric square root：sqrt((m-1) P_at) = sqrt((m-1) S^{-1}) = U diag(sqrt((m-1)/s)) U^T
        T_sqrt = (U * np.sqrt((m - 1) / s)) @ U.T  # (m, m) symmetric

        # Mean part: T_mean = P_at dm, dm = dY^T R^{-1} dy
        dm = dYtRinv @ dy  # (m,)
        T_mean = U @ ((U.T @ dm) / s)  # (m,), divide by s element-wise

        # Transformation including the mean update T = w_mean + T_sqrt
        T = T_mean[:, None] + T_sqrt  # (m, m)
        return dXf @ T  # (Nx, m)

    # 非線形観測のハンドリングに必要
    def _apply_H(self, X):
        """X: (Nx, m)"""
        if self.linear_obs:
            return self.H @ X  # (Ny, m)
        else:
            return self.H(X.T).T  # (Ny, m)
