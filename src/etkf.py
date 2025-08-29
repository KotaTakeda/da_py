import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm

# ==========================================
# EnsembleTransformKalmanFilter(ETKF)
# Refs.
# - C. H. Bishop, B. J. Etherton, and S. J. Majumdar, “Adaptive Sampling with the Ensemble Transform Kalman Filter. Part I: Theoretical Aspects,” Monthly Weather Review, vol. 129, no. 3, pp. 420–436, Mar. 2001.
# - M. K. Tippett, J. L. Anderson, C. H. Bishop, T. M. Hamill, and J. S. Whitaker, “Ensemble Square Root Filters,” Monthly Weather Review, vol. 131, no. 7, pp. 1485–1490, 2003.
# ==========================================


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
        - alpha: (float>=1), multiplicative inflation parameter s.t. Pf -> alpha^2*PF
        - store_ensemble: bool, whether to store ensemble members at each step
        """
        self.M = M
        self.H = H
        self.linear_obs = isinstance(H, np.ndarray)
        if not self.linear_obs:
            self.H = np.vectorize(H, signature="(Nx)->(Ny)")

        self.R = R
        self.Rinv = inv(self.R)

        self.alpha = alpha  # inflation用の定数

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
        # dY = H @ dXf  # (m, Ny): 本来はH(Xf) - H(Xf).mean(axis=1)
        dy = y_obs - self._apply_H(xf)
        # dy = y_obs - H @ xf  # (Ny, )

        # transform
        Xa = xf[:, None] + self._transform_T(dy, dYf, dXf)  # (Nx, m)

        self.X = Xa.T  # (m, Nx)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())

    # 本体
    # def _transform_T(self, dy, dY, dXf):
    #     P_at = inv(
    #         ((self.m - 1) / self.alpha) * self.I + dY.T @ self.Rinv @ dY
    #     )  # アンサンブル空間でのP_a．(m, m)
    #     T = (
    #         P_at @ dY.T @ self.Rinv @ dy + np.real(sqrtm((self.m - 1) * P_at))
    #     ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
    #     return dXf @ T  # (Nx, m)

    def _transform_T(self, dy, dY, dXf):
        """
        Transform the ensemble perturbations to the analysis ensemble perturbations with the mean update.
        Args:
        - dy: (Ny,)
        - dY: (Nx, Ny)
        - dXf: (Nx, m)
        Returns:
        - dx + dXa = dXf @ T' (Nx, m)
        """
        m = self.m

        # S = ((m-1)/alpha)I + dY^T R^{-1} dY
        dYtRinv = dY.T @ self.Rinv  # (Ny, m)
        S = ((m - 1) / self.alpha) * self.I + dYtRinv @ dY  # (m, m)
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
