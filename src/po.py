from functools import cache
import numpy as np
from numpy import eye, random, sqrt, trace
from numpy.linalg import inv
from scipy.linalg import sqrtm

# ==========================================
# EnsembleKalmanFilter (Perturbed Observation: PO)
# ==========================================
"""
Arguments
M: callable(x, dt)
  状態遷移関数
H: ndarray(dim_y, Nx)
  観測行列  
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列
m: アンサンブルメンバーの数
alpha: inflation factor
x: ndarray(Nx)
"""


class PO:
    def __init__(
        self,
        M,
        H,
        R,
        alpha=1.0,
        store_ensemble=False,
        additive_inflation=False,
    ):
        """
        Args:
        - M: (x, dt) -> x: model dynamics
        - H: x -> y: observation operator
        - R: x -> y: covariance of observation noise
        """
        self.M = M
        self.H = H
        self.linear_obs = isinstance(H, np.ndarray)
        if not self.linear_obs:
            self.H = np.vectorize(H, signature="(Nx)->(Ny)")

        self.R = R
        self.invR = inv(self.R)

        self.alpha = alpha  # inflation用の定数
        self.store_ensemble = store_ensemble

        self.additive_inflation = additive_inflation

    # 初期アンサンブル
    def initialize(self, X_0):
        m, Nx = X_0.shape  # ensemble shape
        self.Nx = Nx
        self.m = m
        self.t = 0.0
        self.X = X_0.copy()
        self.I = np.eye(m)  # TODO: メモリ効率改善

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
        if self.linear_obs and self.additive_inflation:
            self._update1(y_obs)
        else:
            self._update2(y_obs)

    def _update1(self, y_obs):
        # !NOTE: 転置している
        Xf = self.X.T  # (Nx, m)
        xf = Xf.mean(axis=1)
        H = self.H
        m = self.m

        dXf = Xf - xf[:, None]  # (Nx, m)
        Pf = dXf @ dXf.T / (m - 1)

        if self.alpha > 0:  # この意味のmultiplicative inflation
            Pf += self.alpha * np.eye(self.Nx)

        K = Pf @ H.T @ np.linalg.inv(H.T @ Pf @ H + self.R)  # (Nx, Ny)

        eta_rep = np.random.multivariate_normal(
            np.zeros_like(y_obs), self.R, self.m
        ).T  # (Ny, m)
        Y_rep = y_obs[:, None] + eta_rep

        Xa = Xf + K @ (Y_rep - H @ Xf)

        self.X = Xa.T  # (m, Nx)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())

    def _update2(self, y_obs):
        # !NOTE: 転置している
        Xf = self.X.T  # (Nx, m)
        xf = Xf.mean(axis=1)
        H = self.H

        # NOTE: この実装ではadditive inflationは使えない
        dXf = Xf - xf[:, None]  # (Nx, m)
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)  # (Ny, m)

        if self.alpha > 1:  # この意味のmultiplicative inflation
            dXf *= self.alpha
            dYf *= self.alpha
            # Xf = xf[:, None] + dXf

        K = dXf @ dYf.T @ np.linalg.inv(dYf @ dYf.T + (self.m - 1) * self.R)  # (Nx, Ny)

        eta_rep = np.random.multivariate_normal(
            np.zeros_like(y_obs), self.R, self.m
        ).T  # (m, Ny)
        Y_rep = y_obs[:, None] + eta_rep

        Xa = Xf + K @ (Y_rep - Yf)

        self.X = Xa.T  # (m, Nx)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.Xa.append(self.X.copy())

    # 非線形観測のハンドリングに必要
    def _apply_H(self, X):
        """X: (Nx, m)"""
        if self.linear_obs:
            return self.H @ X  # (Ny, m)
        else:
            return self.H(X.T).T  # (Ny, m)
