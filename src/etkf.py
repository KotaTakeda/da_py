from functools import cache
import numpy as np
from numpy import eye, random, sqrt, trace
from numpy.linalg import inv
from scipy.linalg import sqrtm

# ==========================================
# EnsembleTransformKalmanFilter(ETKF)
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
        - M: (x, dt) -> x: model dynamics
        - H: observation operator
        - R: covariance of observation noise
        - alpha: (>=1): multiplicative inflation parameter s.t. Pf -> alpha*PF
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
        self._update_T(y_obs)

        # TODO: 様子を見て以下を削除
        # x_f = self.x_f[-1]
        # Xf = self.X
        # H = self.H

        # # transformの準備
        # dXf = Xf - x_f  # (m, Nx)
        # dY = (H @ dXf.T).T  # (m, Ny): 本来はH(Xf) - H(Xf).mean(axis=0)
        # dy = y_obs - H @ x_f  # (Ny, )

        # # transform
        # self.X = x_f + self._transform(dy, dY, dXf)

        # # 更新した値のアンサンブル平均xを保存,
        # self.x.append(self.X.mean(axis=0))
        # if self.store_ensemble:
        #     self.Xa.append(self.X.copy())

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
    # def _transform(self, dy, dY, dXf):
    #     P_at = inv(
    #         ((self.m - 1) / self.alpha) * self.I + dY @ self.invR @ dY.T
    #     )  # アンサンブル空間でのP_a．(m, m)
    #     T = (
    #         P_at @ dY @ self.invR @ dy + sqrtm((self.m - 1) * P_at)
    #     ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
    #     return (dXf.T @ T).T  # (m, Nx)

    # 本体
    def _transform_T(self, dy, dY, dXf):
        P_at = inv(
            ((self.m - 1) / self.alpha) * self.I + dY.T @ self.invR @ dY
        )  # アンサンブル空間でのP_a．(m, m)
        T = (
            P_at @ dY.T @ self.invR @ dy + np.real(sqrtm((self.m - 1) * P_at))
        ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
        return dXf @ T  # (Nx, m)

    # 非線形観測のハンドリングに必要
    def _apply_H(self, X):
        """X: (Nx, m)"""
        if self.linear_obs:
            return self.H @ X  # (Ny, m)
        else:
            return self.H(X.T).T  # (Ny, m)
