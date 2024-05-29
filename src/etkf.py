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
        - H: x -> y: observation operator
        - R: x -> y: covariance of observation noise
        """
        self.M = M
        self.H = H  # NOTE: 線形を仮定
        self.R = R
        self.invR = inv(self.R)

        self.alpha = alpha  # inflation用の定数

        self.store_ensemble = store_ensemble

    # 初期アンサンブル
    def initialize(self, X_0):
        m, Nx = X_0.shape # ensemble shape
        self.Nx = Nx
        self.m = m
        self.t = 0.0
        self.X = X_0
        self.I = np.eye(m) # TODO: メモリ効率改善

        # 初期化
        self.x = []  # 記録用
        self.x_f = []
        if self.store_ensemble:
            self.X_f = []
            self.X_a = []


    # 予報/時間発展
    def forecast(self, dt):
        """dt: 予測時間"""
        # アンサンブルで予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, dt)

        self.t += dt
        self.x_f.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.X_f.append(self.X.copy())

    # 更新/解析
    def update(self, y_obs):
        x_f = self.x_f[-1]
        X_f = self.X
        H = self.H

        # transformの準備
        dX_f = X_f - x_f  # (m, N)
        dY = (H @ dX_f.T).T  # (m, dim_y): 本来はH(X_f) - H(X_f).mean(axis=0)
        dy = y_obs - H @ x_f  # (dim_y, )

        # transform
        self.X = x_f + self._transform(dy, dY, dX_f)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.X_a.append(self.X.copy())

    # 本体
    def _transform(self, dy, dY, dX_f):
        P_at = inv(
            ((self.m - 1) / self.alpha) * self.I + dY @ self.invR @ dY.T
        )  # アンサンブル空間でのP_a．(m, m)
        T = (
            P_at @ dY @ self.invR @ dy + sqrtm((self.m - 1) * P_at)
        ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
        return (dX_f.T @ T).T  # (m, Nx)
