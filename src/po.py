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
        # !NOTE: 転置している
        X = self.X.T # (Nx, m)
        H = self.H

        # NOTE: この実装ではadditive inflationは使えない
        dX = X - X.mean(axis=1, keepdims=True)  # (Nx, m)
        dX *= self.alpha
        dY = H@dX  # (Ny, m): 本来はH(X) - H(X).mean(axis=1)
        K = dX@dY.T@np.linalg.inv(dY@dY.T + self.R) # (Nx, Ny)

        eta_rep = np.random.multivariate_normal(np.zeros_like(y_obs), self.R, self.m).T # (m, Ny)
        Y_rep = y_obs[:, None] + eta_rep

        X = X + K@(Y_rep - H@X)

        self.X = X.T  # (m, Nx)

        # 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.X_a.append(self.X.copy())