from functools import cache
import numpy as np
from numpy import eye, random, sqrt, trace
from numpy.linalg import inv
from scipy.linalg import sqrtm

# ==========================================
# LocalEnsembleTransformKalmanFilter(LETKF)
# ==========================================
"""
Parameters
M: callable(x, dt)
  状態遷移関数
H: ndarray(dim_y, dim_x)
  観測行列  
Q: ndarray(dim_x, dim_x)
  モデルの誤差共分散行列 
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列
x_0: 状態変数の初期値
P_0: 誤差共分散の初期値
m: アンサンブルメンバーの数
alpha: inflation factor
x: ndarray(dim_x)
"""


class ETKF:
    def __init__(
        self,
        M,
        H,
        R,
        x_0,
        P_0,
        m=10,
        alpha=1.0,
        seed=1,
        store_ensemble=False,
    ):
        self.M = M
        self.H = H
        self.R = R
        self.invR = inv(self.R)
        self.m = m  # アンサンブルメンバー数
        self.t = 0.0

        # 実装で技術的に必要
        self.dim_x = x_0.shape[0]
        self.I = eye(m)

        self.alpha = alpha  # inflation用の定数

        # filtering実行用
        self.x = []  # 記録用
        self.x_f = []
        self.store_ensemble = store_ensemble
        if store_ensemble:
            self.X_f = []
            self.X_a = []
        else:
            self.trP = []

        self._initialize(x_0, P_0, m, seed)

    # 初期値のサンプリング
    def _initialize(self, x_0, P_0, m, seed):
        random.seed(seed)
        self.X = x_0 + random.multivariate_normal(np.zeros_like(x_0), P_0, m)  # (m, J)
        self.x_mean = self.X.mean(axis=0)

    # 予報/時間発展
    def forecast(self, dt):
        """dt: 予測時間"""
        # アンサンブルで x(k) 予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, dt)

        self.t += dt
        self.x_mean = self.X.mean(axis=0)
        self.x_f.append(self.x_mean)
        if self.store_ensemble:
            self.X_f.append(self.X.copy())

    # 更新/解析
    def update(self, y_obs):
        x_f = self.x_mean
        X_f = self.X
        H = self.H

        # transformの準備
        dX_f = X_f - x_f  # (m, N)
        dY = (H @ dX_f.T).T  # (m, dim_y): 本来はH(X_f) - H(X_f).mean(axis=0)
        dy = y_obs - H @ x_f  # (dim_y, )
        
        # transform
        self.X = x_f + self._transform(dy, dY, dX_f)

        # 記録: 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        if self.store_ensemble:
            self.X_a.append(self.X.copy())
        else:
            self.trP.append(sqrt(trace(dX_f.T @ dX_f) / (self.dim_x - 1)))
        
    # 本体
    def _transform(self, dy, dY, dX_f):
        P_at = inv(
            ((self.m - 1) / self.alpha) * self.I + dY @ self.invR @ dY.T
        )  # アンサンブル空間でのP_a．(m, m)
        T = (
            P_at @ dY @ self.invR @ dy + sqrtm((self.m - 1) * P_at)
        ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
        return (dX_f.T @ T).T  # (m, dim_x)