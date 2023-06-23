# import multiprocessing as multi
from functools import cache, partial
from multiprocessing import get_context

import numpy as np
from localization import calc_dist, gaspari_cohn
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
localization: localizationの設定
x: ndarray(dim_x)

Implementation:
    iteration:
        - 各観測で状態変数の数N=40回
        - 各i(in 1~40)で
            - x_iを推定．
            - x_iに近い観測を用いる．-> localization
    localization:
        - R-locで実装．R-inverseにlocal functionをかける．
        - local functionとしてgaspari cohn function
"""


class LETKF:
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
        c=3.0,
        localization="gaspari-cohn",
        addaptive=False,
        multi_process=False,
    ):
        self.M = M
        self.H = H
        self.R = R
        self.m = m  # アンサンブルメンバー数
        self.t = 0.0

        # 実装で技術的に必要
        self.dim_x = x_0.shape[0]
        self.I = eye(m)

        self.alpha = alpha  # inflation用の定数
        self.addaptive = addaptive
        self.c = c
        self.localization = localization
        self.multi_process = multi_process

        # filtering実行用
        self.x = []  # 記録用
        self.x_f = []
        self.trP = []

        self._initialize(x_0, P_0, m, seed)

    # 　初期状態
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

    # 更新/解析
    def update(self, y_obs):
        x_f = self.x_mean
        X_f = self.X
        H = self.H

        dX_f = X_f - x_f  # (m, N)
        dY = (H @ dX_f.T).T  # (m, dim_y)
        dy = y_obs - H @ x_f
        # if self.addaptive: # 不完全
        #     alpha = (trace((y_obs - dY).T@(y_obs - dY) - (m-1)*R)/trace(dY.T@dY))**2
        #     print(alpha)

        # 各成分でループ
        if self.multi_process:
            # multi.cpu_count()
            n_process = 4
            with get_context("fork").Pool(n_process) as pl:
                process = partial(self._transform_each, dy=dy, dY=dY, dX_f=dX_f)
                self.X = np.array(pl.map(process, list(range(self.dim_x)))).T
                pl.close()
                pl.join()
        else:
            for i in range(self.dim_x):
                self.X[:, i] = self.x_mean[i] + self._transform_each(i, dy, dY, dX_f)

        # 記録: 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        self.trP.append(
            sqrt(trace(dX_f.T @ dX_f) / (self.dim_x - 1))
        )  # 推定誤差共分散P_fのtraceを保存

    # 本体
    def _transform_each(self, i, dy, dY, dX_f):
        C = dY @ self._locR(i)  # localization: invRの各i行にrho_iをかける．(m, dim_y)
        P_at = inv(
            ((self.m - 1) / self.alpha) * self.I + C @ dY.T
        )  # アンサンブル空間でのP_a．(m, m)
        T = (
            P_at @ C @ dy + sqrtm((self.m - 1) * P_at)
        ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
        return (dX_f.T @ T).T[:, i]  # (m, dim_x)

    # localization用の関数
    @cache
    def _rho(self, i):
        return np.array(
            [
                gaspari_cohn(calc_dist(i, j, J=self.dim_x), self.c)
                for j in range(self.dim_x)
            ]
        )

    @cache
    def _locR(self, i):
        return self._rho(i) * inv(self.R)

    # def calc_sqrtm(self, mat):
    #     return self._symmetric(sqrtm(self._symmetric(mat)))

    # def _symmetric(self, S):
    #     return 0.5*(S + S.T)
