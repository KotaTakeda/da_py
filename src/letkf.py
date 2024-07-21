# import multiprocessing as multi
from functools import cache, partial
from multiprocessing import get_context

import numpy as np
from da.localization import calc_dist, gaspari_cohn
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
H: ndarray(dim_y, Nx)
  観測行列  
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列
m: アンサンブルメンバーの数
alpha: (>=1): multiplicative inflation parameter s.t. Pf -> alpha*PF
localization: localizationの設定
x: ndarray(Nx)

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
        alpha=1.0,
        store_ensemble=False,
        c=3.0,
        localization="gaspari-cohn",
        multi_process=False,
    ):
        self.M = M
        self.H = H
        self.R = R
        self.t = 0.0

        self.alpha = alpha  # inflation用の定数
        self.c = c
        self.localization = localization
        self.multi_process = multi_process

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
        # アンサンブルで x(k) 予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, dt)

        self.t += dt
        self.x_mean = self.X.mean(axis=0)
        self.x_f.append(self.x_mean)

    # 更新/解析
    def update(self, y_obs):
        x_f = self.x_mean
        Xf = self.X
        H = self.H

        dXf = Xf - x_f  # (m, N)
        dY = (H @ dXf.T).T  # (m, dim_y)
        dy = y_obs - H @ x_f

        # 各成分でループ
        if self.multi_process:
            # multi.cpu_count()
            n_process = 4
            with get_context("fork").Pool(n_process) as pl:
                process = partial(self._transform_each, dy=dy, dY=dY, dXf=dXf)
                self.X = np.array(pl.map(process, list(range(self.Nx)))).T
                pl.close()
                pl.join()
        else:
            for i in range(self.Nx):
                self.X[:, i] = self.x_mean[i] + self._transform_each(i, dy, dY, dXf)

        # 記録: 更新した値のアンサンブル平均xを保存,
        self.x.append(self.X.mean(axis=0))
        # self.trP.append(
        # sqrt(trace(dXf.T @ dXf) / (self.Nx - 1))
        # )  # 推定誤差共分散P_fのtraceを保存

    # 本体
    def _transform_each(self, i, dy, dY, dXf):
        C = dY @ self._locR(i)  # localization: invRの各i行にrho_iをかける．(m, dim_y)
        P_at = inv(
            ((self.m - 1) / self.alpha) * self.I + C @ dY.T
        )  # アンサンブル空間でのP_a．(m, m)
        T = (
            P_at @ C @ dy + np.real(sqrtm((self.m - 1) * P_at))
        ).T  # 注:Pythonの仕様上第１項(mean update)が行ベクトルとして足されているので転置．(m, m)
        return (dXf.T @ T).T[:, i]  # (m, Nx)

    # localization用の関数
    @cache
    def _rho(self, i):
        return np.array(
            [gaspari_cohn(calc_dist(i, j, J=self.Nx), self.c) for j in range(self.Nx)]
        )

    @cache
    def _locR(self, i):
        return self._rho(i) * inv(self.R)

    # def calc_sqrtm(self, mat):
    #     return self._symmetric(sqrtm(self._symmetric(mat)))

    # def _symmetric(self, S):
    #     return 0.5*(S + S.T)
