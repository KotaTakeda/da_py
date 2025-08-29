# import multiprocessing as multi
from functools import cache, partial
from multiprocessing import get_context

import numpy as np
from da.localization import calc_dist, gaspari_cohn
from da.etkf import ETKF
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
            - x_iを推定.
            - x_iに近い観測を用いる. -> localization
    localization:
        - R-locで実装. R-inverseにlocal functionをかける.
        - local functionとしてgaspari cohn function
"""


class LETKF(ETKF):
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
        """ "
        Args:
        - M: (x, dt) -> x, model dynamics
        - H: observation operator
        - R: (Ny, Ny), covariance of observation noise
        - alpha: (float>=1), multiplicative inflation parameter s.t. Pf -> alpha^2*Pf
        - store_ensemble: bool, whether to store ensemble members at each step
        - c: localization radius
        - localization: str, currently only "gaspari-cohn" is supported
        - multi_process: bool, whether to use multi processing for loop over state variables
        """
        super().__init__(M, H, R, alpha, store_ensemble)

        self.c = c
        self.localization = localization
        self.multi_process = multi_process

    # 更新/解析
    def update(self, y_obs):
        Xf = self.X.T  # (Nx, m)
        xf = Xf.mean(axis=1)

        # transformの準備
        dXf = Xf - xf[:, None]  # (Nx, m)
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)
        dy = y_obs - self._apply_H(xf)

        # 各成分でループ
        if self.multi_process:
            n_process = 4  # multi.cpu_count()
            with get_context("fork").Pool(n_process) as pl:
                process = partial(self._transform_each, dy=dy, dY=dYf, dXf=dXf)
                self.X = np.array(pl.map(process, list(range(self.Nx)))).T
                pl.close()
                pl.join()
        else:
            for i in range(self.Nx):
                self.X[:, i] = xf[i] + self._transform_each(i, dy, dYf, dXf).squeeze()

        # Save the analysis mean
        self.x.append(self.X.mean(axis=0))

    # 本体
    def _transform_each(self, i, dy, dY, dXf):
        # use _transform_T in etkf.py
        # localized Rinv
        locRinv = self._locR(i)
        # replace Rinv
        orig_Rinv = self.Rinv
        self.Rinv = locRinv

        # transform i-th component of dXf
        dXfi = dXf[i, :][None, :] # (1, m)
        dXi = self._transform_T(dy, dY, dXfi)

        # restore Rinv
        self.Rinv = orig_Rinv
        return dXi # (1, m)

    # localization用の関数
    @cache
    def _rho(self, i):
        return np.array(
            [gaspari_cohn(calc_dist(i, j, J=self.Nx), self.c) for j in range(self.Nx)]
        )

    @cache
    def _locR(self, i):
        """"rho(j - i) * Rinv [j] """
        return self._rho(i) * self.Rinv  # (Ny, Ny)

    # def calc_sqrtm(self, mat):
    #     return self._symmetric(sqrtm(self._symmetric(mat)))

    # def _symmetric(self, S):
    #     return 0.5*(S + S.T)
