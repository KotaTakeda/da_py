# import multiprocessing as multi
from functools import partial
from multiprocessing import get_context

import numpy as np
from .localization import calc_dist, gaspari_cohn
from .etkf import ETKF

# ==========================================
# LocalEnsembleTransformKalmanFilter(LETKF)
# TODO: update as Hunt, B.R., Kostelich, E.J., Szunyogh, I., 2007. Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. Physica D: Nonlinear Phenomena, Data Assimilation 230, 112–126. https://doi.org/10.1016/j.physd.2006.11.00
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
alpha: (>=1): anomaly inflation factor, A -> alpha*A and Pf -> alpha^2*Pf
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
        Q=None,
        rng=None,
        observation_positions=None,
    ):
        """ "
        Args:
        - M: (x, dt) -> x, model dynamics
        - H: observation operator
        - R: (Ny, Ny), covariance of observation noise
        - alpha: (float>=1), anomaly inflation factor, A -> alpha*A and Pf -> alpha^2*Pf
        - store_ensemble: bool, whether to store ensemble members at each step
        - c: localization radius
        - localization: str, currently only "gaspari-cohn" is supported
        - multi_process: bool, whether to use multi processing for loop over state variables
        - Q, rng: model-noise covariance and Generator, applied per forecast
          step via the inherited ETKF forecast (see da.etkf.ETKF)
        - observation_positions: length-Ny locations in grid-index units on
          the periodic ring [0, Nx). Inferred for one-hot selection matrices;
          required for other matrices or callable observation operators.
        """
        if np.isnan(c) or c <= 0:
            raise ValueError("c must be positive (infinity disables localization)")
        if localization != "gaspari-cohn":
            raise ValueError("localization must be 'gaspari-cohn'")
        super().__init__(M, H, R, alpha, store_ensemble, Q=Q, rng=rng)

        self.c = c
        self.localization = localization
        self.multi_process = multi_process
        self._observation_positions_input = observation_positions

    def initialize(self, X_0):
        super().initialize(X_0)
        ny = self.R.shape[0]
        positions = self._observation_positions_input
        if self.linear_obs and self.H.shape != (ny, self.Nx):
            raise ValueError("H must have shape (Ny, Nx) compatible with R and X")
        if positions is None:
            if not self.linear_obs or not (
                np.all((self.H == 0) | (self.H == 1))
                and np.all(np.sum(self.H, axis=1) == 1)
            ):
                raise ValueError(
                    "observation_positions is required unless H is a one-hot "
                    "selection matrix; supply physically meaningful local positions"
                )
            positions = np.argmax(self.H, axis=1)
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (ny,) or not np.all(np.isfinite(positions)):
            raise ValueError("observation_positions must be a finite length-Ny vector")
        if np.any(positions < 0) or np.any(positions >= self.Nx):
            raise ValueError("observation_positions must lie in [0, Nx)")
        self.observation_positions = positions.copy()

    # 更新/解析
    def update(self, y_obs):
        Xf = self.X.T  # (Nx, m)
        xf = Xf.mean(axis=1)

        # transformの準備
        dXf = Xf - xf[:, None]  # (Nx, m)
        Yf = self._apply_H(Xf)
        dYf = Yf - Yf.mean(axis=1, keepdims=True)
        dy = y_obs - Yf.mean(axis=1)

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
        # use _transform_T in etkf.py; alpha has anomaly-inflation semantics there.
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
    def _rho(self, i):
        return np.array(
            [gaspari_cohn(calc_dist(i, j, J=self.Nx), self.c)
             for j in self.observation_positions]
        )

    def _locR(self, i):
        """"rho(j - i) * Rinv [j] """
        return self._rho(i) * self.Rinv  # (Ny, Ny)

    # def calc_sqrtm(self, mat):
    #     return self._symmetric(sqrtm(self._symmetric(mat)))

    # def _symmetric(self, S):
    #     return 0.5*(S + S.T)
