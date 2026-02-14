"""
Particle Filter (Sequential Monte Carlo/Sequential Importance Resampling)
"""

import numpy as np


class ParticleFilter(object):
    def __init__(
        self,
        M,
        h,
        R,
        add_inflation=0.0,
        N_thr=1.0,
        resample_option="systematic",  # ["multinomial", "systematic", "residual"]
    ):
        self.M = M
        self.h = h
        self.R = R
        self.sigma_add = add_inflation  # <=> Q = sigma_add**2 * I_{Nx}の離散modelノイズ
        self.N_thr = N_thr
        self.resample_option = resample_option
        self.t = 0.0

    # 初期アンサンブル
    def initialize(self, X_0):
        m, Nx = X_0.shape  # ensemble shape
        self.Nx = Nx
        self.m = m
        self.Rinv = np.linalg.inv(self.R)
        self.idx = np.arange(self.m)
        self.t = 0.0

        self.X = X_0.copy()
        self.W = np.ones(m) / m

        # 初期化
        self.X0 = X_0.copy()
        self.x = []
        self.Xa = []
        self.Xf = []

        # 記録用
        # self.resample_log = []

    # def resampling_rate(self):
    #     return np.mean(self.resample_log)

    def forecast(self, dt):
        # 各particleで予測
        for k, x in enumerate(self.X):
            self.X[k] = self.M(x, dt)

        if self.sigma_add > 0:
            # x^(k) + xi(k), xi(k) ~ N(0, sigma_add**2 * I_{Nx})
            self.X += np.random.normal(
                loc=0, scale=self.sigma_add, size=(self.m, self.Nx)
            )

        self.Xf.append(self.X.copy())

    def update(self, y_obs):
        self._calculate_weights(y_obs)

        if self._caluculate_eff(self.W) < self.N_thr * self.m:
            self._resample()
            self._calculate_weights(y_obs)

        self.x.append(self.W @ self.X)
        self.Xa.append(self.X)

    def _resample(self):
        # Ref: Doucet, A., Johansen, A.M., n.d. A Tutorial on Particle Filtering and Smoothing: Fifteen years later. in Stoyanov, J., 2012. The Oxford Handbook of Nonlinear Filtering. Journal of the Royal Statistical Society: Series A (Statistics in Society) 175, 824–825. https://doi.org/10.1111/j.1467-985X.2012.01045_6.x
        if self.resample_option == "multinomial":
            reindex = self._multinomial_resample(self.W)
        elif self.resample_option == "systematic":
            reindex = self._systematic_resample(self.W)
        elif self.resample_option == "residual":
            reindex = self._residual_resample(self.W)
        else:
            raise ValueError(f"Unknown resample option: {self.resample_option}")

        self.X = self.X[reindex]
        # initialize weight
        self.W = np.ones(self.m) / self.m

    def _multinomial_resample(self, W):
        m = len(W)
        return np.random.choice(m, size=m, replace=True, p=W)

    def _systematic_resample(self, W):
        m = len(W)
        cumulative = np.cumsum(W)
        cumulative[-1] = 1.0
        u0 = np.random.rand() / m
        positions = u0 + np.arange(m) / m
        return np.searchsorted(cumulative, positions)

    def _residual_resample(self, W):
        m = len(W)
        N = np.floor(m * W).astype(int)
        R_sum = N.sum()
        indices = np.repeat(np.arange(m), N)

        if R_sum < m:
            residual = m * W - N
            residual /= residual.sum()
            extra = np.random.choice(m, size=m - R_sum, replace=True, p=residual)
            indices = np.concatenate([indices, extra])
        return indices

    def _negative_log_likelihood(self, x, y_obs):
        h = self.h
        Rinv = self.Rinv
        dy = y_obs - h(x)
        # dim_obs = R.shape[0]
        # return -np.log(multivariate_normal.pdf(h(x), mean=y_obs, cov=R))
        return (
            0.5
            * dy
            @ Rinv
            @ dy
            # + 0.5 * np.log(np.linalg.det(R)) # const.
            # + 0.5 * dim_obs * np.log(2 * np.pi) # const.
        )

    def _calculate_weights(self, y_obs):
        # nll_i = -log p(y_obs | x_i)
        nll = np.array([self._negative_log_likelihood(x, y_obs) for x in self.X])

        # normalize
        nll_min = np.min(nll)
        w_shift = np.exp(-(nll - nll_min))

        self.W = w_shift / w_shift.sum()

    def _caluculate_eff(self, W):
        return 1 / np.sum(W**2)
