import numpy as np
from numpy import random
from numpy.linalg import inv


class ParticleFilter(object):
    def __init__(self, M, h, R, add_inflation=0.0, N_thr=1.0):
        self.M = M
        self.h = h
        self.R = R
        self.sigma_add = add_inflation # <=> Q = sigma_add**2 * I_{Nx}の離散modelノイズ
        self.N_thr = N_thr
        self.t = 0.0

        # 記録用
        self.x = []
        self.resample_log = []


    # 初期アンサンブル
    def initialize(self, X_0):
        m, Nx = X_0.shape # ensemble shape
        self.Nx = Nx
        self.m = m
        self.idx = np.arange(self.m)
        self.t = 0.0
        self.X = X_0

        # 初期化
        self.x = []  # 記録用
        self.Xa = []
        self.Xf = []

    # def resampling_rate(self):
    #     return np.mean(self.resample_log)

    def forecast(self, dt):
        # 各particleで予測
        for k, x in enumerate(self.X):
            self.X[k] = self.M(x, dt)

        if self.sigma_add > 0:
            self.X += np.random.normal(loc=0, scale=self.sigma_add, size=(self.m, self.Nx)) # x^(k) + xi(k), xi(k) ~ N(0, sigma_add**2 * I_{Nx})

        self.Xf.append(self.X.copy())

    def update(self, y_obs):
        self._calculate_weights(y_obs)

        if self._caluculate_eff(self.W) < self.N_thr:
            self._resample()
            self._calculate_weights(y_obs)

        self.x.append(self.W@self.X)
        self.Xa.append(self.X)

    def _resample(self):
        reindex = np.random.choice(
            self.m, size=self.m, replace=True, p=self.W,
        )  # Weightに従ってサンプル．
        self.X = self.X[reindex]


    def _negative_log_likelihood(self, x, y_obs):
        h = self.h
        R = self.R
        dim_obs = R.shape[0]
        # return -np.log(multivariate_normal.pdf(h(x), mean=y_obs, cov=R))
        return 0.5*(y_obs - h(x)) @ inv(R) @ (y_obs - h(x)) + 0.5*np.log(np.linalg.det(R)) + 0.5*dim_obs*np.log(2*np.pi)
    
    def _calculate_weights(self, y_obs):
        W = np.array([self._negative_log_likelihood(x, y_obs) for x in self.X])
        W -= np.min(W)
        W = np.exp(-W)
        W /= W.sum()
        self.W = W

    def _caluculate_eff(self, W):
        return 1 / (W @ W) / len(W)

    # def _resample(self, W):  # この実装は精度が悪い．
    #     m = self.m
    #     reindex = []
    #     u = np.random.rand() / m
    #     for _ in range(m):
    #         reindex.append(self._F_inv(u, W))
    #         u += 1 / m
    #     return reindex

    # # 重み累積分布関数の逆関数
    # def _F_inv(self, u, W):
    #     """
    #     W: np ndarray (m, )
    #     u: float
    #     """
    #     F = W.cumsum()
    #     if u < F[0]:
    #         return 0
    #     else:
    #         return F[F < u].argmax() + 1
