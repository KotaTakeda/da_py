import numpy as np
from numpy import random
from numpy.linalg import inv


class ParticleFilter(object):
    def __init__(self, M, H, R, x_0, P_0, m, add_inflation=0.0, seed=1, N_thr=1.0):
        self.M = M
        self.H = H
        self.R = R
        self.h = add_inflation
        self.N_thr = N_thr
        self.m = m
        self.idx = np.arange(self.m)
        self.t = 0.0

        # 記録用
        self.x = []
        self.resample_log = []

        # initialize ensemble
        self._initialize(x_0, P_0, m, seed)

    # 　初期状態
    def _initialize(self, x_0, P_0, m, seed):
        random.seed(seed)
        self.X = x_0 + random.multivariate_normal(np.zeros_like(x_0), P_0, m)  # (m, dim_x)
        self.x_mean = self.X.mean(axis=0)

    # def resampling_rate(self):
    #     return np.mean(self.resample_log)

    def forecast(self, dt):
        # 各particleで予測
        for i, x in enumerate(self.X):
            self.X[i] = self.M(x, dt) 
            if self.h > 0:
                self.X[i] += np.random.normal(loc=0, scale=self.h)

    def update(self, y_obs):
        self._calculate_weights(y_obs)

        if self._caluculate_eff(self.W) < self.N_thr:  # NOTE: 100%resampleする
            # reindex = self._resample(W)
            reindex = self._resample_by_choice(self.W)
            self.X = self.X[reindex]
            self._calculate_weights(y_obs)

        self.x.append(self.W@self.X)

    def _resample_by_choice(self, W):
        reindex = np.random.choice(
            self.m, size=self.m, replace=True, p=W,
        )  # Weightに従ってサンプル．
        return reindex

    def _negative_log_likelihood(self, x, y_obs):
        H = self.H
        R = self.R
        dim_obs = R.shape[0]
        # return -np.log(multivariate_normal.pdf(H@x, mean=y_obs, cov=R))
        return 0.5*(y_obs - H @ x) @ inv(R) @ (y_obs - H @ x) + 0.5*np.log(np.linalg.det(R)) + 0.5*dim_obs*np.log(2*np.pi)
    
    def _calculate_weights(self, y_obs):
        nega_log_w = np.array([self._negative_log_likelihood(x, y_obs) for x in self.X])
        W = np.exp(-nega_log_w)
        W += 1e-300
        # W[W < 1e-6] = 1e-6
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
