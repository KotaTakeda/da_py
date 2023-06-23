import numpy as np
from numpy import random, sqrt, trace
from numpy.linalg import inv

# NOTE: not tested


class ParticleFilter:
    def __init__(self, M, H, R, x_0, P_0, m, seed=1, N_thr=1.0):
        self.M = M
        self.H = H
        self.R = R
        self.N_thr = N_thr
        self.m = m
        self.idx = np.arange(self.m)
        self.t = 0.0

        # 記録用
        self.x = []
        self.trP = []
        self.resample_log = []

        # initialize ensemble
        self._initialize(x_0, P_0, m, seed)

    # 　初期状態
    def _initialize(self, x_0, P_0, m, seed):
        random.seed(seed)
        self.X = x_0 + random.multivariate_normal(np.zeros_like(x_0), P_0, m)  # (m, J)
        self.x_mean = self.X.mean(axis=0)

    # def resampling_rate(self):
    #     return np.mean(self.resample_log)

    def forecast(self, dt):
        self._compute_trP()
        # 各particleで予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(
                s, dt
            )  # + np.random.normal(loc=0, scale=1e-5/self.trP[-1])

    def update(self, y_obs):
        nega_log_w = np.array([self._negative_log_likelihood(x, y_obs) for x in self.X])
        nega_log_w = nega_log_w - nega_log_w.min()
        W = np.exp(-nega_log_w)
        # print("---")
        # print(nega_log_w.max(), nega_log_w.min())
        # print(W.max(), W.min(), W.sum())
        # W[W < 1e-6] = 1e-6
        W = W / W.sum()  # 規格化
        # print(W.max(), W.min(), self._caluculate_eff(W))

        # if True or self._caluculate_eff(W) < self.N_thr:  # NOTE: 100%resampleする
        # reindex = self._resample(W)
        reindex = self._resample_by_choice(W)
        self.X = self.X[reindex]

        self.x.append(self.X.mean(axis=0))

    def _resample_by_choice(self, W):
        reindex = np.random.choice(
            self.idx, size=self.m, p=W, replace=True
        )  # Weightに従ってサンプル．
        return reindex

    def _negative_log_likelihood(self, x, y_obs):
        H = self.H
        R = self.R
        m = self.m
        return (y_obs - H @ x) @ inv(R) @ (y_obs - H @ x) / 2
        # return (y_obs - H @ x) @ inv(R) @ (y_obs - H @ x) / (m - 1)

    def _caluculate_eff(self, W):
        return 1 / (W @ W)

    def _compute_trP(self):
        dX = self.X - self.X.mean(axis=0)
        self.trP.append(np.sqrt(np.trace(dX.T @ dX) / (self.m - 1)))

    def _resample(self, W):  # この実装は精度が悪い．
        m = self.m
        reindex = []
        u = np.random.rand() / m
        for _ in range(m):
            reindex.append(self._F_inv(u, W))
            u += 1 / m
        return reindex

    # 重み累積分布関数の逆関数
    def _F_inv(self, u, W):
        """
        W: np ndarray (m, )
        u: float
        """
        F = W.cumsum()
        if u < F[0]:
            return 0
        else:
            return F[F < u].argmax() + 1
