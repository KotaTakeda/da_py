import numpy as np
from numpy import random
from numpy.linalg import inv
import ot


class EnsembleTransformParticleFilter(object):
    def __init__(self, M, H, R, add_inflation=0.0, N_thr=1.0):
        self.M = M
        self.H = H
        self.R = R
        self.h = add_inflation
        self.N_thr = N_thr
        self.t = 0.0

    # 初期アンサンブル
    def initialize(self, X_0):
        m, dim_x = X_0.shape # ensemble shape
        self.dim_x = dim_x
        self.m = m
        self.t = 0.0
        self.X = X_0

        # 初期化
        self.x = []  # 記録用

    def forecast(self, dt):
        for i, x in enumerate(self.X):
            self.X[i] = self.M(x, dt) 
            if self.h > 0:
                self.X[i] += np.random.normal(loc=0, scale=self.h)

    def update(self, y_obs):
        self._calculate_weights(y_obs)
        self._resample_by_ot()
        self._calculate_weights(y_obs)

        self.x.append(self.W@self.X)

    def _resample_by_ot(self):
        a = self.W  # source
        b = np.ones(self.m)/self.m  # target
        dMat = self.X@self.X.T
        T = ot.emd(a, b, dMat)
        S = self.m*T  # (M, M) = (source, target)
        assert np.allclose(S.sum(axis=0), 1.0)
        self.X = S.T@self.X  # (M, 2)

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
