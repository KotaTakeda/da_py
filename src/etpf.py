import numpy as np
import ot
from .pf import ParticleFilter


class EnsembleTransformParticleFilter(ParticleFilter):
    def _resample(self):
        a = self.W  # source
        b = np.ones(self.m) / self.m  # target
        dMat = self.X @ self.X.T
        T = ot.emd(a, b, dMat)
        S = self.m * T  # (M, M) = (source, target)
        # assert np.allclose(S.sum(axis=0), 1.0)
        self.X = S.T @ self.X  # (M, 2)
