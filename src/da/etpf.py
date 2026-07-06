import numpy as np
import ot
from .pf import ParticleFilter


class EnsembleTransformParticleFilter(ParticleFilter):
    def _resample(self):
        a = self.W  # source weights
        b = np.ones(self.m) / self.m  # uniform target weights
        # ETPF cost is the squared Euclidean distance between particles, so the
        # optimal-transport plan performs the *minimal* rearrangement of the
        # weighted forecast ensemble into a uniformly weighted analysis ensemble
        # (Reich, 2013). Using the Gram matrix X @ X.T here would instead
        # maximise the transport distance.
        dMat = ot.dist(self.X, self.X, metric="sqeuclidean")
        T = ot.emd(a, b, dMat)
        S = self.m * T  # (M, M) = (source, target)
        # assert np.allclose(S.sum(axis=0), 1.0)
        self.X = S.T @ self.X  # (M, 2)
