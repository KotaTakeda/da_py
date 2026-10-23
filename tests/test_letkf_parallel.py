from multiprocessing import get_all_start_methods

import numpy as np
import pytest

from da.letkf import LETKF

def advance(x, dt):
    return x + dt


@pytest.mark.skipif("fork" not in get_all_start_methods(), reason="requires fork")
def test_parallel_matches_serial_across_two_cycles():
    X = np.random.default_rng(7).normal(size=(4, 3)) + np.array([2, -3, 7])
    filters = [LETKF(advance, np.eye(3), np.eye(3), multi_process=parallel,
                     store_ensemble=True) for parallel in [False, True]]
    for filt in filters:
        filt.initialize(X)
    for _ in range(2):
        for filt in filters:
            filt.forecast(0.1)
            filt.update(np.array([1.0, 2.0, -1.0]))
            assert filt.X.shape == X.shape
        np.testing.assert_allclose(filters[0].X, filters[1].X, atol=1e-12)
        np.testing.assert_allclose(filters[0].x[-1], filters[1].x[-1], atol=1e-12)
