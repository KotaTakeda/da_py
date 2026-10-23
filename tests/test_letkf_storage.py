import numpy as np

from da.letkf import LETKF


def advance(x, dt):
    return x + dt


def test_saved_analyses_are_independent_and_reset_on_initialize():
    filt = LETKF(advance, np.eye(3), np.eye(3), store_ensemble=True)
    X = np.random.default_rng(8).normal(size=(4, 3))
    filt.initialize(X)
    filt.update(np.zeros(3))
    saved = filt.X.copy()
    filt.forecast(0.1)
    filt.update(np.ones(3))
    assert len(filt.Xa) == 2
    assert all(value.shape == X.shape for value in filt.Xa)
    np.testing.assert_array_equal(filt.Xa[0], saved)
    assert not np.shares_memory(filt.Xa[-1], filt.X)
    filt.initialize(X)
    assert filt.Xa == []


def test_disabled_storage_does_not_keep_analyses():
    filt = LETKF(advance, np.eye(3), np.eye(3))
    filt.initialize(np.ones((4, 3)))
    filt.update(np.zeros(3))
    assert not hasattr(filt, "Xa")
