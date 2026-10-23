import numpy as np
import pytest

from da.enkfn import EnKFN
from da.etkf import ETKF
from da.letkf import LETKF


@pytest.mark.parametrize("filter_class", [ETKF, EnKFN, LETKF])
@pytest.mark.parametrize("nx", [1, 2])
def test_zero_ensemble_observation_innovation_preserves_mean(filter_class, nx):
    X = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, -1.0]])[:, :nx]
    H = lambda x: x**2  # noqa: E731
    options = {"observation_positions": np.arange(nx)} if filter_class is LETKF else {}
    filt = filter_class(lambda x, dt: x, H, np.eye(nx), **options)
    filt.initialize(X)
    filt.update(np.mean([H(x) for x in X], axis=0))
    assert filt.X.shape == X.shape
    np.testing.assert_allclose(filt.X.mean(axis=0), X.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("filter_class", [ETKF, EnKFN, LETKF])
def test_linear_callable_and_matrix_observations_agree(filter_class):
    X = np.random.default_rng(42).normal(size=(6, 3))
    H = np.eye(3)[[0, 2]]
    options = {"observation_positions": [0, 2]} if filter_class is LETKF else {}
    filters = [filter_class(lambda x, dt: x, obs, np.eye(2), **options)
               for obs in (H, lambda x: H @ x)]
    for filt in filters:
        filt.initialize(X)
        filt.update(np.array([0.1, -0.2]))
    np.testing.assert_allclose(filters[0].X, filters[1].X, atol=1e-12)
