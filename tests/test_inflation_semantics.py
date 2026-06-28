import numpy as np

from da.etkf import ETKF


def test_etkf_alpha_is_anomaly_inflation_factor():
    X0 = np.array([[-1.0], [1.0]])
    H = np.array([[1.0]])
    R = np.array([[1.0e12]])
    alpha = 1.5

    etkf = ETKF(lambda x, dt: x, H, R, alpha=alpha)
    etkf.initialize(X0)
    etkf.update(np.array([0.0]))

    xf = X0.mean(axis=0)
    dxf = X0 - xf
    dxa = etkf.X - etkf.X.mean(axis=0)

    np.testing.assert_allclose(dxa, alpha * dxf, rtol=1.0e-10, atol=1.0e-10)
