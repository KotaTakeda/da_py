import numpy as np

from da.var3d import Var3D


def test_multiple_forecasts_then_analysis_and_reinitialization():
    filt = Var3D(lambda x, dt: x+dt, np.eye(1), np.eye(1))
    filt.initialize(np.array([1.0]), np.eye(1))
    filt.forecast(1)
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f, [3.0])
    filt.update(np.array([5.0]))
    np.testing.assert_allclose(filt.x_a, [4.0])
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f, [5.0])
    filt.initialize(np.array([10.0]), np.eye(1))
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f, [11.0])
    assert filt.x == []


def test_linear_composition_preserves_fixed_gain_and_single_cycle_formula():
    F = np.array([[1.0, 0.3], [0.0, 0.9]])
    H = np.array([[1.0, 0.0]])
    B = np.diag([1.0, 2.0])
    R = np.eye(1)
    x0 = np.array([2.0, -1.0])
    K = B @ H.T @ np.linalg.inv(H @ B @ H.T+R)
    for steps in (1, 2):
        filt = Var3D(lambda x, dt: F @ x, H, R)
        filt.initialize(x0, B)
        xf = x0.copy()
        for _ in range(steps):
            filt.forecast(1)
            xf = F @ xf
        np.testing.assert_allclose(filt.x_f, xf)
        filt.update(np.array([3.0]))
        np.testing.assert_allclose(filt.x_a, xf + K @ (3-H @ xf))
        np.testing.assert_array_equal(filt.B, B)
        np.testing.assert_allclose(filt.K, K)
