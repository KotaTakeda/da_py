import numpy as np
import pytest

from da.exkf import ExKF


def test_successive_forecasts_compose_and_restart_after_analysis():
    filt = ExKF(lambda x, dt: x+dt, np.eye(1), np.eye(1))
    filt.initialize([1.0], np.eye(1))
    filt.forecast(1)
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f, [[2], [3]])
    filt.update([5.0])
    np.testing.assert_allclose(filt.x_a, [4.0])
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f_state, [5.0])
    filt.initialize([10.0], np.eye(1))
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f, [[11.0]])


@pytest.mark.parametrize("analytic", [False, True])
def test_state_and_covariance_compose_with_model_noise(analytic):
    F = np.array([[1.0, 0.3], [0.0, 0.9]])
    Q = np.diag([0.1, 0.2])
    P = np.array([[1.0, 0.2], [0.2, 2.0]])
    x = np.array([2.0, -1.0])
    filt = ExKF(lambda x, dt: F @ x, np.eye(2), np.eye(2), Q=Q,
                jacobian=(lambda x, dt: F) if analytic else None)
    filt.initialize(x, P)
    for _ in range(2):
        filt.forecast(1)
        x = F @ x
        P = F @ P @ F.T + Q
    np.testing.assert_allclose(filt.x_f_state, x)
    np.testing.assert_allclose(filt.P, P, rtol=1e-9)
    K = P @ np.linalg.inv(P+np.eye(2))
    filt.update(np.zeros(2))
    np.testing.assert_allclose(filt.x_a, x-K @ x, rtol=1e-9)


@pytest.mark.parametrize("analytic", [False, True])
def test_nonlinear_jacobian_is_evaluated_at_each_forecast_start(analytic):
    filt = ExKF(lambda x, dt: x**2, np.eye(1), np.eye(1),
                jacobian=(lambda x, dt: np.diag(2*x)) if analytic else None)
    filt.initialize([2.0], np.eye(1))
    filt.forecast(1)
    filt.forecast(1)
    np.testing.assert_allclose(filt.x_f_state, [16.0])
    np.testing.assert_allclose(filt.P, [[1024.0]], rtol=1e-9)
