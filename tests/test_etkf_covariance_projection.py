import numpy as np
import pytest

from da.etkf import ETKF


def _identity_model(x, dt):
    return x


def _ensemble():
    return np.array(
        [
            [-1.0, 1.5],
            [0.0, -2.0],
            [1.0, 0.5],
            [2.0, 3.0],
            [3.0, -1.0],
        ]
    )


def _analyze(X, H, R, **kwargs):
    filt = ETKF(_identity_model, H, R, **kwargs)
    filt.initialize(X)
    filt.update(np.array([0.75]))
    return filt.X


def test_none_projection_reproduces_omitted_projection_exactly():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])

    omitted = _analyze(X.copy(), H, R)
    explicit_none = _analyze(X.copy(), H, R, covariance_projection=None)

    np.testing.assert_array_equal(explicit_none, omitted)


def test_identity_projection_reproduces_existing_etkf():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])

    baseline = _analyze(X.copy(), H, R, alpha=1.2)
    projected = _analyze(
        X.copy(),
        H,
        R,
        alpha=1.2,
        covariance_projection=np.eye(2),
    )

    np.testing.assert_allclose(projected, baseline, rtol=1.0e-12, atol=1.0e-12)


def test_projection_confines_mean_increment_and_analysis_anomalies():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])
    P = np.diag([1.0, 0.0])

    analysis = _analyze(X.copy(), H, R, covariance_projection=P)
    analysis_mean = analysis.mean(axis=0)
    analysis_anomalies = analysis - analysis_mean

    np.testing.assert_allclose(analysis_mean[1], X.mean(axis=0)[1], atol=1.0e-12)
    np.testing.assert_allclose(analysis_anomalies[:, 1], 0.0, atol=1.0e-12)


def test_projected_analysis_covariance_matches_inflated_kalman_formula():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])
    P = np.diag([1.0, 0.0])
    alpha = 1.3

    analysis = _analyze(
        X.copy(),
        H,
        R,
        alpha=alpha,
        covariance_projection=P,
    )
    forecast_anomalies = X - X.mean(axis=0)
    forecast_covariance = forecast_anomalies.T @ forecast_anomalies / (len(X) - 1)
    projected_covariance = alpha**2 * P @ forecast_covariance @ P
    gain = (
        projected_covariance
        @ H.T
        @ np.linalg.inv(H @ projected_covariance @ H.T + R)
    )
    expected_covariance = projected_covariance - gain @ H @ projected_covariance

    analysis_anomalies = analysis - analysis.mean(axis=0)
    actual_covariance = analysis_anomalies.T @ analysis_anomalies / (len(X) - 1)

    np.testing.assert_allclose(
        actual_covariance, expected_covariance, rtol=1.0e-10, atol=1.0e-12
    )


@pytest.mark.parametrize(
    ("projection", "message"),
    [
        (np.eye(3), "shape"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), "finite"),
        (np.array([[1.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.diag([1.0, 0.5]), "idempotent"),
        (np.diag([0.0, 1.0]), "incompatible"),
    ],
)
def test_invalid_covariance_projection_is_rejected(projection, message):
    filt = ETKF(
        _identity_model,
        np.array([[1.0, 0.0]]),
        np.array([[1.0]]),
        covariance_projection=projection,
    )

    with pytest.raises(ValueError, match=message):
        filt.initialize(_ensemble())


def test_projection_rejects_callable_observation_operator():
    with pytest.raises(NotImplementedError, match="linear matrix"):
        ETKF(
            _identity_model,
            lambda x: x[:1],
            np.array([[1.0]]),
            covariance_projection=np.eye(2),
        )
