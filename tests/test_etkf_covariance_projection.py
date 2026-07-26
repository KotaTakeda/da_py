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


def _analyze(X, H, R, y, **kwargs):
    filt = ETKF(_identity_model, H, R, **kwargs)
    filt.initialize(X)
    filt.update(y)
    return filt


def test_disabled_projection_reproduces_omitted_projection_exactly():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])
    y = np.array([0.75])

    omitted = _analyze(X.copy(), H, R, y)
    explicitly_disabled = _analyze(X.copy(), H, R, y, project_covariance=False)

    np.testing.assert_array_equal(explicitly_disabled.X, omitted.X)


def test_full_state_observation_reproduces_existing_etkf():
    X = _ensemble()
    H = np.eye(2)
    R = 0.4 * np.eye(2)
    y = np.array([0.75, -0.25])

    baseline = _analyze(X.copy(), H, R, y, alpha=1.2)
    projected = _analyze(X.copy(), H, R, y, alpha=1.2, project_covariance=True)

    np.testing.assert_allclose(
        projected._covariance_projection, np.eye(2), rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(projected.X, baseline.X, rtol=1.0e-12, atol=1.0e-12)


def test_projection_confines_mean_increment_and_analysis_anomalies():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])

    filt = _analyze(X.copy(), H, R, np.array([0.75]), project_covariance=True)
    analysis_mean = filt.X.mean(axis=0)
    analysis_anomalies = filt.X - analysis_mean

    np.testing.assert_allclose(
        filt._covariance_projection, np.diag([1.0, 0.0]), atol=1.0e-15
    )
    np.testing.assert_allclose(analysis_mean[1], X.mean(axis=0)[1], atol=1.0e-12)
    np.testing.assert_allclose(analysis_anomalies[:, 1], 0.0, atol=1.0e-12)


def test_projected_analysis_covariance_matches_inflated_kalman_formula():
    X = _ensemble()
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.4]])
    alpha = 1.3

    filt = _analyze(
        X.copy(),
        H,
        R,
        np.array([0.75]),
        alpha=alpha,
        project_covariance=True,
    )
    P = filt._covariance_projection
    forecast_anomalies = X - X.mean(axis=0)
    forecast_covariance = forecast_anomalies.T @ forecast_anomalies / (len(X) - 1)
    projected_covariance = alpha**2 * P @ forecast_covariance @ P
    gain = (
        projected_covariance @ H.T @ np.linalg.inv(H @ projected_covariance @ H.T + R)
    )
    expected_covariance = projected_covariance - gain @ H @ projected_covariance

    analysis_anomalies = filt.X - filt.X.mean(axis=0)
    actual_covariance = analysis_anomalies.T @ analysis_anomalies / (len(X) - 1)

    np.testing.assert_allclose(
        actual_covariance, expected_covariance, rtol=1.0e-10, atol=1.0e-12
    )


def test_rank_deficient_observation_builds_row_space_projection():
    H = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    filt = ETKF(_identity_model, H, np.eye(2), project_covariance=True)
    expected = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(
        filt._covariance_projection, expected, rtol=1.0e-14, atol=1.0e-14
    )


def test_projected_anomalies_are_recentered_for_large_offset_ensemble():
    offset = np.array([1.0e12, -1.0e12])
    X = _ensemble() + offset
    H = np.eye(2)
    filt = ETKF(_identity_model, H, np.eye(2), project_covariance=True)
    filt.initialize(X)
    captured = {}
    transform = filt._transform_T

    def capture_transform(dy, dY, dXf):
        captured["dY"] = dY.copy()
        captured["dXf"] = dXf.copy()
        return transform(dy, dY, dXf)

    filt._transform_T = capture_transform
    filt.update(offset)

    np.testing.assert_allclose(
        captured["dXf"].mean(axis=1), 0.0, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(captured["dY"].mean(axis=1), 0.0, rtol=0.0, atol=1.0e-15)


def test_projection_rejects_callable_observation_operator():
    with pytest.raises(NotImplementedError, match="linear matrix"):
        ETKF(
            _identity_model,
            lambda x: x[:1],
            np.array([[1.0]]),
            project_covariance=True,
        )


def test_projection_rejects_nonfinite_observation_matrix():
    with pytest.raises(ValueError, match="finite"):
        ETKF(
            _identity_model,
            np.array([[1.0, np.nan]]),
            np.array([[1.0]]),
            project_covariance=True,
        )


def test_projection_rejects_state_dimension_mismatch():
    filt = ETKF(
        _identity_model,
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[1.0]]),
        project_covariance=True,
    )

    with pytest.raises(ValueError, match="state dimension"):
        filt.initialize(_ensemble())
