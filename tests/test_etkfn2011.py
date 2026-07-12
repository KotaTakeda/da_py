import numpy as np

from da.etkf import ETKF
from da.etkfn2011 import ETKFN2011, _objective_gradient_hessian


def _finite_gradient(fun, x, eps=1e-6):
    out = np.empty_like(x)
    for j in range(x.size):
        step = np.zeros_like(x)
        step[j] = eps
        out[j] = (fun(x + step) - fun(x - step)) / (2 * eps)
    return out


def test_objective_gradient_and_hessian_finite_differences():
    rng = np.random.default_rng(4)
    dY = rng.normal(size=(3, 5))
    dY -= dY.mean(axis=1, keepdims=True)
    dy = rng.normal(size=3)
    Rinv = np.diag([1.0, 0.5, 2.0])
    w = rng.normal(size=5)
    w -= w.mean()

    value, gradient, hessian = _objective_gradient_hessian(w, dY, dy, Rinv)
    numeric_gradient = _finite_gradient(
        lambda z: _objective_gradient_hessian(z, dY, dy, Rinv)[0], w
    )
    numeric_hessian = np.column_stack(
        [
            _finite_gradient(
                lambda z: _objective_gradient_hessian(z, dY, dy, Rinv)[1][j], w
            )
            for j in range(w.size)
        ]
    ).T
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numeric_gradient, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(hessian, numeric_hessian, rtol=2e-5, atol=2e-6)


def test_analysis_mean_covariance_and_gauge_consistency():
    rng = np.random.default_rng(8)
    X0 = rng.normal(size=(8, 4))
    H = rng.normal(size=(3, 4))
    R = np.diag([0.7, 1.1, 1.4])
    filt = ETKFN2011(lambda x, dt: x, H, R, store_ensemble=True)
    filt.initialize(X0)
    filt.update(np.array([0.4, -0.3, 0.8]))

    diag = filt.analysis_diagnostics[-1]
    assert abs(diag["gauge_residual"]) < 1e-12
    assert diag["gradient_norm"] < 1e-6
    assert np.all(diag["hessian_eigenvalues"] > 0)
    np.testing.assert_allclose(filt.X.mean(axis=0), filt.x[-1], atol=1e-13)
    np.testing.assert_allclose(
        (filt.X - filt.X.mean(axis=0)).sum(axis=0), np.zeros(4), atol=1e-13
    )
    np.testing.assert_allclose(
        np.cov(filt.X, rowvar=False), diag["analysis_covariance"], atol=1e-12
    )


def test_rejects_fixed_inflation_and_nonlinear_observation():
    H = np.eye(2)
    R = np.eye(2)
    with np.testing.assert_raises(ValueError):
        ETKFN2011(lambda x, dt: x, H, R, alpha=1.1)
    with np.testing.assert_raises(TypeError):
        ETKFN2011(lambda x, dt: x, lambda x: x, R)


def test_large_ensemble_limit_approaches_etkf():
    rng = np.random.default_rng(12)
    X0 = rng.normal(size=(200, 3))
    H = np.eye(3)
    R = np.eye(3)
    y = np.array([0.2, -0.1, 0.5])
    etkf = ETKF(lambda x, dt: x, H, R)
    etkfn = ETKFN2011(lambda x, dt: x, H, R)
    etkf.initialize(X0)
    etkfn.initialize(X0)
    etkf.update(y)
    etkfn.update(y)

    np.testing.assert_allclose(etkfn.X.mean(axis=0), etkf.X.mean(axis=0), atol=3e-4)
    np.testing.assert_allclose(
        np.cov(etkfn.X, rowvar=False),
        np.cov(etkf.X, rowvar=False),
        atol=8e-4,
    )
