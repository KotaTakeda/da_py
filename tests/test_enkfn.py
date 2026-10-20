import numpy as np
import pytest

import da.enkfn as enkfn_module
from da.enkfn import EnKFN, estimate_l1_enkfn_dual
from da.etkf import ETKF


def test_estimate_l1_enkfn_dual_returns_positive_anomaly_inflation():
    dY = np.array([[-1.0, 0.0, 1.0], [0.5, -1.0, 0.5]])
    dy = np.array([0.25, -0.75])
    R = np.diag([0.5, 2.0])

    l1, info = estimate_l1_enkfn_dual(dY, dy, R)

    assert l1 > 0
    np.testing.assert_allclose(info["l1"], l1)
    np.testing.assert_allclose(info["lambda_cov"], l1**2)
    np.testing.assert_allclose(info["zeta"], (dY.shape[1] - 1) / l1**2)
    assert info["requested_method"] == "newton"


def test_estimate_l1_enkfn_dual_is_deterministic():
    dY = np.array([[-1.0, 0.0, 1.0], [0.5, -1.0, 0.5]])
    dy = np.array([0.25, -0.75])
    R = np.diag([0.5, 2.0])

    l1_a, info_a = estimate_l1_enkfn_dual(dY, dy, R)
    l1_b, info_b = estimate_l1_enkfn_dual(dY, dy, R)

    np.testing.assert_allclose(l1_a, l1_b)
    np.testing.assert_allclose(info_a["objective"], info_b["objective"])


def test_enkfn_update_returns_same_ensemble_shape_as_etkf():
    X0 = np.array([[-1.0, 0.0], [0.0, 0.25], [1.0, -0.25]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.5]])
    y = np.array([0.1])

    etkf = ETKF(lambda x, dt: x, H, R)
    etkf.initialize(X0)
    etkf.update(y)

    enkfn = EnKFN(lambda x, dt: x, H, R)
    enkfn.initialize(X0)
    enkfn.update(y)

    assert enkfn.X.shape == etkf.X.shape == X0.shape
    assert len(enkfn.inflation_diagnostics) == 1
    assert enkfn.inflation_diagnostics[0]["l1"] > 0
    np.testing.assert_allclose(
        enkfn.inflation_diagnostics[0]["effective_alpha"],
        enkfn.inflation_diagnostics[0]["l1"],
    )


def test_enkfn_rejects_fixed_alpha_to_avoid_double_inflation():
    with pytest.raises(ValueError, match="estimates total anomaly inflation"):
        EnKFN(lambda x, dt: x, np.array([[1.0]]), np.array([[1.0]]), alpha=1.2)


def test_enkfn_reuses_cholesky_factor_between_updates(monkeypatch):
    calls = 0
    original_cholesky = np.linalg.cholesky

    def counting_cholesky(a):
        nonlocal calls
        calls += 1
        return original_cholesky(a)

    monkeypatch.setattr(np.linalg, "cholesky", counting_cholesky)

    X0 = np.array([[-1.0], [0.0], [1.0]])
    enkfn = EnKFN(lambda x, dt: x, np.array([[1.0]]), np.array([[0.5]]))
    enkfn.initialize(X0)
    enkfn.update(np.array([0.1]))
    enkfn.update(np.array([0.2]))

    assert calls == 1


def test_enkfn_initialize_resets_inflation_diagnostics():
    X0 = np.array([[-1.0], [0.0], [1.0]])
    enkfn = EnKFN(lambda x, dt: x, np.array([[1.0]]), np.array([[0.5]]))

    enkfn.initialize(X0)
    enkfn.update(np.array([0.1]))
    assert len(enkfn.inflation_diagnostics) == 1

    enkfn.initialize(X0)
    assert enkfn.inflation_diagnostics == []


@pytest.mark.parametrize("method", ["brentq", "bounded"])
def test_estimate_l1_enkfn_dual_rejects_removed_solvers(method):
    with pytest.raises(ValueError, match="method must be 'newton'"):
        estimate_l1_enkfn_dual(
            np.array([[-1.0, 1.0]]), np.array([2.0]), np.eye(1), method=method
        )


def test_estimate_l1_enkfn_dual_hessian_matches_gradient_difference(monkeypatch):
    original_newton = enkfn_module.newton

    def checked_newton(func, x0, *, fprime, **kwargs):
        for value in [0.5, 1.0, 2.0]:
            step = 1.0e-5 * value
            difference = (func(value + step) - func(value - step)) / (2 * step)
            np.testing.assert_allclose(fprime(value), difference, rtol=1.0e-7)
        return original_newton(func, x0, fprime=fprime, **kwargs)

    monkeypatch.setattr(enkfn_module, "newton", checked_newton)
    _, info = estimate_l1_enkfn_dual(
        np.array([[-1.0, 1.0]]), np.array([2.0]), np.eye(1)
    )
    assert info["converged"]
    assert abs(info["gradient"]) < 1.0e-8


def test_estimate_l1_enkfn_dual_retains_unconverged_positive_iterate():
    l1, info = estimate_l1_enkfn_dual(
        np.array([[-1.0, 1.0]]), np.array([2.0]), np.eye(1), max_iter=1
    )
    assert np.isfinite(l1) and l1 > 0
    assert not info["converged"]
    assert info["n_iter"] == 1
    assert info["method"] == info["requested_method"] == "newton"


@pytest.mark.parametrize("root", [float("nan"), float("inf"), 0.0, -1.0])
def test_estimate_l1_enkfn_dual_rejects_invalid_newton_result(monkeypatch, root):
    def invalid_newton(*args, **kwargs):
        return root, None

    monkeypatch.setattr(enkfn_module, "newton", invalid_newton)
    with pytest.raises(RuntimeError, match="finite positive l1"):
        estimate_l1_enkfn_dual(
            np.array([[-1.0, 1.0]]), np.array([2.0]), np.eye(1)
        )


def test_estimate_l1_enkfn_dual_multimodal_solution_depends_on_initial_value():
    dY = np.array([[-7.1e-5, 7.1e-5]])
    dy = np.array([10.0])
    R = np.eye(1)
    local, local_info = estimate_l1_enkfn_dual(dY, dy, R)
    other, other_info = estimate_l1_enkfn_dual(dY, dy, R, initial_l1=74000.0)

    assert local_info["converged"] and other_info["converged"]
    np.testing.assert_allclose(local, 1.0, atol=1.0e-6)
    np.testing.assert_allclose(other, 74339.70337, rtol=1.0e-6)
    assert other_info["objective"] < local_info["objective"] - 50
