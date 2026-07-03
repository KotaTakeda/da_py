"""Tests for the time-stepping utilities in da.scheme.

Test-only (issue #10). Each scheme is checked against a closed-form solution
or a classical order/conservation property on a simple system, so the tests
are deterministic and independent of implementation details.
"""

import numpy as np

from da.scheme import euler, implicit_midpoint, rk4


def _linear(t, x, a):
    return a * x


def test_euler_exact_for_constant_rhs():
    x1 = euler(lambda t, x: np.array([2.0]), 0.0, np.array([1.0]), (), 0.5)
    np.testing.assert_allclose(x1, [2.0])


def test_euler_is_first_order():
    # One step of dx/dt = x from x=1: local error O(dt^2), so halving dt
    # shrinks the error by ~4.
    a, dt = 1.0, 0.1
    e = abs(euler(_linear, 0.0, np.array([1.0]), (a,), dt)[0] - np.exp(a * dt))
    e_half = abs(
        euler(_linear, 0.0, np.array([1.0]), (a,), dt / 2)[0] - np.exp(a * dt / 2)
    )
    ratio = e / e_half
    assert 3.0 < ratio < 5.0


def test_rk4_exact_for_cubic_time_polynomial():
    # RK4's quadrature is Simpson's rule in t, exact for f(t) = 4 t^3, so one
    # step reproduces x(t) = t^4 exactly.
    def f(t, x):
        return np.array([4.0 * t**3])

    x1 = rk4(f, 1.0, np.array([1.0]), (), 0.5)
    np.testing.assert_allclose(x1, [1.5**4], rtol=1e-12)


def test_rk4_local_error_is_fifth_order():
    # One step of dx/dt = x: local error O(dt^5), so halving dt shrinks the
    # error by ~32.
    a, dt = 1.0, 0.2
    e = abs(rk4(_linear, 0.0, np.array([1.0]), (a,), dt)[0] - np.exp(a * dt))
    e_half = abs(
        rk4(_linear, 0.0, np.array([1.0]), (a,), dt / 2)[0] - np.exp(a * dt / 2)
    )
    ratio = e / e_half
    assert 25.0 < ratio < 40.0


def test_rk4_matches_exp_closely():
    # Sanity: 10 RK4 steps of dx/dt = x over t in [0, 1] reproduce e to
    # global O(dt^4) accuracy (measured error ~8e-7 at dt=0.1).
    x = np.array([1.0])
    dt = 0.1
    for k in range(10):
        x = rk4(_linear, k * dt, x, (1.0,), dt)
    np.testing.assert_allclose(x, [np.e], rtol=1e-6)


def test_implicit_midpoint_matches_closed_form_linear():
    # For dx/dt = a x the midpoint step has the closed form
    # x1 = x0 (1 + a dt/2) / (1 - a dt/2).
    a, dt, x0 = -2.0, 0.1, 1.5
    x1 = implicit_midpoint(_linear, 0.0, x0, (a,), dt)
    np.testing.assert_allclose(x1, x0 * (1 + a * dt / 2) / (1 - a * dt / 2), rtol=1e-8)


def test_implicit_midpoint_conserves_oscillator_energy():
    # The implicit midpoint rule conserves quadratic invariants; for the
    # harmonic oscillator (q, p) -> (p, -q) the energy q^2 + p^2 is preserved.
    def osc(t, x):
        return np.array([x[1], -x[0]])

    x = np.array([1.0, 0.0])
    for k in range(50):
        x = implicit_midpoint(osc, 0.0, x, (), 0.1)
    np.testing.assert_allclose(x @ x, 1.0, rtol=1e-6)
