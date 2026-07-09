"""Deterministic shape/regression tests for the Lorenz model right-hand sides.

Test-only (issue #10): no implementation code is touched. Expected values are
hard-coded literals computed by hand from the published equations, so these
act as regression guards for the RHS implementations.
"""

import numpy as np

from da.l63 import atr_radious_bound, lorenz63, max_lyapunov_exponent_l63
from da.l96 import lorenz96, two_thirds_observation


def test_lorenz63_shape_and_dtype():
    dx = lorenz63(0.0, np.array([1.0, 2.0, 3.0]))
    assert dx.shape == (3,)
    assert np.issubdtype(dx.dtype, np.floating)


def test_lorenz63_regression_default_params():
    # s=10, r=28, b=8/3 at x=(1, 2, 3):
    #   dx0 = 10*(2-1) = 10
    #   dx1 = 1*(28-3) - 2 = 23
    #   dx2 = 1*2 - (8/3)*3 = -6
    dx = lorenz63(0.0, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(dx, [10.0, 23.0, -6.0], rtol=1e-12)


def test_lorenz63_origin_is_fixed_point():
    np.testing.assert_allclose(lorenz63(0.0, np.zeros(3)), np.zeros(3))


def test_lorenz63_z_axis_decays():
    # On the z-axis (0, 0, z) the flow is (0, 0, -b z).
    dx = lorenz63(0.0, np.array([0.0, 0.0, 4.0]), s=10, r=28, b=2.0)
    np.testing.assert_allclose(dx, [0.0, 0.0, -8.0])


def test_lorenz63_is_autonomous():
    x = np.array([1.5, -2.0, 20.0])
    np.testing.assert_allclose(lorenz63(0.0, x), lorenz63(123.4, x))


def test_atr_radious_bound_default():
    # rho = b (r + s) / (2 sqrt(b - 1)) with s=10, b=8/3, r=28.
    expected = (8 / 3) * 38.0 / (2 * np.sqrt(8 / 3 - 1))
    assert np.isclose(atr_radious_bound(), expected)
    assert np.isclose(max_lyapunov_exponent_l63(), expected - 1.0)


def test_lorenz96_shape_preserved():
    for J in (5, 40):
        x = np.linspace(-1.0, 1.0, J)
        dx = lorenz96(0.0, x, 8.0)
        assert dx.shape == (J,)


def test_lorenz96_regression_small_system():
    # J=5, F=0, x=(1,2,3,4,5):
    #   dx_j = (x_{j+1} - x_{j-2}) x_{j-1} - x_j  (cyclic)
    #        = (-2,-2,3,3,-2)*(5,1,2,3,4) - (1,2,3,4,5)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        lorenz96(0.0, x, 0.0), [-11.0, -4.0, 3.0, 5.0, -13.0], rtol=1e-12
    )


def test_lorenz96_constant_state_is_fixed_point():
    # x = F * ones is an equilibrium: advection term vanishes and -x + F = 0.
    F = 8.0
    x = F * np.ones(40)
    np.testing.assert_allclose(lorenz96(0.0, x, F), np.zeros(40), atol=1e-12)


def test_lorenz96_accepts_vector_forcing():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    F = np.full(5, 8.0)
    np.testing.assert_allclose(lorenz96(0.0, x, F), lorenz96(0.0, x, 8.0))


def test_two_thirds_observation_operator():
    H, observed = two_thirds_observation(60)
    assert H.shape == (40, 60)
    # every third component (index 2, 5, 8, ...) is unobserved
    assert observed.tolist() == [i for i in range(60) if i % 3 != 2]
    # exactly one 1 per observed row, selecting the right column
    assert np.array_equal(H.sum(axis=1), np.ones(40))
    assert np.array_equal(H @ np.arange(60), observed.astype(float))


def test_two_thirds_observation_requires_multiple_of_three():
    import pytest

    with pytest.raises(ValueError, match="divisible by 3"):
        two_thirds_observation(61)
