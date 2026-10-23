import numpy as np
import pytest

from da.nse2d import NSE2DConfig, NSE2DTorus


@pytest.mark.parametrize("n,k1,k2,alias", [(12, 4, 4, -4), (16, 7, 6, -3)])
def test_boundary_and_high_high_alias_are_removed(n, k1, k2, alias):
    model = NSE2DTorus(NSE2DConfig(nx=n, ny=n, viscosity=0, dealias=True))
    x, y = model.grid()
    omega = np.cos(k1*x) + np.cos(k2*x+y)
    tendency = np.fft.fft2(model.rhs(omega)) / n**2
    assert abs(tendency[1, alias % n]) < 1e-14
    np.testing.assert_allclose(tendency, 0, atol=1e-14)


@pytest.mark.parametrize("nx,ny", [(9, 9), (12, 12), (15, 16), (16, 17)])
def test_retained_quadratic_interactions_match_padding(nx, ny):
    model = NSE2DTorus(NSE2DConfig(nx=nx, ny=ny, viscosity=0, dealias=True))
    padded = NSE2DTorus(NSE2DConfig(nx=nx, ny=ny, viscosity=0, dealias="pad"))
    # Supply only retained modes to both methods. Padding also retains modes
    # outside the 2/3 band, so compare only common output modes.
    raw = np.random.default_rng(12).normal(size=(ny, nx))
    truncated = model.ifft(model.fft(raw) * model._dealias_mask)
    actual = model.fft(model.rhs(truncated))
    expected = padded.fft(padded.rhs(truncated)) * model._dealias_mask
    assert np.max(np.abs(expected)) > 0.01
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=1e-12)
    if nx % 3 == 0:
        assert not model._dealias_mask[:, nx//3].any()
    if ny % 3 == 0:
        assert not model._dealias_mask[ny//3, :].any()


def test_truncation_does_not_remove_linear_damping_of_high_modes():
    model = NSE2DTorus(NSE2DConfig(nx=12, ny=12, viscosity=0.1, drag=0.2))
    x, _ = model.grid()
    omega = np.cos(4*x)
    np.testing.assert_allclose(model.rhs(omega), -(0.1*16+0.2)*omega, atol=1e-13)
