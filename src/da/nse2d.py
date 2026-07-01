"""Minimal vorticity-form 2D Navier-Stokes model for da_py experiments.

Usage:
    cfg = NSE2DConfig(nx=64, ny=64, viscosity=1e-3)
    model = NSE2DTorus(cfg)
    omega1 = model.step(omega0, dt=0.01)
    traj = model.solve(omega0, dt=0.01, n_steps=100)

    # internal_steps subdivides the DA forecast interval dt.
    M = model.as_forecast(internal_steps=5)  # M(x_flat, dt) -> x_flat
    obs = model.grid_observation(stride=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class NSE2DConfig:
    nx: int = 64
    ny: int = 64
    viscosity: float = 1.0e-3
    length: float = 2 * np.pi
    dealias: bool = True
    forcing: ArrayLike | Callable[[ArrayLike], ArrayLike] | None = None

    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.viscosity < 0:
            raise ValueError("viscosity must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")


class NSE2DTorus:
    """Pseudo-spectral 2D Navier-Stokes solver on a square periodic torus.

    Public states are real-space vorticity arrays with shape ``(ny, nx)``.
    The inverse Laplacian fixes the spatial mean of the streamfunction to zero;
    consequently the zero Fourier mode of vorticity does not contribute to the
    reconstructed velocity.
    """

    def __init__(self, config: NSE2DConfig):
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.shape = (self.ny, self.nx)
        self.state_dim = self.nx * self.ny
        self.dx = config.length / self.nx
        self.dy = config.length / self.ny

        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self.kx, self.ky = np.meshgrid(kx, ky)
        self.k2 = self.kx**2 + self.ky**2
        self._inv_k2 = np.zeros_like(self.k2)
        nonzero = self.k2 > 0
        self._inv_k2[nonzero] = 1.0 / self.k2[nonzero]
        self._dealias_mask = self._make_dealias_mask()

    def _make_dealias_mask(self):
        if not self.config.dealias:
            return np.ones(self.shape, dtype=bool)
        mx = np.fft.fftfreq(self.nx) * self.nx
        my = np.fft.fftfreq(self.ny) * self.ny
        ix, iy = np.meshgrid(mx, my)
        return (np.abs(ix) <= self.nx / 3) & (np.abs(iy) <= self.ny / 3)

    def _as_state(self, omega):
        arr = np.asarray(omega)
        if arr.shape != self.shape:
            raise ValueError(f"expected state shape {self.shape}, got {arr.shape}")
        return arr

    def fft(self, field):
        return np.fft.fft2(self._as_state(field))

    def ifft(self, field_hat):
        return np.fft.ifft2(field_hat).real

    def rfft(self, field):
        return np.fft.rfft2(self._as_state(field))

    def irfft(self, field_hat):
        return np.fft.irfft2(field_hat, s=self.shape)

    def derivative(self, field, axis):
        field_hat = self.fft(field)
        if axis in (0, "y"):
            return self.ifft(1j * self.ky * field_hat)
        if axis in (1, "x"):
            return self.ifft(1j * self.kx * field_hat)
        raise ValueError("axis must be 0, 1, 'y', or 'x'")

    def _streamfunction_hat(self, omega):
        omega_hat = self.fft(omega)
        psi_hat = omega_hat * self._inv_k2
        psi_hat[0, 0] = 0.0
        return psi_hat

    def streamfunction(self, omega):
        psi_hat = self._streamfunction_hat(omega)
        return self.ifft(psi_hat)

    def velocity(self, omega):
        psi_hat = self._streamfunction_hat(omega)
        u = self.ifft(1j * self.ky * psi_hat)
        v = self.ifft(-1j * self.kx * psi_hat)
        return u, v

    def _forcing(self, omega):
        forcing = self.config.forcing
        if forcing is None:
            return 0.0
        if callable(forcing):
            value = forcing(omega)
        else:
            value = forcing
        value = np.asarray(value, dtype=np.asarray(omega).dtype)
        if value.shape != self.shape:
            raise ValueError(f"expected forcing shape {self.shape}, got {value.shape}")
        return value

    def rhs(self, omega):
        omega = self._as_state(omega)
        omega_hat = self.fft(omega)
        u, v = self.velocity(omega)
        omega_x = self.ifft(1j * self.kx * omega_hat)
        omega_y = self.ifft(1j * self.ky * omega_hat)
        adv_hat = np.fft.fft2(u * omega_x + v * omega_y)
        adv_hat = np.where(self._dealias_mask, adv_hat, 0.0)
        diffusion = self.ifft(-self.config.viscosity * self.k2 * omega_hat)
        return -self.ifft(adv_hat) + diffusion + self._forcing(omega)

    def step(self, omega, dt):
        omega = self._as_state(omega)
        dtype = omega.dtype
        y = omega.astype(np.result_type(dtype, np.float64), copy=False)
        k1 = self.rhs(y)
        k2 = self.rhs(y + 0.5 * dt * k1)
        k3 = self.rhs(y + 0.5 * dt * k2)
        k4 = self.rhs(y + dt * k3)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if np.issubdtype(dtype, np.floating):
            return y_next.astype(dtype, copy=False)
        return y_next

    def step_spectral(self, omega_hat, dt):
        """Advance one rfft2 vorticity state by one time step."""
        omega = self.irfft(omega_hat)
        return self.rfft(self.step(omega, dt))

    def solve_spectral(self, omega_hat0, dt, n_steps, *, store_every=1):
        """Integrate a trajectory whose public state is rfft2 vorticity."""
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if store_every <= 0:
            raise ValueError("store_every must be positive")
        omega_hat = np.asarray(omega_hat0).copy()
        out = [omega_hat.copy()]
        for step in range(1, n_steps + 1):
            omega_hat = self.step_spectral(omega_hat, dt)
            if step % store_every == 0:
                out.append(omega_hat.copy())
        return np.stack(out, axis=0)

    def solve(self, omega0, dt, n_steps, *, store_every=1):
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if store_every <= 0:
            raise ValueError("store_every must be positive")
        omega = self._as_state(omega0).copy()
        out = [omega.copy()]
        for step in range(1, n_steps + 1):
            omega = self.step(omega, dt)
            if step % store_every == 0:
                out.append(omega.copy())
        return np.stack(out, axis=0)

    def energy(self, omega):
        u, v = self.velocity(omega)
        return float(0.5 * np.mean(u**2 + v**2))

    def enstrophy(self, omega):
        omega = self._as_state(omega)
        return float(0.5 * np.mean(omega**2))

    def _forecast_flat(self, x, dt, n_steps):
        arr = np.asarray(x)
        omega = arr.reshape(self.shape)
        for _ in range(n_steps):
            omega = self.step(omega, dt)
        return omega.reshape(-1).astype(arr.dtype, copy=False)

    def as_forecast(self, internal_steps=1):
        """Return a da_py model callback ``M(x_flat, dt) -> x_flat``.

        ``internal_steps`` is the number of RK4 substeps used inside one
        assimilation forecast interval ``dt``.
        """
        if internal_steps <= 0:
            raise ValueError("internal_steps must be positive")

        def forecast(x, dt):
            return self._forecast_flat(x, dt / internal_steps, internal_steps)

        return forecast

    def forecast_fn(self, dt, n_steps=1):
        return lambda x: self._forecast_flat(x, dt, n_steps)

    def forecast_batch_fn(self, dt, n_steps=1):
        forecast_one = self.forecast_fn(dt, n_steps)

        def forecast(x):
            arr = np.asarray(x)
            if arr.ndim == 1:
                return forecast_one(arr)
            return np.stack([forecast_one(member) for member in arr])

        return forecast

    def low_mode_observation(self, kmax, component="vorticity", *, linear=False):
        obs = LowModeObservation(self, kmax=kmax, component=component)
        return obs.as_matrix() if linear else obs

    def independent_low_mode_observation(self, kmax, *, linear=False):
        obs = IndependentLowModeObservation(self, kmax=kmax)
        return obs.as_matrix() if linear else obs

    def grid_observation(self, stride=1, *, linear=False):
        obs = GridObservation(self, stride=stride)
        return obs.as_matrix() if linear else obs


def _dense_observation_matrix(obs):
    matrix = np.empty((obs.obs_dim, obs.state_dim))
    for i in range(obs.state_dim):
        basis = np.zeros(obs.state_dim)
        basis[i] = 1.0
        matrix[:, i] = obs.apply_flat(basis)
    return matrix


@dataclass
class LowModeObservation:
    """Observe real and imaginary parts of selected low Fourier coefficients."""

    model: NSE2DTorus
    kmax: int
    component: str = "vorticity"

    def __post_init__(self):
        if self.kmax < 0:
            raise ValueError("kmax must be non-negative")
        if self.component not in {"vorticity", "streamfunction"}:
            raise ValueError("component must be 'vorticity' or 'streamfunction'")
        kx_index = np.fft.fftfreq(self.model.nx) * self.model.nx
        ky_index = np.fft.fftfreq(self.model.ny) * self.model.ny
        ix, iy = np.meshgrid(kx_index, ky_index)
        self.mask = (np.abs(ix) <= self.kmax) & (np.abs(iy) <= self.kmax)
        self.state_dim = self.model.state_dim
        self.obs_dim = int(2 * np.count_nonzero(self.mask))

    def observe(self, state):
        omega = np.asarray(state).reshape(self.model.shape)
        field = (
            self.model.streamfunction(omega)
            if self.component == "streamfunction"
            else omega
        )
        coeffs = np.fft.fft2(field) / self.model.state_dim
        selected = coeffs[self.mask]
        return np.concatenate([selected.real, selected.imag])

    def apply_flat(self, x_flat):
        return self.observe(np.asarray(x_flat).reshape(self.model.shape))

    def matrix_free_apply(self, x_flat):
        return self.apply_flat(x_flat)

    def as_matrix(self):
        return _dense_observation_matrix(self)


@dataclass
class IndependentLowModeObservation:
    """Observe non-redundant low rfft2 coefficients of a real vorticity field."""

    model: NSE2DTorus
    kmax: int

    def __post_init__(self):
        if self.kmax < 0:
            raise ValueError("kmax must be non-negative")
        self.indices = []
        self.real_only = []
        ky_index = np.fft.fftfreq(self.model.ny) * self.model.ny
        kx_index = np.fft.rfftfreq(self.model.nx) * self.model.nx
        for iy, ky in enumerate(ky_index):
            for ix, kx in enumerate(kx_index):
                if abs(kx) > self.kmax or abs(ky) > self.kmax:
                    continue
                if ix == 0 and ky < 0:
                    continue
                if self.model.nx % 2 == 0 and ix == self.model.nx // 2 and ky < 0:
                    continue
                self_conjugate_x = ix == 0 or (
                    self.model.nx % 2 == 0 and ix == self.model.nx // 2
                )
                self_conjugate_y = iy == 0 or (
                    self.model.ny % 2 == 0 and iy == self.model.ny // 2
                )
                is_real = self_conjugate_x and self_conjugate_y
                self.indices.append((iy, ix))
                self.real_only.append(is_real)
        self.state_dim = self.model.state_dim
        self.spectral_shape = (self.model.ny, self.model.nx // 2 + 1)
        self.obs_dim = sum(1 if real else 2 for real in self.real_only)

    def observe_spectral(self, omega_hat):
        coeffs = np.asarray(omega_hat).reshape(self.spectral_shape)
        values = []
        for index, real_only in zip(self.indices, self.real_only, strict=False):
            coeff = coeffs[index] / self.model.state_dim
            values.append(coeff.real)
            if not real_only:
                values.append(coeff.imag)
        return np.asarray(values)

    def observe(self, omega):
        return self.observe_spectral(self.model.rfft(omega))

    def apply_flat(self, x_flat):
        return self.observe(np.asarray(x_flat).reshape(self.model.shape))

    def apply_spectral_flat(self, x_hat_flat):
        return self.observe_spectral(np.asarray(x_hat_flat).reshape(self.spectral_shape))

    def as_matrix(self):
        return _dense_observation_matrix(self)


@dataclass
class GridObservation:
    """Observe vorticity values on a regular subsampled physical grid."""

    model: NSE2DTorus
    stride: int = 1

    def __post_init__(self):
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        ny_obs = len(range(0, self.model.ny, self.stride))
        nx_obs = len(range(0, self.model.nx, self.stride))
        self.state_dim = self.model.state_dim
        self.obs_dim = ny_obs * nx_obs

    def observe(self, state):
        omega = np.asarray(state).reshape(self.model.shape)
        return omega[:: self.stride, :: self.stride].reshape(-1).copy()

    def apply_flat(self, x_flat):
        return self.observe(np.asarray(x_flat).reshape(self.model.shape))

    def as_matrix(self):
        matrix = np.zeros((self.obs_dim, self.state_dim))
        rows = np.arange(self.obs_dim)
        cols = np.arange(self.state_dim).reshape(self.model.shape)[
            :: self.stride, :: self.stride
        ].reshape(-1)
        matrix[rows, cols] = 1.0
        return matrix
