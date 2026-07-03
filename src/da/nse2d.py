"""Minimal vorticity-form 2D Navier-Stokes model for da_py experiments.

Usage:
    cfg = NSE2DConfig(nx=64, ny=64, viscosity=1e-3)
    model = NSE2DTorus(cfg)
    omega1 = model.step(omega0, dt=0.01)
    traj = model.solve(omega0, dt=0.01, n_steps=100)

    # internal_steps subdivides the DA forecast interval dt.
    M = model.as_forecast(internal_steps=5)  # M(x_flat, dt) -> x_flat
    obs = model.grid_observation(stride=4)

    # dealias=True uses 2/3 truncation; dealias="pad" uses 3/2 padding.
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
    dealias: bool | str = True
    forcing: ArrayLike | Callable[[ArrayLike], ArrayLike] | None = None
    drag: float = 0.0

    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive")
        if self.viscosity < 0:
            raise ValueError("viscosity must be non-negative")
        if self.drag < 0:
            raise ValueError("drag must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.dealias not in {False, True, "2/3", "pad"}:
            raise ValueError("dealias must be False, True, '2/3', or 'pad'")


def inubushi_caulfield_config(
    *,
    nx=64,
    ny=64,
    viscosity=1.0e-3,
    drag=1.0e-1,
    forcing_mode=4,
    forcing_amplitude=1.0,
    length=2 * np.pi,
    dealias="pad",
):
    """Return the forced-damped Kolmogorov-flow reference configuration.

    This uses the vorticity equation
    ``omega_t + u dot grad omega = nu Laplacian omega - drag*omega + F_omega``
    with ``F_omega = -forcing_amplitude*k_f*cos(k_f*y)`` on ``[0, length]^2``.
    The default ``dealias="pad"`` uses 3/2 padding for the nonlinear term.
    """
    cfg = NSE2DConfig(
        nx=nx,
        ny=ny,
        viscosity=viscosity,
        drag=drag,
        length=length,
        dealias=dealias,
    )
    model = NSE2DTorus(cfg)
    return NSE2DConfig(
        nx=nx,
        ny=ny,
        viscosity=viscosity,
        drag=drag,
        length=length,
        dealias=dealias,
        forcing=model.kolmogorov_vorticity_forcing(
            mode=forcing_mode,
            amplitude=forcing_amplitude,
        ),
    )


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
        self.spectral_shape = (self.ny, self.nx // 2 + 1)
        self._spectral_indices, self._spectral_real_only = _independent_rfft_indices(
            self.nx,
            self.ny,
        )
        self.spectral_state_dim = _real_vector_dim(self._spectral_real_only)
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
        self._pad_shape = (3 * self.ny // 2, 3 * self.nx // 2)
        self._pad_scale = np.prod(self._pad_shape) / self.state_dim
        dx_pad = config.length / self._pad_shape[1]
        dy_pad = config.length / self._pad_shape[0]
        kx_pad = 2 * np.pi * np.fft.fftfreq(self._pad_shape[1], d=dx_pad)
        ky_pad = 2 * np.pi * np.fft.fftfreq(self._pad_shape[0], d=dy_pad)
        self._kx_pad, self._ky_pad = np.meshgrid(kx_pad, ky_pad)
        self._pad_y_indices = _mode_indices(self.ny, self._pad_shape[0])
        self._pad_x_indices = _mode_indices(self.nx, self._pad_shape[1])
        self._pad_resolved_mask = self._make_pad_resolved_mask()
        self._low_mode_masks = {}

    def _make_dealias_mask(self):
        if self.config.dealias in {False, "pad"}:
            return np.ones(self.shape, dtype=bool)
        mx = np.fft.fftfreq(self.nx) * self.nx
        my = np.fft.fftfreq(self.ny) * self.ny
        ix, iy = np.meshgrid(mx, my)
        return (np.abs(ix) <= self.nx / 3) & (np.abs(iy) <= self.ny / 3)

    def _make_pad_resolved_mask(self):
        mask = np.ones(self.shape, dtype=bool)
        if self.nx % 2 == 0:
            mask[:, self.nx // 2] = False
        if self.ny % 2 == 0:
            mask[self.ny // 2, :] = False
        return mask

    def _pad_fft(self, field_hat):
        padded = np.zeros(self._pad_shape, dtype=complex)
        padded[np.ix_(self._pad_y_indices, self._pad_x_indices)] = np.where(
            self._pad_resolved_mask,
            field_hat,
            0.0,
        )
        return padded * self._pad_scale

    def _truncate_fft(self, field_hat_pad):
        truncated = field_hat_pad[np.ix_(self._pad_y_indices, self._pad_x_indices)]
        return np.where(self._pad_resolved_mask, truncated, 0.0) / self._pad_scale

    def _as_state(self, omega):
        arr = np.asarray(omega)
        if arr.shape != self.shape:
            raise ValueError(f"expected state shape {self.shape}, got {arr.shape}")
        return arr

    def grid(self):
        x = np.linspace(0.0, self.config.length, self.nx, endpoint=False)
        y = np.linspace(0.0, self.config.length, self.ny, endpoint=False)
        return np.meshgrid(x, y)

    def _kolmogorov_wavenumber(self, mode):
        if mode <= 0:
            raise ValueError("mode must be positive")
        return 2 * np.pi * mode / self.config.length

    def kolmogorov_velocity(self, mode=4, amplitude=1.0):
        """Return the Kolmogorov velocity ``u=A sin(k y), v=0``."""
        _, y = self.grid()
        wavenumber = self._kolmogorov_wavenumber(mode)
        u = amplitude * np.sin(wavenumber * y)
        v = np.zeros_like(u)
        return u, v

    def kolmogorov_vorticity(self, mode=4, amplitude=1.0):
        """Return vorticity ``omega=-A k cos(k y)`` of Kolmogorov flow."""
        _, y = self.grid()
        wavenumber = self._kolmogorov_wavenumber(mode)
        return -amplitude * wavenumber * np.cos(wavenumber * y)

    def kolmogorov_forcing(self, mode=4, amplitude=1.0):
        """Return vorticity forcing making Kolmogorov flow steady.

        For ``u=A sin(k y), v=0`` and ``omega=-A k cos(k y)``, the advection
        term vanishes and the steady vorticity equation requires
        ``forcing = -nu * Laplacian(omega) + drag*omega``.
        """
        _, y = self.grid()
        wavenumber = self._kolmogorov_wavenumber(mode)
        return (
            -(
                self.config.viscosity * amplitude * wavenumber**3
                + self.config.drag * amplitude * wavenumber
            )
            * np.cos(wavenumber * y)
        )

    def kolmogorov_vorticity_forcing(self, mode=4, amplitude=1.0):
        """Return curl of ``amplitude*sin(k y) e_x``.

        For the velocity forcing ``f = amplitude*sin(k y) e_x``, the vorticity
        forcing is ``F_omega = -amplitude*k*cos(k y)``.
        """
        _, y = self.grid()
        wavenumber = self._kolmogorov_wavenumber(mode)
        return -amplitude * wavenumber * np.cos(wavenumber * y)

    def fft(self, field):
        return np.fft.fft2(self._as_state(field))

    def ifft(self, field_hat):
        return np.fft.ifft2(field_hat).real

    def rfft(self, field):
        return np.fft.rfft2(self._as_state(field))

    def irfft(self, field_hat):
        return np.fft.irfft2(field_hat, s=self.shape)

    def pack_spectral_state(self, omega_hat):
        """Pack independent rfft2 vorticity coefficients into a real vector."""
        coeffs = np.asarray(omega_hat).reshape(self.spectral_shape)
        values = []
        for index, real_only in zip(
            self._spectral_indices,
            self._spectral_real_only,
            strict=False,
        ):
            coeff = coeffs[index]
            values.append(coeff.real)
            if not real_only:
                values.append(coeff.imag)
        return np.asarray(values, dtype=float)

    def unpack_spectral_state(self, x_hat):
        """Unpack a real vector into an rfft2 array satisfying Hermitian symmetry."""
        values = np.asarray(x_hat, dtype=float)
        if values.shape != (self.spectral_state_dim,):
            raise ValueError(
                f"expected spectral state shape {(self.spectral_state_dim,)}, "
                f"got {values.shape}"
            )
        coeffs = np.zeros(self.spectral_shape, dtype=complex)
        cursor = 0
        for (iy, ix), real_only in zip(
            self._spectral_indices,
            self._spectral_real_only,
            strict=False,
        ):
            if real_only:
                coeff = values[cursor] + 0.0j
                cursor += 1
            else:
                coeff = values[cursor] + 1j * values[cursor + 1]
                cursor += 2
            coeffs[iy, ix] = coeff
            self_conjugate_x = ix == 0 or (self.nx % 2 == 0 and ix == self.nx // 2)
            self_conjugate_y = iy == 0 or (self.ny % 2 == 0 and iy == self.ny // 2)
            if self_conjugate_x and not self_conjugate_y:
                coeffs[-iy % self.ny, ix] = np.conj(coeff)
        return coeffs

    def to_spectral_state(self, omega):
        """Convert real-space vorticity to the packed spectral state vector."""
        return self.pack_spectral_state(self.rfft(omega))

    def from_spectral_state(self, x_hat):
        """Convert a packed spectral state vector to real-space vorticity."""
        return self.irfft(self.unpack_spectral_state(x_hat))

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

    def _advective_term_hat_padded(self, omega_hat):
        omega_hat_pad = self._pad_fft(omega_hat)
        k2_pad = self._kx_pad**2 + self._ky_pad**2
        inv_k2_pad = np.zeros_like(k2_pad)
        nonzero = k2_pad > 0
        inv_k2_pad[nonzero] = 1.0 / k2_pad[nonzero]
        psi_hat_pad = omega_hat_pad * inv_k2_pad
        psi_hat_pad[0, 0] = 0.0

        u = np.fft.ifft2(1j * self._ky_pad * psi_hat_pad).real
        v = np.fft.ifft2(-1j * self._kx_pad * psi_hat_pad).real
        omega_x = np.fft.ifft2(1j * self._kx_pad * omega_hat_pad).real
        omega_y = np.fft.ifft2(1j * self._ky_pad * omega_hat_pad).real
        return self._truncate_fft(np.fft.fft2(u * omega_x + v * omega_y))

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
        if self.config.dealias == "pad":
            adv_hat = self._advective_term_hat_padded(omega_hat)
        else:
            u, v = self.velocity(omega)
            omega_x = self.ifft(1j * self.kx * omega_hat)
            omega_y = self.ifft(1j * self.ky * omega_hat)
            adv_hat = np.fft.fft2(u * omega_x + v * omega_y)
            adv_hat = np.where(self._dealias_mask, adv_hat, 0.0)
        diffusion = self.ifft(-self.config.viscosity * self.k2 * omega_hat)
        drag = -self.config.drag * omega
        return -self.ifft(adv_hat) + diffusion + drag + self._forcing(omega)

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

    def step_spectral_state(self, x_hat, dt):
        """Advance one packed spectral vorticity state by one time step."""
        omega_hat = self.unpack_spectral_state(x_hat)
        return self.pack_spectral_state(self.step_spectral(omega_hat, dt))

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

    def radial_spectrum(self, omega, *, quantity="enstrophy"):
        """Return shell-summed kinetic-energy or enstrophy spectrum.

        Shells are indexed by integer Fourier mode radius.  The spectrum sums
        to ``energy(omega)`` or ``enstrophy(omega)`` up to roundoff.
        """
        if quantity not in {"energy", "enstrophy"}:
            raise ValueError("quantity must be 'energy' or 'enstrophy'")
        coeff = self.fft(omega) / self.state_dim
        if quantity == "enstrophy":
            density = 0.5 * np.abs(coeff) ** 2
        else:
            density = np.zeros(self.shape)
            nonzero = self.k2 > 0
            density[nonzero] = 0.5 * np.abs(coeff[nonzero]) ** 2 / self.k2[nonzero]

        kx_index = np.fft.fftfreq(self.nx) * self.nx
        ky_index = np.fft.fftfreq(self.ny) * self.ny
        ix, iy = np.meshgrid(kx_index, ky_index)
        shell_index = np.floor(np.sqrt(ix**2 + iy**2)).astype(int)
        shells = np.arange(shell_index.max() + 1)
        spectrum = np.bincount(
            shell_index.reshape(-1),
            weights=density.reshape(-1),
            minlength=len(shells),
        )
        return shells, spectrum

    def spectral_tail_fraction(self, omega, cutoff, *, quantity="enstrophy"):
        """Fraction of spectral diagnostic contained in shells ``k >= cutoff``."""
        shells, spectrum = self.radial_spectrum(omega, quantity=quantity)
        total = float(np.sum(spectrum))
        if total == 0.0:
            return 0.0
        return float(np.sum(spectrum[shells >= cutoff]) / total)

    def palinstrophy(self, omega):
        omega = self._as_state(omega)
        omega_hat = self.fft(omega)
        omega_x = self.ifft(1j * self.kx * omega_hat)
        omega_y = self.ifft(1j * self.ky * omega_hat)
        return float(0.5 * np.mean(omega_x**2 + omega_y**2))

    def low_mode_mask(self, kmax):
        """Boolean full-spectrum mask keeping integer modes |mx|, |my| <= kmax.

        Uses the same square (max-norm) cutoff on integer mode indices as
        :class:`LowModeObservation`, so the projection and the low-mode
        observation operators select the same Fourier coefficients.
        """
        cached = self._low_mode_masks.get(kmax)
        if cached is not None:
            return cached
        mx = np.fft.fftfreq(self.nx) * self.nx
        my = np.fft.fftfreq(self.ny) * self.ny
        MX, MY = np.meshgrid(mx, my)
        mask = (np.abs(MX) <= kmax) & (np.abs(MY) <= kmax)
        self._low_mode_masks[kmax] = mask
        return mask

    def project_low_modes(self, omega, kmax):
        """Low-pass projection ``P_kmax omega`` in Fourier space.

        Keeps the Fourier coefficients with integer mode indices
        ``max(|mx|, |my|) <= kmax`` (including the mean) and zeroes the rest;
        the result is returned in real space. ``P`` is idempotent and
        ``project_low_modes + project_high_modes`` is the identity.
        """
        omega = self._as_state(omega)
        return self.ifft(self.fft(omega) * self.low_mode_mask(kmax))

    def project_high_modes(self, omega, kmax):
        """High-pass complement ``Q_kmax = I - P_kmax`` of the low-pass projection."""
        omega = self._as_state(omega)
        return omega - self.project_low_modes(omega, kmax)

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

    def as_spectral_forecast(self, internal_steps=1):
        """Return ``M(x_hat, dt) -> x_hat`` for packed spectral states."""
        if internal_steps <= 0:
            raise ValueError("internal_steps must be positive")

        def forecast(x_hat, dt):
            arr = np.asarray(x_hat)
            state = arr.astype(np.result_type(arr.dtype, np.float64), copy=False)
            for _ in range(internal_steps):
                state = self.step_spectral_state(state, dt / internal_steps)
            return state.astype(arr.dtype, copy=False)

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

    def spectral_low_mode_observation(self, kmax, *, linear=False):
        obs = SpectralLowModeObservation(self, kmax=kmax)
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


def _real_vector_dim(real_only):
    return sum(1 if is_real else 2 for is_real in real_only)


def _mode_indices(n, n_pad):
    modes = np.rint(np.fft.fftfreq(n) * n).astype(int)
    pad_modes = np.rint(np.fft.fftfreq(n_pad) * n_pad).astype(int)
    pad_positions = {mode: index for index, mode in enumerate(pad_modes)}
    return np.asarray([pad_positions[mode] for mode in modes])


def _packed_positions(indices, real_only):
    positions = {}
    start = 0
    for index, is_real in zip(indices, real_only, strict=False):
        positions[index] = start
        start += 1 if is_real else 2
    return positions


def _independent_rfft_indices(nx, ny, kmax=None):
    indices = []
    real_only = []
    ky_index = np.fft.fftfreq(ny) * ny
    kx_index = np.fft.rfftfreq(nx) * nx
    for iy, ky in enumerate(ky_index):
        for ix, kx in enumerate(kx_index):
            if kmax is not None and (abs(kx) > kmax or abs(ky) > kmax):
                continue
            y_nyquist = ny % 2 == 0 and iy == ny // 2
            if ix == 0 and ky < 0 and not y_nyquist:
                continue
            if nx % 2 == 0 and ix == nx // 2 and ky < 0 and not y_nyquist:
                continue
            self_conjugate_x = ix == 0 or (nx % 2 == 0 and ix == nx // 2)
            self_conjugate_y = iy == 0 or (ny % 2 == 0 and iy == ny // 2)
            indices.append((iy, ix))
            real_only.append(self_conjugate_x and self_conjugate_y)
    return indices, real_only


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
        self.indices, self.real_only = _independent_rfft_indices(
            self.model.nx,
            self.model.ny,
            self.kmax,
        )
        self.state_dim = self.model.state_dim
        self.spectral_shape = self.model.spectral_shape
        self.obs_dim = _real_vector_dim(self.real_only)

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
class SpectralLowModeObservation:
    """Observe low modes from a packed spectral vorticity state.

    The state and observation are both real vectors of independent rfft2
    coefficients.  The coefficients are unnormalized Fourier coefficients,
    matching ``numpy.fft.rfft2`` and ``NSE2DTorus.pack_spectral_state``.
    """

    model: NSE2DTorus
    kmax: int

    def __post_init__(self):
        if self.kmax < 0:
            raise ValueError("kmax must be non-negative")
        self.indices, self.real_only = _independent_rfft_indices(
            self.model.nx,
            self.model.ny,
            self.kmax,
        )
        self.state_dim = self.model.spectral_state_dim
        self.obs_dim = _real_vector_dim(self.real_only)
        self._selection = self._make_selection()

    def _make_selection(self):
        positions = _packed_positions(
            self.model._spectral_indices,
            self.model._spectral_real_only,
        )
        selection = []
        for index, real_only in zip(self.indices, self.real_only, strict=False):
            start = positions[index]
            selection.append(start)
            if not real_only:
                selection.append(start + 1)
        return np.asarray(selection, dtype=int)

    def observe(self, x_hat):
        values = np.asarray(x_hat, dtype=float)
        if values.shape != (self.state_dim,):
            raise ValueError(f"expected state shape {(self.state_dim,)}, got {values.shape}")
        return values[self._selection].copy()

    def apply_flat(self, x_hat):
        return self.observe(x_hat)

    def as_matrix(self):
        matrix = np.zeros((self.obs_dim, self.state_dim))
        matrix[np.arange(self.obs_dim), self._selection] = 1.0
        return matrix

    def to_spectral_state(self, y_obs):
        values = np.asarray(y_obs, dtype=float)
        if values.shape != (self.obs_dim,):
            raise ValueError(f"expected observation shape {(self.obs_dim,)}, got {values.shape}")
        x_hat = np.zeros(self.state_dim)
        x_hat[self._selection] = values
        return x_hat

    def observation_variances(self, sigma0, decay_power=1.0):
        """Diagonal R entries with sigma(k)=sigma0/(1+|k|^2)^(decay_power/2)."""
        variances = []
        ky_index = np.fft.fftfreq(self.model.ny) * self.model.ny
        kx_index = np.fft.rfftfreq(self.model.nx) * self.model.nx
        for (iy, ix), real_only in zip(self.indices, self.real_only, strict=False):
            k2 = kx_index[ix] ** 2 + ky_index[iy] ** 2
            sigma = sigma0 / (1.0 + k2) ** (0.5 * decay_power)
            variances.append(sigma**2)
            if not real_only:
                variances.append(sigma**2)
        return np.asarray(variances)


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
