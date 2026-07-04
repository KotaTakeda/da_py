"""Reference NSE configurations for the Kelly-style 2D NSE benchmarks (#27).

This module is documentation/scaffolding for future ETKF large-ensemble
experiments (issue #28), **not a new benchmark result**: it records, in one
importable place, the solver and assimilation settings used by
``examples/nse2d_partial_obs_enkf.py`` (issues #13/#25) so that follow-up
experiments start from the exact validated numbers instead of re-deriving
them from CLI defaults.

- ``kelly_32``   -- the validated Kelly-like preset (PO EnKF with additive
  inflation reproduces Kelly, Law & Stuart Figs. 1/2/3/4/5; see #25).
- ``kelly_64``   -- the larger repro variant. NOTE: at 64^2 the kelly regime
  went unstable at ``dt=5e-3`` (NaN during the free run, advective CFL
  suspected; see #25), so this config records the halved ``dt=2.5e-3``;
  it has not been validated at full scale.
- ``inubushi_32`` -- the stronger-turbulence Inubushi-Caulfield comparison
  regime (#11/#12). Not a Kelly benchmark; kept for contrast experiments.

Run directly to print a compact table including the observation dimensions;
for ``kelly_32`` the low-mode observation dimension is
``q_low = (2*5+1)^2 = 121``, which is the rank bound relevant to the
large-ensemble ETKF question (#28): multiplicative-inflation ETKF can only be
full rank on the observed space when ``m - 1 >= q_low``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from da.nse2d import NSE2DConfig, NSE2DTorus, inubushi_caulfield_config


@dataclass(frozen=True)
class ReferenceConfig:
    """One named benchmark configuration (solver + assimilation protocol)."""

    name: str
    # solver
    nx: int
    ny: int
    viscosity: float
    drag: float
    forcing: str  # "kolmogorov" or "diagonal"
    forcing_mode: int
    forcing_amplitude: float
    dt: float
    spin_up: int
    # observation protocol
    obs_interval: int  # J model steps between observations
    kmax_obs: int  # |k_lambda| square cutoff
    gamma: float  # observation noise std
    # ensemble protocol
    ensemble_size: int
    init: str  # "climatology" (snapshots from a long free run) or "perturb"
    init_decorrelate: int  # steps between climatological snapshots
    # validated filter for this config ("po" additive / "etkf" multiplicative)
    filter: str
    additive_alpha: float | None
    note: str = ""

    def build_model(self) -> NSE2DTorus:
        """Instantiate the solver for this configuration."""
        if self.forcing == "diagonal":
            base = NSE2DConfig(
                nx=self.nx,
                ny=self.ny,
                viscosity=self.viscosity,
                drag=self.drag,
                length=2 * np.pi,
            )
            helper = NSE2DTorus(base)
            cfg = NSE2DConfig(
                nx=self.nx,
                ny=self.ny,
                viscosity=self.viscosity,
                drag=self.drag,
                length=2 * np.pi,
                forcing=helper.diagonal_vorticity_forcing(
                    mode=(self.forcing_mode, self.forcing_mode),
                    amplitude=self.forcing_amplitude,
                ),
            )
        else:
            cfg = inubushi_caulfield_config(
                nx=self.nx,
                ny=self.ny,
                viscosity=self.viscosity,
                drag=self.drag,
                forcing_mode=self.forcing_mode,
                forcing_amplitude=self.forcing_amplitude,
                length=2 * np.pi,
            )
        return NSE2DTorus(cfg)

    def observation_dims(self) -> dict[str, int]:
        """Observation dimensions for the low/high/full cases of this config."""
        model = self.build_model()
        low = model.independent_low_mode_observation(kmax=self.kmax_obs)
        high = model.high_mode_observation(kmax=self.kmax_obs)
        return {
            "low": low.obs_dim,
            "high": high.obs_dim,
            "full": model.spectral_state_dim,
            "state": model.state_dim,
        }

    def as_dict(self) -> dict:
        return asdict(self)


REFERENCE_CONFIGS: dict[str, ReferenceConfig] = {
    "kelly_32": ReferenceConfig(
        name="kelly_32",
        nx=32,
        ny=32,
        viscosity=1.0e-2,
        drag=0.0,
        forcing="kolmogorov",
        forcing_mode=5,
        forcing_amplitude=10.0,
        dt=5.0e-3,
        spin_up=1000,
        obs_interval=20,
        kmax_obs=5,
        gamma=0.01,
        ensemble_size=48,
        init="climatology",
        init_decorrelate=200,
        filter="po",
        additive_alpha=0.5,
        note="validated Kelly Figs. 1/2/3/4/5 preset (#25, PR #26)",
    ),
    "kelly_64": ReferenceConfig(
        name="kelly_64",
        nx=64,
        ny=64,
        viscosity=1.0e-2,
        drag=0.0,
        forcing="kolmogorov",
        forcing_mode=5,
        forcing_amplitude=10.0,
        dt=2.5e-3,  # dt=5e-3 went NaN at 64^2 (#25); halved, NOT yet validated
        spin_up=2000,
        obs_interval=40,  # keeps the observation time h = J*dt = 0.1
        kmax_obs=5,
        gamma=0.01,
        ensemble_size=100,
        init="climatology",
        init_decorrelate=400,
        filter="po",
        additive_alpha=0.5,
        note="repro variant; dt halved after 64^2 instability, unvalidated",
    ),
    "inubushi_32": ReferenceConfig(
        name="inubushi_32",
        nx=32,
        ny=32,
        viscosity=1.0e-3,
        drag=1.0e-1,
        forcing="kolmogorov",
        forcing_mode=4,
        forcing_amplitude=1.0,
        dt=1.0e-2,
        spin_up=500,
        obs_interval=20,
        kmax_obs=4,
        gamma=0.01,
        ensemble_size=48,
        init="climatology",
        init_decorrelate=200,
        filter="po",
        additive_alpha=0.5,
        note="stronger-turbulence comparison regime (#11/#12); not a Kelly benchmark",
    ),
}


def main():
    header = (
        f"{'name':<12} {'grid':<8} {'nu':<8} {'drag':<6} {'forcing':<16} "
        f"{'dt':<8} {'J':<4} {'kmax':<5} {'gamma':<6} {'m':<5} "
        f"{'q_low':<6} {'q_high':<7} {'q_full':<7} {'init':<12}"
    )
    print(header)
    print("-" * len(header))
    for cfg in REFERENCE_CONFIGS.values():
        dims = cfg.observation_dims()
        forcing = f"{cfg.forcing}(k={cfg.forcing_mode},A={cfg.forcing_amplitude:g})"
        print(
            f"{cfg.name:<12} {cfg.nx}x{cfg.ny:<5} {cfg.viscosity:<8g} "
            f"{cfg.drag:<6g} {forcing:<16} {cfg.dt:<8g} {cfg.obs_interval:<4} "
            f"{cfg.kmax_obs:<5} {cfg.gamma:<6g} {cfg.ensemble_size:<5} "
            f"{dims['low']:<6} {dims['high']:<7} {dims['full']:<7} {cfg.init:<12}"
        )
    for cfg in REFERENCE_CONFIGS.values():
        if cfg.note:
            print(f"note[{cfg.name}]: {cfg.note}")
    q_low = REFERENCE_CONFIGS["kelly_32"].observation_dims()["low"]
    print(
        f"\nkelly_32 low-mode observation dimension q_low = {q_low} "
        f"(= (2*5+1)^2); multiplicative-inflation ETKF needs m - 1 >= q_low "
        f"to be full rank on the observed space (#28)."
    )


if __name__ == "__main__":
    main()
