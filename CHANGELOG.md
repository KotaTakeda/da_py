# Changelog

This project uses a lightweight Semantic Versioning policy:

- Patch versions, such as `0.7.1`, are for compatible bug fixes, documentation
  corrections, and internal maintenance.
- Minor versions, such as `0.7.0`, are for backward-compatible user-visible
  additions, including new models, filters, examples, diagnostics, or docs.
- Major versions, such as `1.0.0`, are reserved for intentional breaking API
  changes after the package interface is considered stable enough to support
  that distinction.

The package is still research software. Before `1.0.0`, minor versions may
still include small API adjustments, but such changes should be called out in
these notes.

The source of truth for the package version is `project.version` in
`pyproject.toml`. At runtime, `da.__version__` is read from the installed package
metadata.

## 0.7.0 - 2026-07-06

### Added

- EnKF-N adaptive inflation support through `EnKFN` and
  `estimate_l1_enkfn_dual`.
- A two-dimensional Navier-Stokes torus model with spectral-state utilities,
  low/high-mode projections, and reference benchmark configurations.
- NSE2D synchronization, partial-observation, large-ensemble, and ETKF example
  scripts, now retained under `examples/archive/` as research-history material.
- A lightweight visualization layer under `da.viz`, including vendored
  publication-style plotting helpers.
- A minimal `ExKF` implementation for extended Kalman filter examples.
- Representative example structure with `examples/scripts/`,
  `examples/notebooks/`, `examples/archive/`, and
  `examples/example_registry.json`.
- Markdown docs for representative examples, notation, and roadmap under
  `docs/`.

### Changed

- Multiplicative inflation semantics are documented and tested consistently as
  anomaly inflation, `A -> alpha A` and `P -> alpha^2 P`.
- Particle-filter weight calculations and ensemble Kalman analysis paths have
  additional tests and robustness checks.
- Package discovery now uses `setuptools` package finding under `src/`.

### Notes

- No PyPI publication is part of this version bump.
- Archived examples are preserved but are not guaranteed to track the current
  public API.
