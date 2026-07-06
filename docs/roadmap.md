# Examples Roadmap

The v1 examples prioritize a small set of representative model/filter
combinations. Future work should add or promote examples only when each item can
ship with a script, a notebook, and a registry entry.

Planned candidates:

- QMC and QMC convergence tutorials.
- Henon map examples.
- Lorenz96 particle-filter examples.
- NSE2D partial-observation, large-ensemble, and synchronization examples.
- GitHub Pages rendering of notebook tutorials.

Benchmark/tutorial extensions beyond the v1 representative notebooks
(`docs/notebook_spec.md`):

- Diagnostics beyond RMSE: ensemble spread and spread-skill ratio, rank
  histograms, observation-space innovations, and error splits between
  observed and unobserved components.
- Observation-network studies: partial and irregular observation points,
  spectral cutoffs for NSE2D, and observation-density sweeps.
- Exercises and parameter studies: guided inflation/localization sweeps and
  ensemble-size studies built on the representative defaults.
- Revisit the provisional ~10 minute per-notebook execution budget when
  heavier benchmarks are promoted.

Operational rule: update `examples/example_registry.json` first, then add the
script, notebook, and docs entry in the same change.
