# da_py docs

This directory contains compact operational documentation for the representative
examples.

- `notation.md`: common mathematical notation used in docs and notebooks.
- `examples.md`: representative model/filter settings, synchronized with
  `examples/example_registry.json`.
- `model_noise.md`: additive Gaussian model noise (`da.noise`) and how it
  differs from the inflation mechanisms.
- `rng_policy.md`: the project-wide random-number policy.
- `roadmap.md`: examples intentionally left outside v1.

The source of truth for representative example metadata is
`examples/example_registry.json`. When adding or renaming an example, update the
registry, the matching script/notebook, and `docs/examples.md` in the same PR.

Contributor/maintenance specifications for authoring the representative
notebooks live under `docs/contributing/` and are not part of the user-facing
documentation.
