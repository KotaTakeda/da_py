# da_py documentation

The documentation is organized by purpose. Start with the example guide for a
runnable introduction, then use the algorithm and reference pages for precise
mathematical and API conventions.

## Guides

- [Representative examples](guides/examples.md): supported scripts,
  notebooks, benchmark settings, and execution instructions.

## Algorithms

- [EnKF-N derivation and implementation map](algorithms/enkfn.md): finite-size
  hierarchy, scalar dual optimization, and mapping to `da.enkfn` and ETKF.
- [Gaussian model noise](algorithms/model_noise.md): the ensemble filters' `Q`
  argument and its distinction from covariance inflation.

## Reference

- [Notation](reference/notation.md): shared mathematical notation and storage
  conventions used by the documentation and notebooks.
- [RNG policy](reference/rng_policy.md): reproducibility and random-number
  ownership across the package.

## Contributing

- [Notebook specification](contributing/notebook_spec.md): requirements for
  representative tutorial notebooks.
- [Examples roadmap](contributing/examples_roadmap.md): examples intentionally
  deferred beyond the current representative set.

The source of truth for representative-example metadata is
`examples/example_registry.json`. When adding or renaming an example, update
the registry, the matching script or notebook, and
[`guides/examples.md`](guides/examples.md) in the same change.
