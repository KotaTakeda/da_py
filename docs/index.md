# da_py docs

This directory contains compact operational documentation for the representative
examples.

- `notation.md`: common mathematical notation used in docs and notebooks.
- `examples.md`: representative model/filter settings, synchronized with
  `examples/example_registry.json`.
- `roadmap.md`: examples intentionally left outside v1.
- `notebook_spec.md`: common specification for the representative tutorial
  notebooks (headings, TeX content contract, RMSE convention, run-time
  budget).
- `archive_audit.md`: inventory and reuse map of `examples/archive/`.

The source of truth for representative example metadata is
`examples/example_registry.json`. When adding or renaming an example, update the
registry, the matching script/notebook, and `docs/examples.md` in the same PR.
