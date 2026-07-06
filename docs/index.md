# da_py docs

This directory contains compact operational documentation for the representative
examples.

- `notation.md`: common mathematical notation used in docs and notebooks.
- `examples.md`: representative model/filter settings, synchronized with
  `examples/example_registry.json`.
- `roadmap.md`: examples intentionally left outside v1.

The source of truth for representative example metadata is
`examples/example_registry.json`. When adding or renaming an example, update the
registry, the matching script/notebook, and `docs/examples.md` in the same PR.
