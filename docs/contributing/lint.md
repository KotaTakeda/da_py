# Lint baseline

Install the development environment with `python -m pip install -e '.[dev]'`.
Run `ruff check .` and `python -m pytest -q` from the repository root.

Ruff is pinned to **0.17.2** in the dev extra and `required-version`.
Its [default rule set](https://docs.astral.sh/ruff/default-rules/) is the
baseline; upgrading Ruff requires reviewing rule changes and running both
checks. `target-version = "py310"` matches the minimum supported Python.
Python modules, scripts, tests and notebook code cells are included.
`examples/archive` remains excluded because it preserves historical work.
Markdown prose and equations are outside the lint scope.

Import order and unnecessary suppressions are corrected without rewriting
notebook markdown, outputs or metadata. Two file-specific suppressions
retain the vendored initializer's upstream import/export grouping (I001,
RUF022), as required by its `_VENDORED.md`; other checks still apply.
LETKF localization methods no longer use a global function cache that keeps
discarded filter instances alive; a weak-reference regression verifies this.
Loop variables used by finite-difference test callbacks are bound explicitly.
