# Local checks and GitHub Actions

Use Python 3.10 or newer and install the package, including its metadata:

```sh
python -m pip install -e '.[dev]'
python -m pytest -ra --tb=short
ruff check .
```

`.github/workflows/ci.yml` runs pytest on Python **3.10.22, 3.11.17, 3.12.15, 3.13.17, 3.14.9 and 3.15.1** for every
pull request and push to main. Editable installation ensures runtime version
metadata is tested as installed. A separate Ruff job uses the pinned dev
requirement; its rule/version policy is in [lint.md](lint.md).
Nonzero exit codes fail the corresponding check, with failing test names and
tracebacks retained in job logs. Fail-fast is disabled across Python versions.

POT is optional: Python 3.15.1 installs the `etpf` extra, while the other versions
exercise the installation without it. Tests that need POT use import-based
skips when it is absent. Notebook dependencies are included via `dev`;
notebook source is linted, but notebooks are not executed by the workflow.
The existing lightweight smoke tests run as part of pytest. Long archived
benchmarks and inflation-tuning sweeps are not launched separately.

The workflow uses the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) actions, with
read-only repository permissions and no external secrets. Dependency caching
uses pyproject.toml. Repeated runs on the same PR supersede older runs.

## Version policy (verified 2026-10-23)

The matrix covers the declared `requires-python = ">=3.10"` lower bound and
all currently supported stable CPython minor series through 3.15. Python 3.10
reached upstream end of life on 2026-10-01; it remains in CI for this package's
existing compatibility contract. This does not imply upstream security support.
The current stable release is 3.15.1; 3.16 prereleases are excluded.
See [Python release status](https://www.python.org/downloads/).

Actions are pinned to the full commit SHAs of stable releases
[checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) and
[setup-python v7.1.0](https://github.com/actions/setup-python/releases/tag/v7.1.0).
Both use Node.js 24 and require Actions Runner >= 2.327.1. Checkout's authenticated
Git commands inside Docker container actions additionally need >= 2.329.0;
this workflow does not use container actions. GitHub-hosted `ubuntu-latest`
provides the runner; the actual runner/image versions are retained in each job's
setup log. Interpreter versions are fixed to the patch releases above, and Ruff
remains exactly 0.17.2 from project metadata, matching PR #82.

This fixes Actions, Python, and Ruff versions rather than selecting an unbounded
`latest` interpreter. The hosted OS image and other dependencies are intentionally
not locked: CI checks current dependency compatibility, so the whole environment
is not bit-for-bit reproducible. Review patch pins periodically and when adopting
a new supported Python minor release.
