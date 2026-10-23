# Local checks and GitHub Actions

Use Python 3.10 or newer and install the package, including its metadata:

```sh
python -m pip install -e '.[dev]'
python -m pytest -ra --tb=short
ruff check .
```

`.github/workflows/ci.yml` runs pytest on Python **3.10 and 3.12** for every
pull request and push to main. Editable installation ensures runtime version
metadata is tested as installed. A separate Ruff job uses the pinned dev
requirement; its rule/version policy is in [lint.md](lint.md).
Nonzero exit codes fail the corresponding check, with failing test names and
tracebacks retained in job logs. Fail-fast is disabled across Python versions.

POT is optional: Python 3.12 installs the `etpf` extra, while Python 3.10
exercises the installation without it. Tests that need POT use import-based
skips when it is absent. Notebook dependencies are included via `dev`;
notebook source is linted, but notebooks are not executed by the workflow.
The existing lightweight smoke tests run as part of pytest. Long archived
benchmarks and inflation-tuning sweeps are not launched separately.

The workflow uses the official [checkout](https://github.com/actions/checkout)
and [setup-python](https://github.com/actions/setup-python) actions, with
read-only repository permissions and no external secrets. Dependency caching
uses pyproject.toml. Repeated runs on the same PR supersede older runs.
