# Representative Examples

This page is synchronized manually with `examples/example_registry.json`. Treat
the registry as the operational source of truth for paths and parameters.

| ID | Model | Filter | Script | Notebook | Status |
| --- | --- | --- | --- | --- | --- |
| `l63_3dvar_exkf` | Lorenz63 | 3DVar / ExKF | `examples/scripts/l63_3dvar_exkf.py` | `examples/notebooks/l63_3dvar_exkf.ipynb` | representative |
| `l63_pf` | Lorenz63 | Particle Filter | `examples/scripts/l63_pf.py` | `examples/notebooks/l63_pf.ipynb` | representative |
| `l63_etkf` | Lorenz63 | ETKF | `examples/scripts/l63_etkf.py` | `examples/notebooks/l63_etkf.ipynb` | representative |
| `l63_etpf` | Lorenz63 | ETPF | `examples/scripts/l63_etpf.py` | `examples/notebooks/l63_etpf.ipynb` | optional; requires POT |
| `l96_etkf` | Lorenz96 | ETKF | `examples/scripts/l96_etkf.py` | `examples/notebooks/l96_etkf.ipynb` | representative |
| `l96_letkf` | Lorenz96 | LETKF | `examples/scripts/l96_letkf.py` | `examples/notebooks/l96_letkf.ipynb` | representative |
| `nse2d_etkf` | NSE2D | ETKF | `examples/scripts/nse2d_etkf.py` | `examples/notebooks/nse2d_etkf.ipynb` | representative |

## Usage

Run scripts from the repository root:

```sh
python examples/scripts/l63_etkf.py
python examples/scripts/l96_letkf.py
python examples/scripts/nse2d_etkf.py
```

The scripts use small default settings intended for smoke tests and benchmark
orientation, not publication-quality experiments. Major benchmark parameters
(observation noise, inflation, ensemble/particle size, cycles) are exposed as
CLI options; run a script with `--help` for the full list.

## Notebooks

Each representative example has a tutorial notebook under
`examples/notebooks/` that follows `docs/contributing/notebook_spec.md`: it states the
model, observation model, and filter update in TeX (symbols defined in
`docs/notation.md`), visualizes truth/observations and the analysis, and
plots the analysis RMSE against the observation-noise scale
$\sigma_{\mathrm{obs}} = \sqrt{\operatorname{tr}(R)/N_y}$. Notebooks use
longer runs than the smoke-test scripts; each completes well within the
~10 minute budget.

## Archive

Older notebooks, scripts, generated PDFs, and local example arrays were moved
to `examples/archive/`. They are retained as research history but are not kept
in sync with the current public API.
