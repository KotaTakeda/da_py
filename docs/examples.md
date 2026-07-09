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
| `l96_enkfn` | Lorenz96 | EnKF-N / tuned ETKF | `examples/scripts/l96_enkfn.py` | `examples/notebooks/l96_enkfn.ipynb` | representative |
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

### EnKF-N benchmark and ETKF tuning

The `l96_enkfn` example compares EnKF-N (adaptive inflation) against a
fixed-inflation ETKF tuned offline, on a 60-variable, 2/3-observed Lorenz-96
setup. The tuned inflation `alpha_* = 1.08` is selected by a separate
multi-seed sweep script. Like the other example scripts, both are
regeneration-only: the tuning script writes its CSV summary and inflation-sweep
figure under `examples/output/` (git-ignored, produced on demand), and the
benchmark script prints its diagnostics and writes optional figures only when
`--series-csv` / `--figure-output` are given.

```sh
python examples/scripts/l96_enkfn_tuning.py   # sweep alpha, pick alpha_*, write CSV + figure
python examples/scripts/l96_enkfn.py          # benchmark tuned ETKF vs EnKF-N
```

The multi-seed ETKF inflation sweep (5 seeds, 400 cycles, post-spin-up mean
RMSE) selects the seed-averaged minimum; the representative result is:

| `alpha` | RMSE mean | RMSE std |
| --- | --- | --- |
| 1.00 | 4.164 | 0.140 |
| 1.02 | 3.796 | 0.182 |
| 1.04 | 2.030 | 1.039 |
| 1.06 | 1.146 | 0.705 |
| **1.08** | **0.328** | **0.013** |
| 1.10 | 0.336 | 0.005 |
| 1.12 | 0.353 | 0.004 |
| 1.14 | 0.376 | 0.007 |
| 1.16 | 0.393 | 0.006 |
| 1.18 | 0.410 | 0.004 |
| 1.20 | 0.429 | 0.009 |

The sharp drop between `alpha = 1.06` and `1.08` (and the large std below it) is
the divergence cliff discussed below; `alpha_* = 1.08` is the seed-averaged
minimum.

The benchmark uses `m = 30` members: at `Nx = 60` the Lorenz-96 unstable
subspace is roughly 20-dimensional, so a global (unlocalized) filter needs an
ensemble comfortably above that. At the smaller `m = 25` both filters are only
marginally stable and diverge for a fraction of initial ensembles; `m = 30`
converges reliably and lets EnKF-N match the tuned ETKF (post-spin-up RMSE
`~0.33`, below `sigma_obs = 1`) with no inflation tuning.

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
