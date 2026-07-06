# Archive Audit and Reuse Map

Inventory of `examples/archive/` assets relevant to the representative
examples (issue #38). Each asset is classified as one of:

- **reuse**: usable almost directly (current API, validated parameters).
- **port**: worth porting, but requires API or structural updates.
- **reference**: keep archived; use only as a parameter or visualization
  reference for the representative examples.
- **archive**: leave archived; no planned reuse.

## Lorenz-63 assets

| Asset | Content | Key parameters | Visualizations | Class |
| --- | --- | --- | --- | --- |
| `l63.ipynb` | Nature-run generation, error growth | `dt=0.01`, `obs_per=12` (`Dt=0.12`), long run | 3D trajectory, per-component time series, error growth | port |
| `l63_etkf.ipynb` | ETKF benchmark and eigenvalue study | obs var `64`, `m=10`, `alpha=1.0`, `H=I`, `P0=38I` | RMSE curves, forecast/analysis eigenvalues | reuse |
| `l63_pf.ipynb` | PF with three resampling schemes | `obs_per=6`, obs var `1.0`, `m=60`, additive inflation `0.1` | RMSE per resampling scheme | reuse |
| `l63_etpf.ipynb` | ETPF (optimal-transport resampling) | obs var `1.0`, `m=100`, additive inflation `0.4` | RMSE curves, particle clouds | reuse |
| `l63_po.ipynb` | Perturbed-observation EnKF, partial obs | `obs_per=3`, obs var `0.1`, `m=2`, `H=[1,0,0]` | RMSE, covariance evolution | port |
| `l63_lypunov.ipynb` | Lyapunov exponents vs model parameters | varied `s, b, r` | exponent heatmaps | reference |
| `l63_ball.pdf`, `l63_error_development.pdf` | Figures from `l63.ipynb` | — | attractor ball, error growth | reference |
| `x_true_l63_full.npy` | Precomputed L63 truth | — | — | reference |

## Lorenz-96 assets

| Asset | Content | Key parameters | Visualizations | Class |
| --- | --- | --- | --- | --- |
| `l96.ipynb` | Nature run, Lyapunov, Hovmöller | `J=40`, `F=8`, `dt=0.01`, `obs_per=5` | time series, Hovmöller (space-time) | port |
| `l96_etkf.ipynb` | ETKF inflation comparison | obs var `1.0`, `m=25`, `alpha=1.0` vs `1.2`, `H=I` | RMSE curves, Hovmöller of truth/obs/analysis | reuse |
| `l96_etkf.py` | ETKF script, modern API | `m=20`, `alpha=1.1`, spin-up | publication-style RMSE (`da.viz`) | reuse |
| `l96_letkf.ipynb` | LETKF hyperparameter sweep | `m=8-20`, `alpha≈1.03`, `c=5-10`, Gaspari-Cohn | RMSE, `alpha`-`c` and `m`-`c` heatmaps | reuse |
| `l96_pf.ipynb` | Large-particle PF | `dt=0.1`, `m=1000`, additive inflation `0.05` | RMSE, ensemble evolution | reference |
| `l96_partial_3dvar.ipynb` | 3DVar with partial observations | every 3rd component observed | RMSE, state snapshots | port |
| `x_true_l96.npy` | Precomputed L96 truth | — | — | reference |

## NSE2D assets

| Asset | Content | Key parameters | Visualizations | Class |
| --- | --- | --- | --- | --- |
| `nse2d_etkf.py` | Spectral-state ETKF smoke benchmark | `16x16`, `m=8`, `alpha=1.02`, `kmax_obs=2` | RMSE printout | reuse |
| `nse2d_reference_configs.py` | Validated benchmark configurations | `kelly_32` (validated), `kelly_64`, `inubushi_32` | — | reuse |
| `nse2d_etkf_large_ensemble_partial_obs.py` | Large-ensemble multiplicative-ETKF sweep (#28) | `m∈{48,122,160}`, `alpha∈{1.0,1.3,1.6}` | error curves, heatmaps | reference |
| `nse2d_partial_obs_enkf.py` | PO-EnKF Kelly et al. reproduction (#13, #25) | `kelly_32`, `m=48`, additive inflation `0.5` | error splits, spread | reference |
| `nse2d_synchronization.py` | Continuous DA / synchronization (#12) | `32x32`, cutoff sweep | relative-error curves | reference |
| `nse2d_torus_forecast.py` | Vorticity snapshot figure generation | `32-64` grids, Kolmogorov forcing | multi-panel vorticity snapshots | reuse |

## Other assets

| Asset | Content | Class |
| --- | --- | --- |
| `henon.ipynb` | Hénon map dynamics | archive |
| `qmc.ipynb`, `qmc_convergence.ipynb` | QMC sampling and toy-EnKF convergence | reference (roadmap item) |
| `dim_ens.ipynb` | Ensemble dimension diagnostic | reference |
| `xnmc.ipynb` | LETKF study on a pre-`da` API (`from letkf import ...`) | archive |

## Reuse guidance for the representative examples

- **L63 3DVar/ExKF** (`l63_3dvar_exkf`): no archived asset exists; keep the
  current script regime, take plotting conventions from `l63_etkf.ipynb`.
- **L63 PF / ETKF / ETPF**: take obs-noise, particle/ensemble sizes and
  additive-inflation values from the corresponding archived notebooks.
- **L96 ETKF / LETKF**: take `m`, `alpha`, `c` regimes and the Hovmöller
  truth/analysis visualization from `l96_etkf.ipynb` / `l96_letkf.ipynb`.
- **NSE2D ETKF**: keep the lightweight `inubushi_caulfield_config` regime of
  the current script; take multi-panel vorticity plotting from
  `nse2d_torus_forecast.py`. The Kelly-regime scripts remain research
  benchmarks outside the representative tutorial.
