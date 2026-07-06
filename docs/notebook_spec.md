# Representative Notebook Specification

Common specification for the representative tutorial notebooks under
`examples/notebooks/` (issue #38). Model-specific issues (#39, #40, #41)
implement this contract. `examples/example_registry.json` remains the
operational source of truth for paths and parameters.

## Heading structure

Every representative notebook uses exactly this section structure:

1. **Overview** — what model/filter pair is benchmarked and why.
2. **Model** — the state-space model with TeX equations and parameter symbols.
3. **Observation** — observation operator, noise model, observation timing.
4. **DA method** — forecast step and analysis step of the filter in TeX.
5. **Parameters** — table of the representative default values.
6. **Truth and observations** — generation code plus visualization.
7. **Assimilation results** — filter run plus analysis-state visualization.
8. **RMSE** — RMSE time series against the observation-noise scale.
9. **Summary** — one short interpretation paragraph.

## Content contract

- **Self-contained TeX**: Markdown cells state the continuous model, the
  discrete forecast map $x_k = M_k(x_{k-1})$, the observation model
  $y_k = H_k x_k + \varepsilon_k$, $\varepsilon_k \sim N(0, R_k)$, and the
  analysis update of the specific filter, using the symbols of
  `docs/notation.md`. A reader should understand the benchmark without
  opening other files.
- **Concrete parameters**: all default values (time step, observation
  interval, observation noise, ensemble/particle size, inflation,
  localization) appear both in the Parameters section and in the code.
- **Script pairing**: defaults match the paired CLI script under
  `examples/scripts/`, and the major parameters are adjustable from that
  script's command line.
- **Truth and observation visualization**: every notebook plots the truth
  trajectory and the observations used.
- **Analysis visualization**: appropriate to the model — per-component time
  series for Lorenz-63, space-time (Hovmöller) plots for Lorenz-96, and
  multi-panel vorticity snapshots (at minimum truth and analysis) for NSE2D.
- **RMSE time series**: computed with the common definition below and drawn
  with the observation-noise scale as a horizontal reference line.
- **Successful defaults**: the representative default setting must reach an
  assimilation RMSE small relative to the observation-noise scale.

## RMSE and observation-noise-scale convention

For truth $x_k \in \mathbb{R}^{N_x}$ and analysis estimate $\hat{x}_k^a$
(the analysis ensemble mean for ensemble/particle filters, the analysis
state for 3DVar/ExKF),

$$
\mathrm{RMSE}_k
= \sqrt{\frac{1}{N_x} \sum_{i=1}^{N_x} (\hat{x}_{k,i}^a - x_{k,i})^2}.
$$

The observation-noise scale reported for comparison is

$$
\sigma_{\mathrm{obs}} = \sqrt{\operatorname{tr}(R) / N_y},
$$

the root-mean-square observation-error standard deviation. Notebooks report
both, and a successful benchmark shows a time-averaged post-spin-up RMSE
clearly below $\sigma_{\mathrm{obs}}$.

## Execution budget

A full notebook run should stay at roughly **10 minutes or less** on a
laptop. This threshold is provisional and may be revisited when heavier
benchmarks are promoted. CI executes only the lightweight scripts and the
registry consistency checks, not the notebooks.
