# Notation

This page defines the notation used by da_py docs and the representative
tutorial notebooks. The notation is package-oriented and may differ from the
author's thesis notation.

| Symbol | Meaning |
| --- | --- |
| $x_{n} \in \mathbb{R}^{N_{x}}$ | true state at assimilation time $n$ |
| $x_{n}^{f}$ | forecast state before assimilating $y_{n}$ |
| $x_{n}^{a}$ | analysis state after assimilating $y_{n}$ |
| $M_{n}$ | nonlinear forecast map from $n-1$ to $n$ |
| $F_{n}$ | tangent-linear or finite-difference Jacobian of $M_{n}$ |
| $y_{n} \in \mathbb{R}^{N_{y}}$ | observation vector |
| $H_{n}$ | linear observation operator or its local linear representation |
| $R_{n}$ | observation-error covariance |
| $B$ | background-error covariance used by 3DVar |
| $P_{n}^{f}$, $P_{n}^{a}$ | forecast / analysis error covariance |
| $X_{n} \in \mathbb{R}^{m \times N_{x}}$ | ensemble matrix whose rows are members |
| $\bar{x}_{n}$ | ensemble mean |
| $A_{n}$ | ensemble perturbation (anomaly) matrix |
| $m$ | ensemble size or number of particles |
| $w^{(i)}$ | particle weight of member $i$ |
| $\alpha$ | anomaly inflation factor; $A \to \alpha A$ so $P \to \alpha^{2} P$ |
| $\rho$, $c$ | localization function / localization radius |

## Reference notation comparison

The project default keeps code-facing symbols plain and finite-dimensional.
Reference-specific symbols and fonts are used only when comparing with a cited
source or reproducing a derivation.

| Concept | da_py / this note | Takeda thesis | Takeda and Miyoshi paper notation | Remarks |
| --- | --- | --- | --- | --- |
| State | $`x \in \mathbb{R}^{N_{x}}`$ | $`u_{n}\in\mathcal{H}`$ for the true state; $`\varpi_{n}\in\mathcal{H}`$ for a KF/3DVar estimate; $`v^{(k)}\in\mathcal{H}`$ for ensemble members | $`{\boldsymbol{x}}_{n} \in \mathbb{R}^{N_{x}}`$ | da_py keeps code-facing symbols plain; the paper-facing notation uses bold finite-dimensional vectors. |
| True state | $`x_{n}`$ or `x_true` | $`u_{n}`$ | $`{\boldsymbol{x}}_{n}`$ | Code names such as `x_true` remain backticked implementation identifiers. |
| Observation | $`y_{n}`$ or `y_obs` | $`Y_{n}=h(U_{n})+\eta_{n}`$ for random variables; $`y_{n}`$ for realized data in algorithms | $`{\boldsymbol{y}}_{n}=\boldsymbol{H}{\boldsymbol{x}}_{n}+\eta_{n}`$ | Use lower-case $`y_{n}`$ in algorithmic formulas; use upper-case only for random variables when following the thesis. |
| Observation operator | $`H_{n}`$ or $`H`$ | $`h`$ for a general observation function; $`H`$ for a linear operator | $`\boldsymbol{H}\in\mathbb{R}^{N_{y}\times N_{x}}`$ | da_py uses `H` for both ndarray and callable observations; the paper-facing notation uses a matrix observation operator. |
| Observation error covariance | $`R_{n}`$ or $`R`$ | $`R`$ | $`\boldsymbol{R}\in\mathbb{R}^{N_{y}\times N_{y}}`$ | The thesis also uses $`r^{2} I_{\mathcal{H}}`$ under full-observation assumptions. |
| Ensemble size | $`m`$ | $`m`$ | $`m\in\mathbb{N}`$, $`m^{*}`$ | $`m^{*}`$ is the minimum ensemble size in the paper-facing notation. |
| Ensemble member | $`x_{n}^{(k)}`$ | $`v^{(k)} \in \mathcal{H}`$ | $`{\boldsymbol{x}}_{n}^{f,(k)}`$, $`{\boldsymbol{x}}_{n}^{a,(k)}`$ | The paper puts forecast/analysis labels before the parenthesized member index. |
| Ensemble | $`X_{n}=[x_{n}^{(k)}]_{k=1}^{m}`$ | $`\boldsymbol{V}=[v^{(k)}]_{k=1}^{m}\in\mathcal{H}^{m}`$ | $`\boldsymbol{X}\in\mathbb{R}^{N_{x}\times m}`$, $`{\boldsymbol{X}}_{0}`$ | The thesis uses bold letters for a set of vectors; da_py stores members as rows in implementation arrays. |
| Ensemble mean | $`\bar{x}_{n}=\frac{1}{m}\sum_{k=1}^{m} x_{n}^{(k)}`$ | $`\overline{v}=\frac{1}{m}\sum_{k=1}^{m}v^{(k)}`$ for a generic ensemble; $`{\overline{\widehat{v}}}_{n}`$ and $`\overline{v}_{n}`$ for forecast and analysis means | $`{\overline{\boldsymbol{x}}}_{n}^{f}=\frac{1}{m}\sum_{k=1}^{m}{\boldsymbol{x}}_{n}^{f,(k)}`$; $`{\overline{\boldsymbol{x}}}_{n}^{a}`$ is the analysis mean updated by the ETKF formula | da_py uses $`\bar{x}_{n}`$ to avoid overloading the state symbol. The 2026 paper explicitly defines the forecast mean and then uses the analysis mean in the update and ensemble reconstruction. |
| Ensemble deviation / anomaly | $`A_{n}=X_{n}-\mathbf{1}_{m}\bar{x}_{n}^{\top}`$ | $`d\boldsymbol{V}=[v^{(k)}-\overline{v}]_{k=1}^{m}`$ with $`\boldsymbol{V}=\overline{v}\mathbf{1}+d\boldsymbol{V}`$ | $`{\boldsymbol{V}}_{n}^{f}=[{\boldsymbol{x}}_{n}^{f,(1)}-{\overline{\boldsymbol{x}}}_{n}^{f},\ldots,{\boldsymbol{x}}_{n}^{f,(m)}-{\overline{\boldsymbol{x}}}_{n}^{f}]`$, $`{\boldsymbol{V}}_{n}^{a}`$ | In the paper, $`\boldsymbol{V}`$ denotes perturbation/anomaly matrices, not the ensemble itself. A separate "mean ensemble" row is unnecessary because it is only the mean part of this decomposition. |
| Sample covariance | $`P_{n}=\frac{1}{m-1}A_{n}^{\top}A_{n}`$ | $`\mathrm{Cov}_{m}(\boldsymbol{V})=\frac{1}{m-1}d\boldsymbol{V}d\boldsymbol{V}^{*}`$ | $`{\boldsymbol{C}}_{n}^{f}=\frac{1}{m-1}{\boldsymbol{V}}_{n}^{f}({\boldsymbol{V}}_{n}^{f})^{\top}`$ | da_py arrays are row-major, so $`A_{n}^{\top}A_{n}`$ gives the state covariance. Takeda thesis uses $`{}^{*}`$ for Hilbert-space adjoints. |
| Forecast ensemble | $`X_{n}^{f}`$ | $`{\widehat{\boldsymbol{V}}}_{n}=[{\widehat{v}}_{n}^{(k)}]_{k=1}^{m}`$ | $`{\boldsymbol{x}}_{n}^{f,(k)}=\Psi({\boldsymbol{x}}_{n-1}^{a,(k)})`$ | Takeda thesis uses hat notation for prediction/forecast quantities rather than superscript $`f`$. |
| Analysis ensemble | $`X_{n}^{a}`$ | $`{\boldsymbol{V}}_{n}=[v_{n}^{(k)}]_{k=1}^{m}`$ | $`{\boldsymbol{x}}_{n}^{a,(k)}={\overline{\boldsymbol{x}}}_{n}^{a}+{\boldsymbol{v}}_{n}^{a,(k)}`$ | $`{\boldsymbol{v}}_{n}^{a,(k)}`$ is the $`k`$th column of $`{\boldsymbol{V}}_{n}^{a}`$. |
| Forecast covariance | $`P_{n}^{f}`$ | $`{\widehat{P}}_{n}=\mathrm{Cov}_{m}(d{\widehat{\boldsymbol{V}}}_{n})`$ | $`{\boldsymbol{C}}_{n}^{f}`$ | In the thesis, $`\mathrm{Cov}_{m}({\widehat{\boldsymbol{V}}}_{n})=\mathrm{Cov}_{m}(d{\widehat{\boldsymbol{V}}}_{n})`$ by Section 2.3.1. |
| Analysis covariance | $`P_{n}^{a}`$ | $`\mathrm{Cov}_{m}({\boldsymbol{V}}_{n})`$ | not explicitly named in the ETKF definition | The analysis spread is represented by $`{\boldsymbol{V}}_{n}^{a}`$. |
| Innovation | $`y_{n}-H_{n}\bar{x}_{n}^{f}`$ | $`y_{n}-H{\overline{\widehat{v}}}_{n}`$ | $`{\boldsymbol{y}}_{n}-\boldsymbol{H}{\overline{\boldsymbol{x}}}_{n}^{f}`$ | Code variable: `dy`; do not typeset `dy` as a differential. |
| Kalman gain | $`K_{n}=P_{n}^{f}H_{n}^{\top}(H_{n}P_{n}^{f}H_{n}^{\top}+R_{n})^{-1}`$ | $`K_{n}={\widehat{P}}_{n}H^{*}(H{\widehat{P}}_{n}H^{*}+R)^{-1}`$ | $`{\boldsymbol{K}}_{n}={\boldsymbol{C}}_{n}^{f}\boldsymbol{H}^{\top}(\boldsymbol{H}{\boldsymbol{C}}_{n}^{f}\boldsymbol{H}^{\top}+\boldsymbol{R})^{-1}`$ | The paper uses bold $`{\boldsymbol{K}}_{n}`$ and $`\top`$. |
| Transform matrix | $`T`$ | $`T_{n}\in\mathbb{R}^{m\times m}`$ | $`{\boldsymbol{T}}_{n}=\left(\boldsymbol{I}_{m}+\frac{1}{m-1}({\boldsymbol{V}}_{n}^{f})^{\top}\boldsymbol{H}^{\top}\boldsymbol{R}^{-1}\boldsymbol{H}{\boldsymbol{V}}_{n}^{f}\right)^{-1/2}`$ | $`{\boldsymbol{T}}_{n}`$ is a matrix square root chosen symmetric positive definite in the paper. |
| Inflation parameter | $`\alpha`$ | $`{\widehat{P}}_{n}^{\alpha}=\alpha^{2}{\widehat{P}}_{n}`$, $`d{\widehat{\boldsymbol{V}}}_{n}^{\alpha}=\alpha d{\widehat{\boldsymbol{V}}}_{n}`$ | $`\alpha>1`$ | The 2026 paper uses multiplicative covariance inflation in Algorithm 3. |
| Minimum ensemble size | not used as a package symbol | not part of the ETKF notation here | $`m^{*}=N_{+}+1`$ | Added because this is central to the 2026 paper's notation and conclusions. |
| Error metric | RMSE in examples | problem-dependent | $`\mathrm{SE}_{n}=\|{\boldsymbol{x}}_{n}-{\overline{\boldsymbol{x}}}_{n}^{a}\|^{2}`$, $`\mathrm{RMSE}_{n}`$ | Typeset metric names in roman capitals. |

References checked:

- Kota Takeda, *Error Analysis of the Ensemble Square Root Filter for Dissipative
  Dynamical Systems*, doctoral thesis, Department of Mathematics, Kyoto
  University, first version February 1, 2025; modified January 1, 2026.
  Available at <https://kotatakeda.github.io/math/pdf/thesis.pdf>. The URL was
  checked on 2026-07-10. See Sections 2.3.1, 3.1, 4.1--4.3, and Definition
  4.13 for $`\boldsymbol{V}`$, the ensemble mean
  $`\overline{v}=\frac{1}{m}\sum_{k} v^{(k)}`$, $`\overline{v}\mathbf{1}`$,
  $`d\boldsymbol{V}`$, $`\mathrm{Cov}_{m}(\boldsymbol{V})`$, $`\varpi_{n}`$,
  $`\widehat{\boldsymbol{V}}_{n}`$, $`\widehat{P}_{n}`$, and $`T_{n}`$.
- K. Takeda and T. Miyoshi, "Noise-scaled accuracy of the ensemble Kalman filter
  with an instability-based minimum ensemble size," *Nonlinear Processes in
  Geophysics*, 33, 335--346, 2026.
  <https://doi.org/10.5194/npg-33-335-2026>. The DOI was checked on
  2026-07-10 and resolves to the NPG article page.
- Existing repository code and examples: `src/etkf.py`, `src/letkf.py`,
  `src/po.py`, `src/var3d.py`, and representative Lorenz-63/Lorenz-96
  notebooks.

## State-space and observation model

All representative examples use the discrete-time state-space model

$$
x_{n} = M_{n}(x_{n-1}), \qquad
y_{n} = H_{n} x_{n} + \varepsilon_{n}, \quad \varepsilon_{n} \sim N(0, R_{n}),
$$

where $M_{n}$ advances the model over one assimilation window (numerical
integration of the underlying ODE/PDE with the fourth-order Runge-Kutta
scheme, `da.scheme.rk4`).

## Models

**Lorenz-63** ($N_{x} = 3$, chaotic for the standard parameters
$\sigma = 10$, $r = 28$, $b = 8/3$):

$$
\dot{x} = \sigma (y - x), \qquad
\dot{y} = x (r - z) - y, \qquad
\dot{z} = x y - b z.
$$

**Lorenz-96** ($N_{x} = J$ variables on a ring, forcing $F$; chaotic for
$J = 40$, $F = 8$):

$$
\dot{x}_{j} = (x_{j+1} - x_{j-2})\, x_{j-1} - x_{j} + F,
\qquad j = 1, \dots, J \ (\text{indices mod } J).
$$

**2D Navier-Stokes (vorticity form on the torus)** with viscosity $\nu$ and
stationary (Kolmogorov-type) forcing $f$ (the representative example uses no
linear drag):

$$
\partial_{t} \omega + (u \cdot \nabla)\, \omega
= \nu \Delta \omega + f,
\qquad u = \nabla^{\perp} \Delta^{-1} \omega ,
$$

solved pseudo-spectrally; the DA state is the packed `rfft2` vorticity
spectrum (`NSE2DTorus.to_spectral_state`).

## Filters

**3DVar** uses a fixed background covariance $B$ and gain

$$
K = B H^{\top} (H B H^{\top} + R)^{-1}, \qquad
x_{n}^{a} = x_{n}^{f} + K (y_{n} - H x_{n}^{f}).
$$

**ExKF** propagates the covariance through the linearized forecast map,

$$
P_{n}^{f} = F_{n} P_{n-1}^{a} F_{n}^{\top} + Q_{n}, \qquad
K_{n} = P_{n}^{f} H^{\top} (H P_{n}^{f} H^{\top} + R)^{-1},
$$

with $x_{n}^{a} = x_{n}^{f} + K_{n} (y_{n} - H x_{n}^{f})$ and
$P_{n}^{a} = (I - K_{n} H) P_{n}^{f}$.

**ETKF** decomposes the forecast ensemble into mean and anomalies,
$`X_{n}^{f} = \mathbf{1}\bar{x}_{n}^{f} + A_{n}^{f}`$, and forms the analysis
in the $`m`$-dimensional ensemble space. In the compact equations below,
$`A^{f}`$ denotes the transposed anomaly matrix used by the transform step,
$`A^{f}\in\mathbb{R}^{N_{x}\times m}`$:

$$
\tilde{P} = \left[(m-1) I + (H A^{f})^{\top} R^{-1} (H A^{f})\right]^{-1},
$$

$$
\bar{x}^{a} = \bar{x}^{f} + A^{f} \tilde{P} (H A^{f})^{\top} R^{-1}
(y - H \bar{x}^{f}), \qquad
A^{a} = A^{f} \big[(m-1) \tilde{P}\big]^{1/2}.
$$

Multiplicative inflation is applied to the anomalies, $A \to \alpha A$, so
the covariance is inflated by $\alpha^{2}$.

**LETKF** performs the ETKF update independently for each state component
$j$, using observation-error covariance localized by the Gaspari-Cohn
function $\rho$ with radius $c$:
$R^{-1} \to R^{-1} \circ \rho(d_{j}/c)$, where $d_{j}$ is the distance
between component $j$ and each observation location.

**Particle filter (bootstrap)** propagates $m$ particles through the model,
updates weights with the Gaussian likelihood

$$
w_{n}^{(i)} \propto w_{n-1}^{(i)}
\exp\!\Big(-\tfrac{1}{2}\,\|y_{n} - H x_{n}^{(i)}\|_{R^{-1}}^{2}\Big),
$$

and resamples (multinomial / systematic / residual) when the effective
sample size $N_{\mathrm{eff}} = 1 / \sum_{i} (w^{(i)})^{2}$ falls below a
threshold. Small additive noise (`add_inflation`) counteracts weight
collapse.

**ETPF** replaces stochastic resampling by a deterministic linear ensemble
transform $X^{a} = T^{\top} X^{f}$, where $T$ is $m$ times the optimal-transport
coupling that maps the weighted forecast ensemble onto the uniform analysis
ensemble with minimal squared-Euclidean cost (solved with the POT package).

## RMSE convention

See `docs/contributing/notebook_spec.md`. Analysis RMSE is
$`\mathrm{RMSE}_{n}=\sqrt{\frac{1}{N_{x}}\sum_{i}(\hat{x}_{n,i}^{a}-x_{n,i})^{2}}`$,
where $`\hat{x}_{n}^{a}`$ is the analysis (ensemble-mean) estimate. It is
compared against the observation-noise scale
$`\sigma_{\mathrm{obs}}=\sqrt{\frac{\mathrm{tr}(R)}{N_{y}}}`$.
