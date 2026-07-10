# Notation

This page defines the notation used by da_py docs and the representative
tutorial notebooks. The notation is package-oriented and may differ from the
author's thesis notation.

| Symbol | Meaning |
| --- | --- |
| $x_n \in \mathbb{R}^{N_x}$ | true state at assimilation time $n$ |
| $x_n^f$ | forecast state before assimilating $y_n$ |
| $x_n^a$ | analysis state after assimilating $y_n$ |
| $M_n$ | nonlinear forecast map from $n-1$ to $n$ |
| $F_n$ | tangent-linear or finite-difference Jacobian of $M_n$ |
| $y_n \in \mathbb{R}^{N_y}$ | observation vector |
| $H_n$ | linear observation operator or its local linear representation |
| $R_n$ | observation-error covariance |
| $B$ | background-error covariance used by 3DVar |
| $P_n^f$, $P_n^a$ | forecast / analysis error covariance |
| $X_n \in \mathbb{R}^{m \times N_x}$ | ensemble matrix whose rows are members |
| $\bar{x}_n$ | ensemble mean |
| $A_n$ | ensemble perturbation (anomaly) matrix |
| $m$ | ensemble size or number of particles |
| $w^{(i)}$ | particle weight of member $i$ |
| $\alpha$ | anomaly inflation factor; $A \to \alpha A$ so $P \to \alpha^2 P$ |
| $\rho$, $c$ | localization function / localization radius |

## Reference notation comparison

The project default keeps code-facing symbols plain and finite-dimensional.
Reference-specific symbols and fonts are used only when comparing with a cited
source or reproducing a derivation.

| Concept | da_py / this note | Takeda thesis | Takeda and Miyoshi paper notation | Remarks |
| --- | --- | --- | --- | --- |
| State | $x \in \mathbb{R}^{N_x}$ | $u_n\in H$ for the true state; $\varpi_n\in H$ for a KF/3DVar estimate; $v^{(k)}\in H$ for ensemble members | $\boldsymbol{x}_n \in \mathbb{R}^{N_x}$ | da_py keeps code-facing symbols plain; the paper-facing notation uses bold finite-dimensional vectors. |
| True state | $x_n$ or `x_true` | $u_n$ | $\boldsymbol{x}_n$ | Code names such as `x_true` remain backticked implementation identifiers. |
| Observation | $y_n$ or `y_obs` | $Y_n=h(U_n)+\eta_n$ for random variables; $y_n$ for realized data in algorithms | $\boldsymbol{y}_n=\boldsymbol{H}\boldsymbol{x}_n+\boldsymbol{\eta}_n$ | Use lower-case $y_n$ in algorithmic formulas; use upper-case only for random variables when following the thesis. |
| Observation operator | $H_n$ or $H$ | $h$ for a general observation function; $H$ for a linear operator | $\boldsymbol{H}\in\mathbb{R}^{N_y\times N_x}$ | da_py uses `H` for both ndarray and callable observations; the paper-facing notation uses a matrix observation operator. |
| Observation error covariance | $R_n$ or $R$ | $R$ | $\boldsymbol{R}\in\mathbb{R}^{N_y\times N_y}$ | The thesis also uses $r^2 I_H$ under full-observation assumptions. |
| Ensemble size | $m$ | $m$ | $m\in\mathbb{N}$, $m^*$ | $m^*$ is the minimum ensemble size in the paper-facing notation. |
| Ensemble member | $x_n^{(k)}$ | $v^{(k)} \in H$ | $\boldsymbol{x}_n^{f(k)}$, $\boldsymbol{x}_n^{a(k)}$ | The paper puts forecast/analysis labels before the parenthesized member index. |
| Ensemble | $X_n=[x_n^{(k)}]_{k=1}^m$ | $V=[v^{(k)}]_{k=1}^m \in H^m$ | $\boldsymbol{X}\in\mathbb{R}^{N_x\times m}$, $\boldsymbol{X}_0$ | The paper and algorithms use bold ensemble matrices; da_py stores members as rows in implementation arrays. |
| Ensemble mean | $\bar{x}_n=m^{-1}\sum_{k=1}^m x_n^{(k)}$ | $v=m^{-1}\sum_{k=1}^m v^{(k)}$ for a generic ensemble; $\hat{v}_n$ and $v_n$ for forecast and analysis means | $\boldsymbol{x}_n^f=m^{-1}\sum_{k=1}^m\boldsymbol{x}_n^{f(k)}$, $\boldsymbol{x}_n^a$ | da_py uses $\bar{x}_n$ to avoid overloading the state symbol. |
| Mean ensemble | $\mathbf{1}_m\bar{x}_n^{\mathsf T}\in\mathbb{R}^{m\times N_x}$ | $v1=[v,\ldots,v]$ | $\boldsymbol{x}_n^f\boldsymbol{1}^{\top}$ when needed | da_py stores ensemble members as rows; the thesis and paper-facing notation write ensemble matrices column-wise. |
| Ensemble deviation / anomaly | $A_n=X_n-\bar{X}_n$ | $dV=[v^{(k)}-v]_{k=1}^m$ | $\boldsymbol{V}_n^f=[\boldsymbol{x}_n^{f(1)}-\boldsymbol{x}_n^f,\ldots,\boldsymbol{x}_n^{f(m)}-\boldsymbol{x}_n^f]$, $\boldsymbol{V}_n^a$ | In the paper, $\boldsymbol{V}$ denotes perturbation/anomaly matrices, not the ensemble itself. |
| Sample covariance | $P_n=(m-1)^{-1}A_n^{\mathsf T}A_n$ | $\operatorname{Cov}_m(V)=(m-1)^{-1}dVdV^*$ | $\boldsymbol{C}_n^f=\boldsymbol{V}_n^f(\boldsymbol{V}_n^f)^\top/(m-1)$ | da_py arrays are row-major, so $A_n^{\mathsf T}A_n$ gives the state covariance. Takeda thesis uses $^*$ for Hilbert-space adjoints. |
| Forecast ensemble | $X_n^f$ | $\hat{V}_n=[\hat{v}_n^{(k)}]_{k=1}^m$ | $\boldsymbol{x}_n^{f(k)}=\boldsymbol{\Psi}(\boldsymbol{x}_{n-1}^{a(k)})$ | Takeda thesis uses hat notation for prediction/forecast quantities rather than superscript $f$. |
| Analysis ensemble | $X_n^a$ | $V_n=[v_n^{(k)}]_{k=1}^m$ | $\boldsymbol{x}_n^{a(k)}=\boldsymbol{x}_n^a+\boldsymbol{v}_n^{a(k)}$ | $\boldsymbol{v}_n^{a(k)}$ is the $k$th column of $\boldsymbol{V}_n^a$. |
| Forecast covariance | $P_n^f$ | $\hat{P}_n=\operatorname{Cov}_m(\hat{V}_n)$ | $\boldsymbol{C}_n^f$ | In the thesis, $\operatorname{Cov}_m(\hat{V}_n)=\operatorname{Cov}_m(d\hat{V}_n)$ by Section 2.3.1. |
| Analysis covariance | $P_n^a$ | $\operatorname{Cov}_m(V_n)$ | not explicitly named in the ETKF definition | The analysis spread is represented by $\boldsymbol{V}_n^a$. |
| Innovation | $y_n-H_n\bar{x}_n^f$ | $y_n-H\hat{v}_n$ | $\boldsymbol{y}_n-\boldsymbol{H}\boldsymbol{x}_n^f$ | Code variable: `dy`; do not typeset `dy` as a differential. |
| Kalman gain | $K_n=P_n^fH_n^{\mathsf T}(H_nP_n^fH_n^{\mathsf T}+R_n)^{-1}$ | $K_n=\hat{P}_nH^*(H\hat{P}_nH^*+R)^{-1}$ | $\boldsymbol{K}_n=\boldsymbol{C}_n^f\boldsymbol{H}^\top(\boldsymbol{H}\boldsymbol{C}_n^f\boldsymbol{H}^\top+\boldsymbol{R})^{-1}$ | The paper uses bold $\boldsymbol{K}_n$ and $\top$. |
| Transform matrix | $T$ | $T_n\in\mathbb{R}^{m\times m}$ | $\boldsymbol{T}_n=(\boldsymbol{I}_m+(m-1)^{-1}(\boldsymbol{V}_n^f)^\top\boldsymbol{H}^\top\boldsymbol{R}^{-1}\boldsymbol{H}\boldsymbol{V}_n^f)^{-1/2}$ | $\boldsymbol{T}_n$ is a matrix square root chosen symmetric positive definite in the paper. |
| Inflation parameter | $\alpha$ | $\hat{P}_n^\alpha=\alpha^2\hat{P}_n$, $d\hat{V}_n^\alpha=\alpha d\hat{V}_n$ | $\alpha>1$ | The 2026 paper uses multiplicative covariance inflation in Algorithm 3. |
| Minimum ensemble size | not used as a package symbol | not part of the ETKF notation here | $m^*=N_+ + 1$ | Added because this is central to the 2026 paper's notation and conclusions. |
| Error metric | RMSE in examples | problem-dependent | $\mathrm{SE}_n=\|\boldsymbol{x}_n-\boldsymbol{x}_n^a\|^2$, $\mathrm{RMSE}_n$ | Typeset metric names in roman capitals. |

Typographic conventions:

- Use plain italic scalars and unbold finite-dimensional variables in da_py
  derivations: $x_n$, $y_n$, $H_n$, $R_n$, $P_n$.
- Use `\operatorname{...}` for named operators and functions:
  $\operatorname{Cov}_m$, $\operatorname{rank}$, $\operatorname{Ran}$,
  $\operatorname{Tr}$, $\operatorname{diag}$, and $\operatorname{id}$.
- Do not romanize the `d` in perturbation symbols such as $dV$; in the thesis
  this is part of the variable name, not a differential.
- Use $\mathsf T$ for finite-dimensional transposes in da_py row-major formulas
  and $^*$ only for Hilbert-space adjoints or when quoting the thesis.
- Use calligraphic or blackboard letters only for spaces and probability objects
  when needed. Avoid using the same plain $H$ simultaneously for both the state
  space and the observation operator in a local derivation.
- Keep implementation identifiers, e.g. `X`, `Xf`, `Xa`, `Rinv`, and `dy`, in
  code font and separate from mathematical symbols.

References checked:

- Kota Takeda, *Error Analysis of the Ensemble Square Root Filter for Dissipative
  Dynamical Systems*, doctoral thesis, Department of Mathematics, Kyoto
  University, first version February 1, 2025; modified January 1, 2026.
  Available at <https://kotatakeda.github.io/math/pdf/thesis.pdf>. The URL was
  checked on 2026-07-10. See Sections 2.3.1, 3.1, 4.1--4.3, and Definition 4.13
  for $V$, $v$, $v1$, $dV$, $\operatorname{Cov}_m(V)$, $\varpi_n$,
  $\hat{V}_n$, $\hat{P}_n$, and $T_n$.
- K. Takeda and T. Miyoshi, "Noise-scaled accuracy of the ensemble Kalman filter
  with an instability-based minimum ensemble size," paper notation checked
  against the manuscript notation available to this project. A public NPG
  article page was not found by web search on 2026-07-10, so do not treat the
  volume/page metadata as verified here.
- Existing repository code and examples: `src/etkf.py`, `src/letkf.py`,
  `src/po.py`, `src/var3d.py`, and representative Lorenz-63/Lorenz-96
  notebooks.

## State-space and observation model

All representative examples use the discrete-time state-space model

$$
x_n = M_n(x_{n-1}), \qquad
y_n = H_n x_n + \varepsilon_n, \quad \varepsilon_n \sim N(0, R_n),
$$

where $M_n$ advances the model over one assimilation window (numerical
integration of the underlying ODE/PDE with the fourth-order Runge-Kutta
scheme, `da.scheme.rk4`).

## Models

**Lorenz-63** ($N_x = 3$, chaotic for the standard parameters
$\sigma = 10$, $r = 28$, $b = 8/3$):

$$
\dot{x} = \sigma (y - x), \qquad
\dot{y} = x (r - z) - y, \qquad
\dot{z} = x y - b z.
$$

**Lorenz-96** ($N_x = J$ variables on a ring, forcing $F$; chaotic for
$J = 40$, $F = 8$):

$$
\dot{x}_j = (x_{j+1} - x_{j-2})\, x_{j-1} - x_j + F,
\qquad j = 1, \dots, J \ (\text{indices mod } J).
$$

**2D Navier-Stokes (vorticity form on the torus)** with viscosity $\nu$ and
stationary (Kolmogorov-type) forcing $f$ (the representative example uses no
linear drag):

$$
\partial_t \omega + (u \cdot \nabla)\, \omega
= \nu \Delta \omega + f,
\qquad u = \nabla^{\perp} \Delta^{-1} \omega ,
$$

solved pseudo-spectrally; the DA state is the packed `rfft2` vorticity
spectrum (`NSE2DTorus.to_spectral_state`).

## Filters

**3DVar** uses a fixed background covariance $B$ and gain

$$
K = B H^{\mathsf T} (H B H^{\mathsf T} + R)^{-1}, \qquad
x_n^a = x_n^f + K (y_n - H x_n^f).
$$

**ExKF** propagates the covariance through the linearized forecast map,

$$
P_n^f = F_n P_{n-1}^a F_n^{\mathsf T} + Q_n, \qquad
K_n = P_n^f H^{\mathsf T} (H P_n^f H^{\mathsf T} + R)^{-1},
$$

with $x_n^a = x_n^f + K_n (y_n - H x_n^f)$ and
$P_n^a = (I - K_n H) P_n^f$.

**ETKF** decomposes the forecast ensemble into mean and anomalies,
$X_n^f = \mathbf{1}\bar{x}_n^f + A_n^f$, and forms the analysis in the
$m$-dimensional ensemble space:

$$
\tilde{P} = \big[(m-1) I + (H A^f)^{\mathsf T} R^{-1} (H A^f)\big]^{-1},
$$

$$
\bar{x}^a = \bar{x}^f + A^f \tilde{P} (H A^f)^{\mathsf T} R^{-1}
(y - H \bar{x}^f), \qquad
A^a = A^f \big[(m-1) \tilde{P}\big]^{1/2}.
$$

Multiplicative inflation is applied to the anomalies, $A \to \alpha A$, so
the covariance is inflated by $\alpha^2$.

**LETKF** performs the ETKF update independently for each state component
$j$, using observation-error covariance localized by the Gaspari-Cohn
function $\rho$ with radius $c$:
$R^{-1} \to R^{-1} \circ \rho(d_{j}/c)$, where $d_j$ is the distance
between component $j$ and each observation location.

**Particle filter (bootstrap)** propagates $m$ particles through the model,
updates weights with the Gaussian likelihood

$$
w_n^{(i)} \propto w_{n-1}^{(i)}
\exp\!\Big(-\tfrac{1}{2}\,\|y_n - H x_n^{(i)}\|_{R^{-1}}^2\Big),
$$

and resamples (multinomial / systematic / residual) when the effective
sample size $N_{\mathrm{eff}} = 1 / \sum_i (w^{(i)})^2$ falls below a
threshold. Small additive noise (`add_inflation`) counteracts weight
collapse.

**ETPF** replaces stochastic resampling by a deterministic linear ensemble
transform $X^a = T^{\mathsf T} X^f$, where $T$ is $m$ times the optimal-transport
coupling that maps the weighted forecast ensemble onto the uniform analysis
ensemble with minimal squared-Euclidean cost (solved with the POT package).

## RMSE convention

See `docs/contributing/notebook_spec.md`: analysis RMSE is
$\mathrm{RMSE}_n = \sqrt{\frac{1}{N_x}\sum_i (\hat{x}^a_{n,i} - x_{n,i})^2}$
with $\hat{x}^a_n$ the analysis (ensemble-mean) estimate, compared against
the observation-noise scale
$\sigma_{\mathrm{obs}} = \sqrt{\operatorname{tr}(R)/N_y}$.
