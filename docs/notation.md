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
| State | $x \in \mathbb{R}^{N_x}$ | $u_n\in\mathcal{H}$ for the true state; $\varpi_n\in\mathcal{H}$ for a KF/3DVar estimate; $v^{(k)}\in\mathcal{H}$ for ensemble members | $\mathbf{x}_n \in \mathbb{R}^{N_x}$ | da_py keeps code-facing symbols plain; the paper-facing notation uses bold finite-dimensional vectors. |
| True state | $x_n$ or `x_true` | $u_n$ | $\mathbf{x}_n$ | Code names such as `x_true` remain backticked implementation identifiers. |
| Observation | $y_n$ or `y_obs` | $Y_n=h(U_n)+\eta_n$ for random variables; $y_n$ for realized data in algorithms | $\mathbf{y}_n=\mathbf{H}\mathbf{x}_n+\eta_n$ | Use lower-case $y_n$ in algorithmic formulas; use upper-case only for random variables when following the thesis. |
| Observation operator | $H_n$ or $H$ | $h$ for a general observation function; $H$ for a linear operator | $\mathbf{H}\in\mathbb{R}^{N_y\times N_x}$ | da_py uses `H` for both ndarray and callable observations; the paper-facing notation uses a matrix observation operator. |
| Observation error covariance | $R_n$ or $R$ | $R$ | $\mathbf{R}\in\mathbb{R}^{N_y\times N_y}$ | The thesis also uses $r^2 I_{\mathcal{H}}$ under full-observation assumptions. |
| Ensemble size | $m$ | $m$ | $m\in\mathbb{N}$, $m^*$ | $m^*$ is the minimum ensemble size in the paper-facing notation. |
| Ensemble member | $x_n^{(k)}$ | $v^{(k)} \in \mathcal{H}$ | $\mathbf{x}_n^{f(k)}$, $\mathbf{x}_n^{a(k)}$ | The paper puts forecast/analysis labels before the parenthesized member index. |
| Ensemble | $X_n=[x_n^{(k)}]_{k=1}^m$ | $\mathbf{V}=[v^{(k)}]_{k=1}^{m}\in\mathcal{H}^{m}$ | $\mathbf{X}\in\mathbb{R}^{N_x\times m}$, $\mathbf{X}_0$ | The thesis uses bold letters for a set of vectors; da_py stores members as rows in implementation arrays. |
| Ensemble mean | $\bar{x}_n=m^{-1}\sum_{k=1}^m x_n^{(k)}$ | $\overline{v}=m^{-1}\sum_{k=1}^{m}v^{(k)}$ for a generic ensemble; $\overline{\widehat{v}}_n$ and $\overline{v}_n$ for forecast and analysis means | $\overline{\mathbf{x}}_n^f=m^{-1}\sum_{k=1}^m\mathbf{x}_n^{f(k)}$; $\overline{\mathbf{x}}_n^a$ is the analysis mean updated by the ETKF formula | da_py uses $\bar{x}_n$ to avoid overloading the state symbol. The 2026 paper explicitly defines the forecast mean and then uses the analysis mean in the update and ensemble reconstruction. |
| Ensemble deviation / anomaly | $A_n=X_n-\mathbf{1}_m\bar{x}_n^{\top}$ | $d\mathbf{V}=[v^{(k)}-\overline{v}]_{k=1}^{m}$ with $\mathbf{V}=\overline{v}\mathbf{1}+d\mathbf{V}$ | $\mathbf{V}_n^f=[\mathbf{x}_n^{f(1)}-\overline{\mathbf{x}}_n^f,\ldots,\mathbf{x}_n^{f(m)}-\overline{\mathbf{x}}_n^f]$, $\mathbf{V}_n^a$ | In the paper, $\mathbf{V}$ denotes perturbation/anomaly matrices, not the ensemble itself. A separate "mean ensemble" row is unnecessary because it is only the mean part of this decomposition. |
| Sample covariance | $P_n=(m-1)^{-1}A_n^{\top}A_n$ | $\mathrm{Cov}_m(\mathbf{V})=(m-1)^{-1}d\mathbf{V}d\mathbf{V}^*$ | $\mathbf{C}_n^f=\mathbf{V}_n^f(\mathbf{V}_n^f)^\top/(m-1)$ | da_py arrays are row-major, so $A_n^{\top}A_n$ gives the state covariance. Takeda thesis uses $^*$ for Hilbert-space adjoints. |
| Forecast ensemble | $X_n^f$ | $\widehat{\mathbf{V}}_n=[\widehat{v}_n^{(k)}]_{k=1}^{m}$ | $\mathbf{x}_n^{f(k)}=\Psi(\mathbf{x}_{n-1}^{a(k)})$ | Takeda thesis uses hat notation for prediction/forecast quantities rather than superscript $f$. |
| Analysis ensemble | $X_n^a$ | $\mathbf{V}_n=[v_n^{(k)}]_{k=1}^{m}$ | $\mathbf{x}_n^{a(k)}=\overline{\mathbf{x}}_n^a+\mathbf{v}_n^{a(k)}$ | $\mathbf{v}_n^{a(k)}$ is the $k$th column of $\mathbf{V}_n^a$. |
| Forecast covariance | $P_n^f$ | $\widehat{P}_n=\mathrm{Cov}_m(d\widehat{\mathbf{V}}_n)$ | $\mathbf{C}_n^f$ | In the thesis, $\mathrm{Cov}_m(\widehat{\mathbf{V}}_n)=\mathrm{Cov}_m(d\widehat{\mathbf{V}}_n)$ by Section 2.3.1. |
| Analysis covariance | $P_n^a$ | $\mathrm{Cov}_m(\mathbf{V}_n)$ | not explicitly named in the ETKF definition | The analysis spread is represented by $\mathbf{V}_n^a$. |
| Innovation | $y_n-H_n\bar{x}_n^f$ | $y_n-H\overline{\widehat{v}}_n$ | $\mathbf{y}_n-\mathbf{H}\overline{\mathbf{x}}_n^f$ | Code variable: `dy`; do not typeset `dy` as a differential. |
| Kalman gain | $K_n=P_n^fH_n^{\top}(H_nP_n^fH_n^{\top}+R_n)^{-1}$ | $K_n=\widehat{P}_nH^*(H\widehat{P}_nH^*+R)^{-1}$ | $\mathbf{K}_n=\mathbf{C}_n^f\mathbf{H}^\top(\mathbf{H}\mathbf{C}_n^f\mathbf{H}^\top+\mathbf{R})^{-1}$ | The paper uses bold $\mathbf{K}_n$ and $\top$. |
| Transform matrix | $T$ | $T_n\in\mathbb{R}^{m\times m}$ | $\mathbf{T}_n=(\mathbf{I}_m+(m-1)^{-1}(\mathbf{V}_n^f)^\top\mathbf{H}^\top\mathbf{R}^{-1}\mathbf{H}\mathbf{V}_n^f)^{-1/2}$ | $\mathbf{T}_n$ is a matrix square root chosen symmetric positive definite in the paper. |
| Inflation parameter | $\alpha$ | $\widehat{P}_n^\alpha=\alpha^2\widehat{P}_n$, $d\widehat{\mathbf{V}}_n^\alpha=\alpha d\widehat{\mathbf{V}}_n$ | $\alpha>1$ | The 2026 paper uses multiplicative covariance inflation in Algorithm 3. |
| Minimum ensemble size | not used as a package symbol | not part of the ETKF notation here | $m^*=N_+ + 1$ | Added because this is central to the 2026 paper's notation and conclusions. |
| Error metric | RMSE in examples | problem-dependent | $\mathrm{SE}_n=\|\mathbf{x}_n-\overline{\mathbf{x}}_n^a\|^2$, $\mathrm{RMSE}_n$ | Typeset metric names in roman capitals. |

References checked:

- Kota Takeda, *Error Analysis of the Ensemble Square Root Filter for Dissipative
  Dynamical Systems*, doctoral thesis, Department of Mathematics, Kyoto
  University, first version February 1, 2025; modified January 1, 2026.
  Available at <https://kotatakeda.github.io/math/pdf/thesis.pdf>. The URL was
  checked on 2026-07-10. See Sections 2.3.1, 3.1, 4.1--4.3, and Definition 4.13
  for $\mathbf{V}$, the ensemble mean
  $\overline{v}=m^{-1}\sum_k v^{(k)}$, $\overline{v}\mathbf{1}$,
  $d\mathbf{V}$, $\mathrm{Cov}_m(\mathbf{V})$, $\varpi_n$,
  $\widehat{\mathbf{V}}_n$, $\widehat{P}_n$, and $T_n$.
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
K = B H^{\top} (H B H^{\top} + R)^{-1}, \qquad
x_n^a = x_n^f + K (y_n - H x_n^f).
$$

**ExKF** propagates the covariance through the linearized forecast map,

$$
P_n^f = F_n P_{n-1}^a F_n^{\top} + Q_n, \qquad
K_n = P_n^f H^{\top} (H P_n^f H^{\top} + R)^{-1},
$$

with $x_n^a = x_n^f + K_n (y_n - H x_n^f)$ and
$P_n^a = (I - K_n H) P_n^f$.

**ETKF** decomposes the forecast ensemble into mean and anomalies,
$X_n^f = \mathbf{1}\bar{x}_n^f + A_n^f$, and forms the analysis in the
$m$-dimensional ensemble space:

$$
\tilde{P} = \big[(m-1) I + (H A^f)^{\top} R^{-1} (H A^f)\big]^{-1},
$$

$$
\bar{x}^a = \bar{x}^f + A^f \tilde{P} (H A^f)^{\top} R^{-1}
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
transform $X^a = T^{\top} X^f$, where $T$ is $m$ times the optimal-transport
coupling that maps the weighted forecast ensemble onto the uniform analysis
ensemble with minimal squared-Euclidean cost (solved with the POT package).

## RMSE convention

See `docs/contributing/notebook_spec.md`: analysis RMSE is
$\mathrm{RMSE}_n = \sqrt{\frac{1}{N_x}\sum_i (\hat{x}^a_{n,i} - x_{n,i})^2}$
with $\hat{x}^a_n$ the analysis (ensemble-mean) estimate, compared against
the observation-noise scale
$\sigma_{\mathrm{obs}} = \sqrt{\mathrm{tr}(R)/N_y}$.
