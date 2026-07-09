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

| Concept | da_py / this note | Takeda thesis | EnKF-N papers | TeX and typography remarks |
| --- | --- | --- | --- | --- |
| State | $x \in \mathbb{R}^{N_x}$ | $v \in H$ | commonly $x$ | Use plain italic state variables. Use $\mathbb{R}^{N_x}$ for finite-dimensional examples; reserve Hilbert-space $H$ for reference settings to avoid collision with the observation operator. |
| True state | $x_n$ or `x_true` | problem-dependent exact state | commonly truth or $x^t$ | Use time index $n$ as a subscript. Put code names such as `x_true` in backticks, not math mode. |
| Observation | $y_n$ or `y_obs` | observation-space element | commonly $y$ | Use plain italic $y$. Put implementation names such as `y_obs` in backticks. |
| Observation operator | $H_n$ or $H$ | observation map/operator | $H$ or $\mathcal{H}$ | Use plain $H$ for the code-facing operator. Use $\mathcal{H}$ only when a source distinguishes nonlinear observation maps from linear matrices. |
| Observation error covariance | $R_n$ or $R$ | observation-noise covariance | $R$ | Use plain italic capital $R$. Code names such as `Rinv` remain backticked implementation identifiers. |
| Ensemble size | $m$ | $m$ | often $N$ | Use italic scalar $m$ for da_py to avoid conflict with state dimension $N_x$ and observation dimension $N_y$. |
| Ensemble member | $x_n^{(k)}$ | $v^{(k)} \in H$ | often $x_k$ or $x^{(k)}$ | Use ensemble index $k$ as a parenthesized superscript, and time index $n$ as a subscript. |
| Ensemble | $X_n=[x_n^{(k)}]_{k=1}^m$ | $V=[v^{(k)}]_{k=1}^m \in H^m$ | often $E$ or $X$ | Use plain italic capital $X$, not bold $\mathbf{X}$. Mathematical derivations may use column-wise members; code stores members as rows. |
| Ensemble mean | $\bar{x}_n$ | $v=m^{-1}\sum_{k=1}^m v^{(k)}$ | often $\bar{x}$ | Use a bar over the state symbol. Avoid using bare $x$ for the mean in new docs. |
| Mean ensemble | $\bar{X}_n=[\bar{x}_n,\ldots,\bar{x}_n]$ | $v1=[v,\ldots,v]$ | often implicit | Use $\bar{X}$ for the repeated-mean ensemble. Do not introduce bold-one notation unless a derivation requires it. |
| Ensemble deviation / anomaly | $A_n=X_n-\bar{X}_n$ | $dV=[v^{(k)}-v]_{k=1}^m$ | often anomaly matrix, sometimes written with $X$ | Use $A$ for the anomaly matrix in da_py. Use $dV$ only when reproducing thesis notation; the `d` in $dV$ is not a differential. |
| Sample covariance | $P_n=(m-1)^{-1}A_nA_n^{\mathsf T}$ | $\operatorname{Cov}_m(V)=(m-1)^{-1}dVdV^*$ | commonly $(N-1)^{-1}XX^{\mathsf T}$ for anomaly matrix $X$ | Typeset named operators in roman form, e.g. $\operatorname{Cov}$. Use $^{\mathsf T}$ for finite-dimensional transpose and $^*$ only for Hilbert-space adjoints. |
| Forecast ensemble | $X_n^f$ | forecast version of $V$ | often $E^f$ or $X^f$ | Use forecast label $f$ as a superscript. Code variable: `Xf`. |
| Analysis ensemble | $X_n^a$ | analysis version of $V$ | often $E^a$ or $X^a$ | Use analysis label $a$ as a superscript. Code variable: `Xa`. |
| Forecast covariance | $P_n^f$ | $\operatorname{Cov}_m(V^f)$ | commonly $P^f$ | Use $P$ for covariance and $^{\mathsf T}$ for finite-dimensional formulas. ETKF represents this implicitly by anomalies. |
| Analysis covariance | $P_n^a$ | $\operatorname{Cov}_m(V^a)$ | commonly $P^a$ | Use $P_n^a$ for the finite-dimensional analysis covariance; it is represented by the transformed ensemble in ETKF. |
| Innovation | $y_n-H_n\bar{x}_n^f$ | observation residual | commonly $y-Hx^f$ | Use a minus sign between mathematical symbols. Code variable: `dy`; do not typeset `dy` as a differential. |
| Kalman gain | $K_n=P_n^fH_n^{\mathsf T}(H_nP_n^fH_n^{\mathsf T}+R_n)^{-1}$ | standard Kalman-gain notation | standard Kalman-gain notation | Use plain italic capital $K$ and finite-dimensional transpose $^{\mathsf T}$. ETKF does not need to form $K$ explicitly. |
| Transform matrix | $T$ | ETKF transform matrix | ensemble transform or weight transform | Use plain italic capital $T$ for the transform matrix. This is distinct from transpose notation $^{\mathsf T}$. |
| Inflation parameter | $\alpha$ | multiplicative inflation parameter | often $\lambda$ or a prior-related scale parameter | Use Greek italic $\alpha$ for da_py. Conventions differ: some inflate anomalies, others inflate covariance or estimate inflation. |

The EnKF-N literature also introduces prior parameters and objective functions
for adaptive inflation. These are algorithmic quantities rather than adopted
da_py notation, and should be introduced only in a dedicated EnKF-N derivation.

References checked:

- Takeda thesis, Section 2.3.1 and ETKF definitions: $V$, $v$, $v1$, $dV$,
  and $\operatorname{Cov}_m(V)$.
- M. Bocquet, "Ensemble Kalman filtering without the intrinsic need for
  inflation," Nonlinear Processes in Geophysics, 18, 735--750, 2011.
- M. Bocquet and P. Sakov, "Combining inflation-free and iterative ensemble
  Kalman filters for strongly nonlinear systems," Nonlinear Processes in
  Geophysics, 19, 383--399, 2012.
- P. N. Raanes, M. Bocquet, and A. Carrassi, "Adaptive covariance inflation in
  the ensemble Kalman filter by Gaussian scale mixtures," Quarterly Journal of
  the Royal Meteorological Society, 145, 53--75, 2019.
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
