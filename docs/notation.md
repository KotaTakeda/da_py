# Notation

This page defines the notation used by da_py docs and the representative
tutorial notebooks. The notation is package-oriented and may differ from the
author's thesis notation.

| Symbol | Meaning |
| --- | --- |
| $x_k \in \mathbb{R}^{N_x}$ | true state at assimilation time $k$ |
| $x_k^f$ | forecast state before assimilating $y_k$ |
| $x_k^a$ | analysis state after assimilating $y_k$ |
| $M_k$ | nonlinear forecast map from $k-1$ to $k$ |
| $F_k$ | tangent-linear or finite-difference Jacobian of $M_k$ |
| $y_k \in \mathbb{R}^{N_y}$ | observation vector |
| $H_k$ | linear observation operator or its local linear representation |
| $R_k$ | observation-error covariance |
| $B$ | background-error covariance used by 3DVar |
| $P_k^f$, $P_k^a$ | forecast / analysis error covariance |
| $X_k \in \mathbb{R}^{m \times N_x}$ | ensemble matrix whose rows are members |
| $\bar{x}_k$ | ensemble mean |
| $A_k$ | ensemble perturbation (anomaly) matrix |
| $m$ | ensemble size or number of particles |
| $w^{(i)}$ | particle weight of member $i$ |
| $\alpha$ | anomaly inflation factor; $A \to \alpha A$ so $P \to \alpha^2 P$ |
| $\rho$, $c$ | localization function / localization radius |

## State-space and observation model

All representative examples use the discrete-time state-space model

$$
x_k = M_k(x_{k-1}), \qquad
y_k = H_k x_k + \varepsilon_k, \quad \varepsilon_k \sim N(0, R_k),
$$

where $M_k$ advances the model over one assimilation window (numerical
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

**2D Navier-Stokes (vorticity form on the torus)** with viscosity $\nu$,
linear drag $\gamma$, and stationary (Kolmogorov-type) forcing $f$:

$$
\partial_t \omega + (u \cdot \nabla)\, \omega
= \nu \Delta \omega - \gamma \omega + f,
\qquad u = \nabla^{\perp} \Delta^{-1} \omega ,
$$

solved pseudo-spectrally; the DA state is the packed `rfft2` vorticity
spectrum (`NSE2DTorus.to_spectral_state`).

## Filters

**3DVar** uses a fixed background covariance $B$ and gain

$$
K = B H^{\mathsf T} (H B H^{\mathsf T} + R)^{-1}, \qquad
x_k^a = x_k^f + K (y_k - H x_k^f).
$$

**ExKF** propagates the covariance through the linearized forecast map,

$$
P_k^f = F_k P_{k-1}^a F_k^{\mathsf T} + Q_k, \qquad
K_k = P_k^f H^{\mathsf T} (H P_k^f H^{\mathsf T} + R)^{-1},
$$

with $x_k^a = x_k^f + K_k (y_k - H x_k^f)$ and
$P_k^a = (I - K_k H) P_k^f$.

**ETKF** decomposes the forecast ensemble into mean and anomalies,
$X_k^f = \mathbf{1}\bar{x}_k^f + A_k^f$, and forms the analysis in the
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
w_k^{(i)} \propto w_{k-1}^{(i)}
\exp\!\Big(-\tfrac{1}{2}\,\|y_k - H x_k^{(i)}\|_{R^{-1}}^2\Big),
$$

and resamples (multinomial / systematic / residual) when the effective
sample size $N_{\mathrm{eff}} = 1 / \sum_i (w^{(i)})^2$ falls below a
threshold. Small additive noise (`add_inflation`) counteracts weight
collapse.

**ETPF** replaces stochastic resampling by a deterministic linear ensemble
transform $X^a = T^{\mathsf T} X^f$, where $T$ is $m$ times the optimal
transport coupling between the weighted forecast ensemble and the uniform
analysis ensemble (solved with the POT package).

## RMSE convention

See `docs/notebook_spec.md`: analysis RMSE is
$\mathrm{RMSE}_k = \sqrt{\frac{1}{N_x}\sum_i (\hat{x}^a_{k,i} - x_{k,i})^2}$
with $\hat{x}^a_k$ the analysis (ensemble-mean) estimate, compared against
the observation-noise scale
$\sigma_{\mathrm{obs}} = \sqrt{\operatorname{tr}(R)/N_y}$.
