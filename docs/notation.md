# Notation

This page defines the compact notation used by da_py examples. The notation is
package-oriented and may differ from the author's thesis notation.

| Symbol | Meaning |
| --- | --- |
| `x_k` | true state at assimilation time `k` |
| `x_k^f` | forecast state before assimilating `y_k` |
| `x_k^a` | analysis state after assimilating `y_k` |
| `M_k` | nonlinear forecast map from `k - 1` to `k` |
| `F_k` | tangent-linear or finite-difference Jacobian of `M_k` |
| `y_k` | observation vector |
| `H_k` | linear observation operator or its local linear representation |
| `R_k` | observation-error covariance |
| `B_k` | background-error covariance used by 3DVar |
| `P_k^f` | forecast-error covariance |
| `P_k^a` | analysis-error covariance |
| `X_k` | ensemble matrix whose rows are ensemble members |
| `\bar{x}_k` | ensemble mean |
| `A_k` | ensemble perturbation matrix |
| `m` | ensemble size or number of particles |
| `\alpha` | anomaly inflation factor; `A -> alpha A` and `P -> alpha^2 P` |
| `\rho` | localization function or localization radius context |

## Core Equations

The observation model is

```text
y_k = H_k x_k + epsilon_k,  epsilon_k ~ N(0, R_k).
```

3DVar in the linear-Gaussian setting uses the fixed background covariance
`B_k` and gain

```text
K_k = B_k H_k^T (H_k B_k H_k^T + R_k)^{-1}.
```

ExKF propagates the covariance through a linearized forecast map:

```text
P_k^f = F_k P_{k-1}^a F_k^T + Q_k.
```

ETKF and LETKF represent forecast uncertainty by ensemble perturbations. In
da_py, multiplicative inflation is applied to anomalies, so the covariance
inflation factor is `alpha^2`.
