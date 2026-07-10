# Gaussian Model Noise (Additive Stochastic Inflation)

`da.noise` implements the stochastic state-space forecast model

$$
x_{n+1}^{f,(k)} = M\!\left(x_n^{a,(k)}\right) + \eta_{n+1}^{(k)},
\qquad
\eta_{n+1}^{(k)} \overset{\mathrm{i.i.d.}}{\sim} N(0, Q),
$$

drawing one independent perturbation per ensemble member per assimilation
cycle. The perturbation is added **after** deterministic model propagation and
**before** the analysis step; the ETKF analysis transform itself stays
deterministic.

## Where the noise is applied

The filters in this package advance sub-steps with `forecast(dt)` — typically
several times per assimilation cycle — so only the driver loop knows where a
cycle ends. Model noise is therefore applied in the driver loop, not inside
the filter classes:

```python
import numpy as np
from da import ETKF, GaussianModelNoise

rng = np.random.default_rng(7)
noise = GaussianModelNoise(Q)      # validates Q once, caches the factorization

filt = ETKF(model_step, H, R, alpha=1.02)
filt.initialize(X0)
for k in range(1, cycles + 1):
    for _ in range(n_obs):
        filt.forecast(dt)              # deterministic propagation
    filt.X += noise.sample(rng, filt.m)  # one draw per member, this cycle
    filt.update(y_obs[k])              # deterministic ETKF analysis
```

With no noise line (or a zero `Q`), the loop reproduces the deterministic
filter exactly. Because the sampler is independent of the filter class, the
same pattern works for `ETKF`, `EnKFN`, `LETKF`, and `PO`.

`Q` may be a dense symmetric positive-semidefinite `(Nx, Nx)` matrix —
rank-deficient covariances such as $\sigma^2 P$ with $P$ an orthogonal
projection are supported via an eigendecomposition — or a 1-D vector of
per-component variances for diagonal noise. For fully custom perturbations,
skip the wrapper and add your own samples to `filt.X` directly.

## Not to be confused with

| Mechanism | Where | Nature |
| --- | --- | --- |
| **Gaussian model noise** (`da.noise`, this page) | forecast, once per cycle | stochastic, additive: $x^f \mathrel{+}= \eta$, $\eta \sim N(0, Q)$ |
| **Multiplicative anomaly inflation** `alpha` (`ETKF`, `LETKF`; adaptive in `EnKFN`) | analysis | deterministic rescaling: $A \to \alpha A$, so $P^f \to \alpha^2 P^f$ |
| **Additive covariance regularization** `PO(additive_inflation=True)` | analysis | deterministic: $P^f \to P^f + \alpha I$; no sampling despite the name |
| **Observation perturbations** (`PO`) | analysis | stochastic replicates $y + \varepsilon^{(k)}$, $\varepsilon^{(k)} \sim N(0, R)$ — part of the stochastic-EnKF analysis, not model error |
| **`ParticleFilter(add_inflation=sigma)`** | forecast, every `forecast(dt)` sub-step | legacy stochastic jitter $N(0, \sigma^2 I)$ from the global RNG; equivalent to isotropic model noise but per sub-step, not per cycle |
| **`ExKF(Q=...)`** | covariance propagation | the same model-error covariance $Q$ used deterministically: $P^f = F P^a F^{\mathsf T} + Q$; nothing is sampled |

## RNG

`GaussianModelNoise.sample(rng, size)` takes an explicit
`numpy.random.Generator`, following the project RNG policy
(`docs/rng_policy.md`). A fixed seed makes the full filtering run
reproducible.
