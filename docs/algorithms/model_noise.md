# Gaussian Model Noise (Additive Stochastic Inflation)

The ensemble filters (`ETKF`, `EnKFN`, `LETKF`) accept a model-noise
covariance `Q`, turning the deterministic forecast step into the stochastic
forecast model

$$
x^{(k)} \leftarrow M\!\left(x^{(k)}, \delta t\right) + \eta^{(k)},
\qquad
\eta^{(k)} \overset{\mathrm{i.i.d.}}{\sim} N(0, Q),
$$

with one independent perturbation per ensemble member at **every
`forecast(dt)` step**. The analysis transform itself stays deterministic.

## Usage

```python
import numpy as np
from da import ETKF

filt = ETKF(model_step, H, R, alpha=1.02,
            Q=Q, rng=np.random.default_rng(7))
filt.initialize(X0)
for k in range(1, cycles + 1):
    for _ in range(n_obs):
        filt.forecast(dt)   # stochastic propagation: M(x, dt) + eta
    filt.update(y_obs[k])   # deterministic analysis
```

The driver loop is unchanged from the deterministic case. With `Q=None`
(default) or a zero `Q`, the filter reproduces the deterministic run exactly.
`rng` must be an explicit `numpy.random.Generator` whenever `Q` is given, so
one seed reproduces the whole run. The noise is added before the forecast
diagnostics are recorded, so `x_f` (and `Xf` under `store_ensemble=True`)
describe the same perturbed ensemble the analysis consumes.

## Timing semantics: per forecast step

`Q` acts per `forecast(dt)` call — the same timing as `ExKF(Q=...)`, which
adds the identical `Q` to the covariance propagation at every forecast step,
and as `ParticleFilter(add_inflation=...)`. When the assimilation window
spans $n_{\mathrm{obs}}$ steps, the noise accumulated over one cycle has
covariance of roughly $n_{\mathrm{obs}}\, Q$ (exactly that for static
dynamics). To match a specification stated per assimilation cycle,
use $Q_{\mathrm{step}} = Q_{\mathrm{cycle}} / n_{\mathrm{obs}}$.

## Covariance forms and custom noise

`Q` may be a dense symmetric positive-semidefinite `(Nx, Nx)` matrix —
rank-deficient covariances such as $\sigma^2 P$ with $P$ an orthogonal
projection are supported via an eigendecomposition, factorized once and
reused — or a 1-D vector of per-component variances for diagonal noise.
Shapes, symmetry, and positive-semidefiniteness are validated with
informative errors (the state dimension is checked at `initialize`).

For fully custom perturbations (e.g. a user-supplied sampler), no wrapper is
needed: leave `Q=None` and add your own samples to `filt.X` in the driver
loop between the last `forecast(dt)` and `update()` — note that the recorded
forecast diagnostics then describe the unperturbed propagation.

## Not to be confused with

| Mechanism | Where | Nature |
| --- | --- | --- |
| **Gaussian model noise** `Q` (`ETKF`/`EnKFN`/`LETKF`, this page) | forecast, every `forecast(dt)` step | stochastic, additive: $x \mathrel{+}= \eta$, $\eta \sim N(0, Q)$ per member |
| **Multiplicative anomaly inflation** `alpha` (`ETKF`, `LETKF`; adaptive in `EnKFN`) | analysis | deterministic rescaling: $A \to \alpha A$, so $P^f \to \alpha^2 P^f$ |
| **Additive covariance regularization** `PO(additive_inflation=True)` | analysis | deterministic: $P^f \to P^f + \alpha I$; no sampling despite the name |
| **Observation perturbations** (`PO`) | analysis | stochastic replicates $y + \varepsilon^{(k)}$, $\varepsilon^{(k)} \sim N(0, R)$ — part of the stochastic-EnKF analysis, not model error |
| **`ParticleFilter(add_inflation=sigma)`** | forecast, every `forecast(dt)` step | legacy isotropic variant of the same mechanism: $N(0, \sigma^2 I)$ jitter drawn from the global RNG |
| **`ExKF(Q=...)`** | covariance propagation, every `forecast(dt)` step | the same model-error covariance $Q$ with the same timing, used deterministically: $P^f = F P^a F^{\mathsf T} + Q$; nothing is sampled |

## RNG

The filters take an explicit `numpy.random.Generator` via the `rng`
constructor argument whenever `Q` is given, following the project RNG policy
([RNG policy](../reference/rng_policy.md)). A fixed seed makes the full filtering run
reproducible.
