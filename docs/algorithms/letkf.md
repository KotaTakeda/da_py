# LETKF observation geometry and local precision

`LETKF` uses a one-dimensional periodic state grid with positions
`0, ..., Nx-1` and period `Nx`. Observation positions are in the same grid
units and may be fractional. Supply a length-`Ny` `observation_positions`
vector, ordered like the rows of `H`, the entries of `y`, and both axes of `R`.
For a one-hot selection matrix the positions are inferred automatically.
Other matrices and callable operators require explicit, physically meaningful
local observation positions; arbitrary nonlocal operators have no inferred
localization geometry. These positions do not imply a general treatment of
nonlocal observations.

```python
import numpy as np
from da.l96 import lorenz96, two_thirds_observation
from da.letkf import LETKF
from da.scheme import rk4

H, positions = two_thirds_observation(60)  # H.shape == (40, 60)
def forecast(x, dt):
    return rk4(lorenz96, 0.0, x, (8.0,), dt)

filt = LETKF(forecast, H, np.eye(40), c=4.0,
             observation_positions=positions)
filt.initialize(np.random.default_rng(0).normal(size=(12, 60)))
filt.update(np.zeros(40))
assert filt.X.shape == (12, 60)
```

At state position `i`, the periodic distance is
`min(abs(i-j) % Nx, Nx - abs(i-j) % Nx)`. Gaspari–Cohn weights have support
up to `2*c`; the weight at distance `c` is `5/24`. `c` must be positive;
`c=np.inf` sets all weights to one and recovers the global ETKF update.

