# RNG Policy

## Policy

All randomness in new `da_py` APIs is driven by an **explicit
`numpy.random.Generator` passed as an argument** (create one with
`numpy.random.default_rng(seed)`). Library code must not create its own
generator internally or touch the global `numpy.random` state; the caller owns
seeding, so a single seed at the top of a script reproduces the entire run.

```python
rng = np.random.default_rng(seed)
X0 = attractor_ensemble(step, rng, x0, dt, m)
filt = ETKF(model_step, H, R, Q=Q, rng=rng)  # model noise driven by rng
```

## Survey (issue #50)

Status of RNG usage across the project as of the introduction of
model noise (`da.noise`):

| Area | RNG style |
| --- | --- |
| `examples/scripts/*` (incl. `_common.py`) | explicit `default_rng(seed)` passed as argument |
| `examples/notebooks/*` | explicit `default_rng(seed)` |
| `tests/*` (except PO tests) | explicit `default_rng(seed)` |
| `da.viz` helpers | explicit `default_rng(seed)` |
| `ETKF`/`EnKFN`/`LETKF` model noise `Q` (new) | explicit `Generator` via the `rng` constructor argument |
| `da.pf.ParticleFilter` | **legacy**: global `np.random.normal/choice/rand` (jitter, resampling) |
| `da.po.PO` | **legacy**: global `np.random.multivariate_normal` (observation perturbations); tests must use `np.random.seed()` |

The explicit-Generator convention is coherent everywhere except inside the two
stochastic filters `PO` and `ParticleFilter`, which predate it.

## Migration direction

`PO` and `ParticleFilter` should eventually accept an explicit `rng`
(constructor argument defaulting to `None` → global state for backward
compatibility). This is deliberately **not** bundled with feature work: any
change to how these filters draw numbers alters the results of existing
seeded runs (see the notes in `da/po.py`), so it should land as a dedicated,
clearly-flagged change. Until then, seed the global RNG (`np.random.seed`)
when reproducibility of `PO`/`ParticleFilter` runs is required, and use
explicit generators everywhere else.
