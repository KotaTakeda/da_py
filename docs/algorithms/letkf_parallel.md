# LETKF multiprocessing

`multi_process=True` requires a platform supporting `fork` and picklable
filter attributes (a module-level model callable and a matrix `H`).
Worker outputs have shape `(1, m)`; stacking them along the state axis
and adding the forecast mean produces an analysis of shape `(m, Nx)`.
The parallel and serial analyses agree within floating-point rounding error.
Subsequent forecast/update calls follow the same lifecycle on both paths.
