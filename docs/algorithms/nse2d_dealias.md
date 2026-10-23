# NSE2D quadratic dealiasing

`dealias=True` (or `"2/3"`) uses the strict integer-mode cutoff
`3*abs(kx) < nx` and `3*abs(ky) < ny`. The boundary modes at `N/3` are
excluded when N is divisible by three: a pair of such modes can otherwise
alias onto `-N/3` in the retained band.

The nonlinear operator is `-P[(u(P omega) dot grad)(P omega)]`, where P is
this Fourier projection. Velocity and vorticity gradients use the same
truncated input, and the product is projected again. Output-only truncation
cannot remove high-high interactions that have already aliased into low
modes. Diffusion, drag and forcing continue to act on the full input state;
this option is not an unconditional projection of the public state.

`dealias="pad"` computes the product using 3/2 padding and retains a wider
output band. The two methods agree on the common retained output modes
when given the same 2/3-band-limited input, up to rounding error. They need
not agree for input fields containing modes discarded by the 2/3 rule.
