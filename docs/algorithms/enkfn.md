# EnKF-N: derivation and implementation map

This page documents the implementation in `src/da/enkfn.py`.  The class
`EnKFN` is the **dual, Gaussian-scale-mixture EnKF-N** of Raanes, Bocquet,
and Carrassi (2019), with the practical hyperprior corrections used by
DAPPER.  It is not the direct non-quadratic ETKF-N analysis of Bocquet
(2011, Sect. 3).  The two methods share the finite-ensemble prior, but make
different Gaussian approximations to the posterior.

## 1. Forecast ensemble and ETKF conventions

Let `m` be the number of members.  Although the public ensemble is stored as
`X.shape == (m, Nx)`, the analysis uses

\[
 E=[x_1^f,\ldots,x_m^f]\in\mathbb R^{N_x\times m},\quad
 \bar x^f=E\mathbf1/m,\quad A=E-\bar x^f\mathbf1^T,
 \quad \widehat P^f=AA^T/(m-1).
\]

For the linear observation model used by `EnKFN`,

\[
 Y=HA\in\mathbb R^{N_y\times m},\qquad
 \delta=y-H\bar x^f\in\mathbb R^{N_y}.
\]

`dY`, `dy`, and `R` in `estimate_l1_enkfn_dual` are respectively
\(Y\), \(\delta\), and the positive-definite observation covariance.  The
inflation symbols used by this repository are

\[
 l_1:\hbox{ anomaly factor},\qquad
 \lambda_{\rm cov}=l_1^2:\hbox{ covariance factor},\qquad
 \zeta=(m-1)/l_1^2:\hbox{ ensemble-space prior precision}.
\]

Thus `l1` must not be confused with a paper's covariance-inflation symbol.

## 2. Finite-ensemble hierarchy and the two EnKF-N analyses

Assume the truth and ensemble members are conditionally iid
\(N(b,B)\), while the unknown mean and covariance have the Jeffreys
hyperprior.  Marginalizing \(b\) and \(B\) gives the predictive prior
\(p(x\mid E)\) (Bocquet 2011, Eqs. (7)--(21); Raanes et al. 2019,
Eqs. (7)--(21)).  Raanes et al. reduce it exactly to the scalar Gaussian
scale mixture

\[
 p(x\mid E)=\int_0^\infty
 N(x\mid\bar x^f,\alpha\widehat P^f)\,p(\alpha\mid E)\,d\alpha
 \tag{Raanes et al. (2019), Eq. (10)}.
\]

The effective prior is Student-type, not Gaussian.  Bocquet's original
ETKF-N therefore minimizes a non-quadratic ensemble-space posterior
(Bocquet 2011, Eqs. (35)--(36)), starting at \(w=0\), and obtains the
analysis covariance from the inverse Hessian (Eqs. (37)--(41)).  In
contrast, the present code selects one Gaussian component by a scalar dual
optimization and then applies a standard ETKF transform.  This distinction
is why a separate original ETKF-N implementation can coexist with `EnKFN`.

## 3. State coordinate, dual precision, and scalar objective

Write \(x=\bar x^f+Aw\).  Here \(w\in\mathbb R^m\) is a state coordinate;
\(\zeta>0\) is an auxiliary precision, not a state coordinate.  Raanes et
al. obtain

\[
 \mathcal J(\zeta,w)=\varepsilon_m\zeta-(m+g)\log\zeta
 +\zeta\lVert w\rVert^2+\lVert\delta-Yw\rVert_{R^{-1}}^2,
 \tag{39}
\]

and the stationary relation

\[
 \zeta(w)={m+g\over\varepsilon_m+\lVert w\rVert^2}.
 \tag{38c}
\]

For fixed \(\zeta\), the quadratic problem has

\[
 P_w(\zeta)=(\zeta I+Y^TR^{-1}Y)^{-1},\qquad
 \bar w_a(\zeta)=P_w(\zeta)Y^TR^{-1}\delta,
 \tag{41}
\]

and substitution gives the one-dimensional dual \(D(\zeta)\) in their
Eq. (42).  A stationary dual solution and \(\bar w_a(\zeta)\) form a
stationary point of the effective posterior (Sect. 3.6).  Their Gaussian
posterior approximation is Eqs. (46)--(47), with squared prior inflation
\(\alpha_*=(m-1)/\zeta_*\) (Eq. (48)).

The code whitens with the Cholesky factor \(R=LL^T\), forms

\[
 Z=(L^{-1}Y)^T\in\mathbb R^{m\times N_y},\quad
 Z=U\,\mathrm{diag}(s_i)V^T,\quad
 d_u=V^TL^{-1}\delta,
\]

and uses \(l_1^2=(m-1)/\zeta\).  Woodbury reduction of Eq. (42), removal
of terms independent of \(l_1\), and the DAPPER normalization yield

\[
 J(l_1)=\sum_{i=1}^{r}{d_{u,i}^2\over (l_1s_i)^2+m-1}
       +{e_N\over l_1^2}+c_L\log(l_1^2),\qquad l_1>0. \tag{1}
\]

Only the SVD rank \(r\le\min(m,N_y)\) contributes to the first term;
innovation components orthogonal to `V` contribute only an omitted constant.
`_hyperprior_coeffs` initially sets
\(e_N=(m+1)/m\) and \(c_L=(m+g)/(m-1)\), then applies DAPPER's
`xN` scaling and its mode correction based on
\(i_{KH}=\frac{m-1}{m}\sum_{j=1}^{m}(s_j^2+m-1)^{-1}\), with \(s_j=0\)
for indices beyond the computed SVD rank (zero padding).  Consequently, (1)
is the DAPPER practical dual objective:
with the corrections disabled it is a normalized/reparameterized form of
Raanes et al. Eq. (42), but with the default correction it is not
algebraically identical to the uncorrected paper objective.

## 4. Derivatives and solver policy

With \(q_i=(l_1s_i)^2+m-1\), the source implements term by term

\[
 J'(l_1)=-2l_1\sum_i{s_i^2d_{u,i}^2\over q_i^2}
          -{2e_N\over l_1^3}+{2c_L\over l_1},
\]

\[
 J''(l_1)=8l_1^2\sum_i{s_i^4d_{u,i}^2\over q_i^3}
          +{6e_N\over l_1^4}-{2c_L\over l_1^2}.
\]

The requested default is Newton's method from `initial_l1=1`.  Failure,
non-finite output, or a non-positive root triggers a bracketed Brent root of
\(J'\); failure to bracket triggers bounded minimization on
\([10^{-8},10^8]\).  A requested Brent solve has the same bounded fallback.
The result is positive and finite after the final check, but the objective is
not asserted convex: `converged` reports only the numerical solver's status,
not uniqueness or global optimality.  `method` records the solver actually
used and `requested_method` records the request.

## 5. Mapping into the ETKF transform

`EnKFN._update_T` estimates `l1`, temporarily assigns it to `ETKF.alpha`,
and calls `ETKF._transform_T`.  Therefore

\[
 S={m-1\over l_1^2}I+Y^TR^{-1}Y=\zeta I+Y^TR^{-1}Y,
 \quad P_w=S^{-1},
\]

\[
 w_a=P_wY^TR^{-1}\delta,\qquad
 T=w_a\mathbf1^T+[(m-1)P_w]^{1/2},\qquad
 E^a=\bar x^f\mathbf1^T+AT.
\]

The symmetric eigensquare-root used in `_transform_T` makes the ensemble
mean and sample covariance agree with these equations.  `l1` is the total
cycle inflation; `EnKFN(alpha != 1)` is rejected to prevent double inflation.
The cached Cholesky factor means `R` is assumed constant for the object's
lifetime.

## 6. API, diagnostics, and code map

`estimate_l1_enkfn_dual(dY, dy, R, xN=1, g=0, *, method="newton",
initial_l1=1, xtol=1e-8, max_iter=100)` accepts shapes `(Ny, m)`, `(Ny,)`,
and `(Ny, Ny)` and returns `(l1, info)`.  `EnKFN` additionally accepts the
usual ETKF model, observation operator, storage, model-noise, and RNG
arguments.  Its observation operator must be linear for the scale estimate
to have the derivation above.

| Mathematical/computational role | Source |
| --- | --- |
| corrected \(e_N,c_L\) | `_hyperprior_coeffs` |
| whitening, SVD, (1), derivatives | `_estimate_l1_enkfn_dual_from_cholesky` |
| \(l_1,\lambda_{cov},\zeta\), solver diagnostics | returned `info` |
| per-cycle estimation and temporary total inflation | `EnKFN._update_T` |
| \(S,w_a,T,E^a\) | `ETKF._transform_T` |

Diagnostics contain `l1`, `lambda_cov`, `zeta`, `objective`, `gradient`,
`converged`, `n_iter`, both method fields, `singular_values`, `du`, `eN`, and
`cL`; `effective_alpha` is added by `EnKFN`.

| Concept | Raanes et al. (2019) | DAPPER | `da_py` |
| --- | --- | --- | --- |
| ensemble size | \(N\) | `N` | `m` |
| state coordinate | \(w\) | `w` | implicit ETKF mean weights |
| dual precision | \(\zeta\) | `za` | `zeta` |
| anomaly factor | \(\sqrt{\alpha_*}\) | `l1` | `l1` / temporary `alpha` |
| covariance factor | \(\alpha_*\) | `l1**2` | `lambda_cov` |
| whitened anomalies | \(R^{-1/2}Y\) | `Y`/`V,s` | `Z.T`/`s` |

## 7. Verified deviations

1. `da_py` copies DAPPER's dual scalar objective and hyperprior mode
   correction, but uses SciPy solvers and explicit fallbacks.
2. It estimates the scale once, then reuses the ordinary symmetric ETKF
   covariance; it does not use Bocquet's non-quadratic Hessian covariance or
   the Hessian correction discussed by Raanes et al. after Eq. (49).
3. DAPPER pads the innovation to the full observation dimension; `da_py`
   sums only the SVD-rank coordinates.  The discarded part is independent of
   `l1`, so the optimizer is unchanged.
4. The registry remains the operational source for examples; no metadata
   field is needed for this explanatory page.

## References

- Bocquet, M. (2011), “Ensemble Kalman filtering without the intrinsic need
  for inflation,” *Nonlinear Processes in Geophysics*, 18, 735–750.
  <https://doi.org/10.5194/npg-18-735-2011>
- Raanes, P. N., Bocquet, M., and Carrassi, A. (2019), “Adaptive covariance
  inflation in the ensemble Kalman filter by Gaussian scale mixtures,”
  *Quarterly Journal of the Royal Meteorological Society*, 145, 53–75.
  <https://doi.org/10.1002/qj.3386>
- DAPPER, `dapper.da_methods.ensemble` (`EnKF_N`, `effective_N`, and
  `hyperprior_coeffs`). <https://nansencenter.github.io/DAPPER/reference/da_methods/ensemble/>
- Bishop, C. H., Etherton, B. J., and Majumdar, S. J. (2001), “Adaptive
  Sampling with the Ensemble Transform Kalman Filter. Part I,” *Monthly
  Weather Review*, 129, 420–436. <https://doi.org/10.1175/1520-0493(2001)129%3C0420:ASWTET%3E2.0.CO;2>
- Tippett, M. K. et al. (2003), “Ensemble Square Root Filters,” *Monthly
  Weather Review*, 131, 1485–1490.
  <https://doi.org/10.1175/1520-0493(2003)131%3C1485:ESRF%3E2.0.CO;2>
