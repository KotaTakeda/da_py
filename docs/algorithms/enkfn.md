# EnKF-N: derivation and implementation map

This page documents both finite-size filters.  `EnKFN` in `src/da/enkfn.py`
is the **dual, Gaussian-scale-mixture EnKF-N** of Raanes, Bocquet, and
Carrassi (2019), with the practical hyperprior corrections used by DAPPER.
`ETKFN2011` in `src/da/etkfn2011.py` is the direct non-quadratic ETKF-N
analysis of Bocquet (2011, Sect. 3).  The methods share the finite-ensemble
prior, but use different posterior approximations.

## 1. Forecast ensemble and ETKF conventions

Let `m` be the number of members.  Although the public ensemble is stored as
`X.shape == (m, Nx)`, the analysis uses

$$
 E=[x_1^f,\ldots,x_m^f]\in\mathbb R^{N_x\times m},\quad
 \bar x^f=E\mathbf1/m,\quad A=E-\bar x^f\mathbf1^T,
 \quad \widehat P^f=AA^T/(m-1).
$$

For the linear observation model used by `EnKFN`,

$$
 Y=HA\in\mathbb R^{N_y\times m},\qquad
 \delta=y-H\bar x^f\in\mathbb R^{N_y}.
$$

`dY`, `dy`, and `R` in `estimate_l1_enkfn_dual` are respectively
$Y$, $\delta$, and the positive-definite observation covariance.  The
inflation symbols used by this repository are

$$
 l_1:\hbox{ anomaly factor},\qquad
 \lambda_{\rm cov}=l_1^2:\hbox{ covariance factor},\qquad
 \zeta=(m-1)/l_1^2:\hbox{ ensemble-space prior precision}.
$$

Thus `l1` must not be confused with a paper's covariance-inflation symbol.

## 2. Finite-ensemble hierarchy and the two EnKF-N analyses

Assume the truth and ensemble members are conditionally iid
$N(b,B)$, while the unknown mean and covariance have the Jeffreys
hyperprior.  Marginalizing $b$ and $B$ gives the predictive prior
$p(x\mid E)$ (Bocquet 2011, Eqs. (7)--(21); Raanes et al. 2019,
Eqs. (7)--(21)).  Raanes et al. reduce it exactly to the scalar Gaussian
scale mixture

$$
 p(x\mid E)=\int_0^\infty
 N(x\mid\bar x^f,\alpha\widehat P^f)\,p(\alpha\mid E)\,d\alpha
 \tag{Raanes et al. (2019), Eq. (10)}.
$$

The effective prior is Student-type, not Gaussian.  Bocquet's original
ETKF-N therefore minimizes a non-quadratic ensemble-space posterior
(Bocquet 2011, Eqs. (35)--(36)), starting at $w=0$, and obtains the
analysis covariance from the inverse Hessian (Eqs. (37)--(41)).  In
contrast, the present code selects one Gaussian component by a scalar dual
optimization and then applies a standard ETKF transform.  This distinction
is why a separate original ETKF-N implementation can coexist with `EnKFN`.

## 3. Paper equations and implementation transformations

### 3.1 Original equations: Raanes et al. (2019)

The equations below retain the paper's symbols and numbering. The source
checked here is [the author manuscript, arXiv:1801.08474v4,
Sects. 3.6--3.7](https://arxiv.org/pdf/1801.08474v4#page=8).
The paper writes $\lVert v\rVert_C^2=v^TC^{-1}v$; this page uses an explicit
inverse in quadratic forms to avoid confusing covariance and precision.
Here $N$ is the ensemble size, $X$ the anomaly matrix,
$\bar\delta=y-H\bar x$, $Y=HX$, and $\varepsilon_N=1+1/N$.

$$
 x=\bar x+Xw,\qquad
 \zeta(w)=\frac{N+g}{\varepsilon_N+\lVert w\rVert^2}.
 \tag{29, 38c}
$$

$$
 \mathcal J(\zeta,w)=\varepsilon_N\zeta-(N+g)\log\zeta
 +\zeta\lVert w\rVert^2
 +(\bar\delta-Yw)^TR^{-1}(\bar\delta-Yw).
 \tag{39b}
$$

$$
 \bar P_w^a(\zeta)=(\zeta I_N+Y^TR^{-1}Y)^{-1},\qquad
 \bar w^a(\zeta)=\bar P_w^a(\zeta)Y^TR^{-1}\bar\delta.
 \tag{41a,b}
$$

Substituting $w=\bar w^a(\zeta)$ gives the original dual objective:

$$
 D(\zeta)=\varepsilon_N\zeta-(N+g)\log\zeta
 +\bar\delta^T\left(R+\frac{YY^T}{\zeta}\right)^{-1}\bar\delta,
 \qquad \zeta>0.
 \tag{42}
$$

The paper uses the scaled SVD and covariance inflation

$$
 U\,\operatorname{diag}(\bar\sigma_i)V^T
       =[(N-1)R]^{-1/2}Y,\qquad
 \alpha_*=(N-1)/\zeta_*.
 \tag{45, 48}
$$

In Sect. 3.5, $g$ is the nullity of $X$ (typically 1 under the paper's
assumptions). The implementation instead exposes `g` as a parameter with
default 0. Algebraic comparisons below hold for the **same chosen $g$**;
the default is not an automatic implementation of the paper's nullity rule.

### 3.2 Change of notation, whitening, and scalar normalization

In this repository $N=m$, $X=A$, $\bar\delta=\delta$, and
$n=m-1$. The implementation's `X` array is the ensemble, not the paper's
anomaly matrix $X$. The optimization variable is changed by

$$
 \alpha=l_1^2=\lambda_{\rm cov},\qquad
 \zeta=\frac{n}{l_1^2},\qquad l_1>0.
$$

With $R=LL^T$, the implementation computes

$$
 \widetilde Y=L^{-1}Y,\quad v=L^{-1}\delta,\quad
 Z=\widetilde Y^T=U_c\operatorname{diag}(s_i)V_c^T,\quad d_u=V_c^Tv.
$$

These are the unscaled singular values: $s_i=\sqrt n\,\bar\sigma_i$.
Cholesky whitening and symmetric whitening differ by an orthogonal factor;
the SVD factors need not be identical to those in Eq. (45).
Here $r=\min(m,N_y)$ is the number of retained thin-SVD coordinates
(including any zero singular values), not necessarily the numerical rank.
Let $v_\perp=(I-V_cV_c^T)v$. Direct diagonalization gives

$$
 \delta^T(R+YY^T/\zeta)^{-1}\delta
 =\sum_{i=1}^{r}\frac{\zeta d_{u,i}^2}{\zeta+s_i^2}
  +\lVert v_\perp\rVert^2
 =n\sum_{i=1}^{r}\frac{d_{u,i}^2}{n+l_1^2s_i^2}
  +\lVert v_\perp\rVert^2.
$$

Thus, before any practical hyperprior correction,

$$
 \frac{D(n/l_1^2)}{n}=J_0(l_1)+C,
$$

$$
 J_0(l_1)=\sum_{i=1}^{r}\frac{d_{u,i}^2}{n+l_1^2s_i^2}
 +\frac{\varepsilon_m}{l_1^2}
 +\frac{m+g}{n}\log(l_1^2),\qquad
 C=\frac{\lVert v_\perp\rVert^2-(m+g)\log n}{n}.
$$

The positive constant scaling $1/n$ and additive constant $C$ preserve
minimizers. This is a reparameterization of an objective, not a change of
probability-density coordinates; no Jacobian term is added.

### 3.3 Practical coefficients used by the implementation

The following formulas describe `_hyperprior_coeffs`, rather than an
identity obtained solely by transforming Eq. (42):

$$
 e_0=\frac{m+1}{m},\quad c_0=\frac{m+g}{n},\quad
 p_0=\frac{e_0}{c_0},\quad
 i_{KH}=\frac{n}{m}\sum_{j=1}^{m}\frac{1}{s_j^2+n},\quad
 h=\sqrt{p_0^{\,i_{KH}}},
$$

$$
 e_N=\frac{e_0}{h}\,x_N,\qquad c_L=c_0h\,x_N.
$$

Singular values are padded with zeros to length $m$ in $i_{KH}$.
`xN` is $x_N$, not an ensemble member. The corrected objective is

$$
 J(l_1)=\sum_{i=1}^{r}\frac{d_{u,i}^2}{n+l_1^2s_i^2}
       +\frac{e_N}{l_1^2}+c_L\log(l_1^2),\qquad l_1>0. \tag{1}
$$

Only when $h=1$ and $x_N=1$ does this reduce to $J_0$ for the same $g$.
Setting `xN=1` alone does **not** disable the mode correction; the current
API has no switch to disable it. These practical coefficients follow the
DAPPER formulation. They must be distinguished from the uncorrected paper
objective and from the purely algebraic transformations in Sect. 3.2.

## 4. Derivatives and Newton-only solver policy

### 4.1 Derivatives in the implementation variable

The following derivatives are derived here from (1); they are not quoted
expanded derivatives from the paper. For $q_i=n+l_1^2s_i^2$,

$$
 J'(l_1)=-2l_1\sum_i\frac{s_i^2d_{u,i}^2}{q_i^2}
          -\frac{2e_N}{l_1^3}+\frac{2c_L}{l_1},
$$

$$
 J''(l_1)=-2\sum_i\frac{s_i^2d_{u,i}^2}{q_i^2}
          +8l_1^2\sum_i\frac{s_i^4d_{u,i}^2}{q_i^3}
          +\frac{6e_N}{l_1^4}-\frac{2c_L}{l_1^2}.
$$

The first term in $J''$ comes from differentiating the prefactor $-2l_1$
in $J'$. It was missing from both the implementation and this page before
[Issue #59](https://github.com/KotaTakeda/da_py/issues/59) was addressed.

For the uncorrected objective, the chain rule also gives

$$
 J_0'(l_1)=-\frac{2}{l_1^3}D'(n/l_1^2),\qquad
 J_0''(l_1)=\frac{4n}{l_1^6}D''(n/l_1^2)
             +\frac{6}{l_1^4}D'(n/l_1^2).
$$

### 4.2 Paper prescription versus implementation policy

Raanes et al., Sect. 3.7, specify Newton's method with initial
$\zeta=N-1$, corresponding to $l_1=1$. They explicitly acknowledge
multiple minima and dependence on the optimizer and initial guess.
Newton iteration is not invariant under nonlinear reparameterization:
even for $J_0$, iterating in $l_1$ need not follow the same path as in
$\zeta$. The paper does not supply the expanded $l_1$ derivatives above
or the SciPy stopping parameters used here.

The implementation solves $J'(l_1)=0$ using

$$
 l_1^{(k+1)}=l_1^{(k)}-\frac{J'(l_1^{(k)})}{J''(l_1^{(k)})}.
$$

Only `method="newton"` is accepted. Defaults are `initial_l1=1`,
`xtol=1e-8`, and `max_iter=100`. There is no Brent root solve, bounded
minimization, or other fallback. A finite positive iterate is returned even
if the solver reports nonconvergence; `converged=False` exposes that status.
A non-finite or non-positive result raises `RuntimeError`. Solver/input
exceptions propagate rather than triggering another method.

The selected stationary point depends on the initial value. `converged`
reports the solver's stopping status, not a proof of stationarity, positive
curvature, uniqueness, or global optimality. No search over all stationary
points is performed. For the Issue #59 example
`dY=[[-7.1e-5, 7.1e-5]]`, `dy=[10]`, `R=[[1]]`, the default solve selects
the local minimum near $l_1=1$ although a point near $74339.7$ has a lower
objective value. This is an explicit local-solution policy, consistent with
the paper's acknowledgement of initial-value dependence, rather than a
guarantee of global minimization. `method` and `requested_method` are both
`"newton"` in returned diagnostics.

## 5. Mapping into the ETKF transform

`EnKFN._update_T` estimates `l1`, temporarily assigns it to `ETKF.alpha`,
and calls `ETKF._transform_T`.  Therefore

$$
 S={m-1\over l_1^2}I+Y^TR^{-1}Y=\zeta I+Y^TR^{-1}Y,
 \quad P_w=S^{-1},
$$

$$
 w_a=P_wY^TR^{-1}\delta,\qquad
 T=w_a\mathbf1^T+[(m-1)P_w]^{1/2},\qquad
 E^a=\bar x^f\mathbf1^T+AT.
$$

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
| corrected $e_N,c_L$ | `_hyperprior_coeffs` |
| whitening, SVD, (1), derivatives | `_estimate_l1_enkfn_dual_from_cholesky` |
| $l_1,\lambda_{cov},\zeta$, solver diagnostics | returned `info` |
| per-cycle estimation and temporary total inflation | `EnKFN._update_T` |
| $S,w_a,T,E^a$ | `ETKF._transform_T` |

Diagnostics contain `l1`, `lambda_cov`, `zeta`, `objective`, `gradient`,
`converged`, `n_iter`, both method fields, `singular_values`, `du`, `eN`, and
`cL`; `effective_alpha` is added by `EnKFN`.

| Concept | Raanes et al. (2019) | DAPPER | `da_py` |
| --- | --- | --- | --- |
| ensemble size | $N$ | `N` | `m` |
| state coordinate | $w$ | `w` | implicit ETKF mean weights |
| dual precision | $\zeta$ | `za` | `zeta` |
| anomaly factor | $\sqrt{\alpha_*}$ | `l1` | `l1` / temporary `alpha` |
| covariance factor | $\alpha_*$ | `l1**2` | `lambda_cov` |
| whitened anomalies | $R^{-1/2}Y$ | `Y`/`V,s` | `Z.T`/`s` |

## 7. Verified deviations

1. `da_py` copies DAPPER's dual scalar objective and hyperprior mode
   correction, but uses SciPy Newton iteration in $l_1$ with no fallback.
   The default `g=0` also differs from the paper's usual nullity $g=1$.
2. It estimates the scale once, then reuses the ordinary symmetric ETKF
   covariance; it does not use Bocquet's non-quadratic Hessian covariance or
   the Hessian correction discussed by Raanes et al. after Eq. (49).
3. DAPPER pads the innovation to the full observation dimension; `da_py`
   sums only the thin-SVD coordinates.  The discarded part is independent of
   `l1`, so the optimizer is unchanged.
4. The registry remains the operational source for examples; no metadata
   field is needed for this explanatory page.

## 8. Original ETKF-N (`ETKFN2011`)

For $x=\bar x^f+Aw$, `ETKFN2011` minimizes Bocquet (2011), Eq. (35),

$$
 J_a(w)=\frac12\lVert\delta-Yw\rVert_{R^{-1}}^2
 +\frac m2\log(1+m^{-1}+w^Tw),
$$

from $w=0$.  Its analytic gradient and Hessian are Eqs. (37)--(39):

$$
 \nabla J_a=-Y^TR^{-1}(\delta-Yw)+\frac{m}{c}w,
\qquad
 \nabla^2J_a=Y^TR^{-1}Y+\frac{m}{c^2}(cI-2ww^T),
\quad c=1+m^{-1}+w^Tw.
$$

Optimization is performed in an orthonormal basis of
$\{w:\mathbf1^Tw=0\}$, which fixes the redundant ensemble gauge exactly.
At the optimum, the analysis transform is constructed from the positive
definite reduced Hessian as prescribed by Eqs. (40)--(42); it does not call
`ETKF._transform_T`.  The class inherits only forecast and storage
infrastructure from `ETKF`, requires a linear observation matrix, and rejects
additional fixed inflation.  `analysis_diagnostics` records the objective,
reduced gradient norm, iterations, convergence, optimizer message, weights,
gauge residual, and reduced-Hessian eigenvalues.
The diagnostics also include the state-space `analysis_covariance`, which is
the inverse-Hessian covariance represented by the returned ensemble.

## References

- Bocquet, M. (2011), “Ensemble Kalman filtering without the intrinsic need
  for inflation,” *Nonlinear Processes in Geophysics*, 18, 735–750.
  <https://doi.org/10.5194/npg-18-735-2011>
- Raanes, P. N., Bocquet, M., and Carrassi, A. (2019), “Adaptive covariance
  inflation in the ensemble Kalman filter by Gaussian scale mixtures,”
  *Quarterly Journal of the Royal Meteorological Society*, 145, 53–75.
  <https://doi.org/10.1002/qj.3386>. Equations in Sect. 3 above were checked
  against the [author manuscript, v4](https://arxiv.org/pdf/1801.08474v4),
  Sects. 3.5--3.7; section numbering refers to that version.
- DAPPER, `dapper.da_methods.ensemble` (`EnKF_N`, `effective_N`, and
  `hyperprior_coeffs`). <https://nansencenter.github.io/DAPPER/reference/da_methods/ensemble/>
- Bishop, C. H., Etherton, B. J., and Majumdar, S. J. (2001), “Adaptive
  Sampling with the Ensemble Transform Kalman Filter. Part I,” *Monthly
  Weather Review*, 129, 420–436. <https://doi.org/10.1175/1520-0493(2001)129%3C0420:ASWTET%3E2.0.CO;2>
- Tippett, M. K. et al. (2003), “Ensemble Square Root Filters,” *Monthly
  Weather Review*, 131, 1485–1490.
  <https://doi.org/10.1175/1520-0493(2003)131%3C1485:ESRF%3E2.0.CO;2>
