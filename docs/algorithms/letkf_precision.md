# LETKF correlated observation errors

For symmetric positive-definite `R`, including correlated observation
errors, the local precision is

$$
P_i=D_iR^{-1}D_i,\qquad
D_i=\operatorname{diag}(\sqrt{\rho_{ij}}).
$$

This defines the local quadratic form
`(D_i residual).T @ inv(R) @ (D_i residual)`. It is a symmetric
positive-semidefinite congruence of `R^{-1}`. For diagonal `R`, its diagonal
is exactly `rho_ij / R_jj`. With correlated errors, this is an explicitly
chosen precision-weighting model, not the inverse of a covariance submatrix
obtained by marginalizing excluded observations. Nonsymmetric or non-positive-
definite `R` is rejected.

