# PO multiplicative inflation

The non-additive PO update first constructs the inflated state ensemble
`X_inflated = mean(X) + alpha * (X - mean(X))`, then evaluates
`Y_inflated[k] = H(X_inflated[k])`. Both the sample Kalman gain and the
particle corrections use this same pair of ensembles. In particular,
nonlinear H is evaluated after state inflation; multiplying the observation
anomalies alone would define a different approximation.

For linear H, the forecast sample covariance is scaled by `alpha**2`.
With zero-mean observation perturbations whose sample covariance is R and
whose sample cross-covariance with the state anomalies is zero, the analysis
mean and covariance equal the corresponding Kalman formulas. Random finite
ensembles approximate these moment conditions and need not reproduce the
theoretical covariance exactly. For Pf=R=1 and alpha=2 the ideal analysis
variance is 0.8. The linear additive-inflation path is unchanged.
