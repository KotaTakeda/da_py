# Ensemble observation innovation

ETKF, LETKF and EnKFN form the predicted observation ensemble
`Y[:,k] = H(X[:,k])`, its mean `ybar = mean(Y, axis=1)`, anomalies
`dY = Y - ybar[:,None]`, and innovation `dy = y_obs - ybar`.
Thus the ensemble regression uses the same centering for covariance and
mean updates. EnKFN uses this innovation for adaptive inflation as well.
If `y_obs == ybar`, the analysis mean equals the forecast ensemble mean
within floating-point error.

For linear H, `mean(H(X)) == H(mean(X))`, so the existing linear-observation
formulas are unchanged. For nonlinear H these differ in general. Callable
operators implement an ensemble regression approximation; exact linear
Gaussian Kalman formulas and the linear-observation EnKF-N derivation do not
thereby become exact nonlinear posterior formulas. LETKF additionally needs
physically meaningful observation positions for localization.
