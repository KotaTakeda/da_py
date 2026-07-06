"""Extended Kalman filter."""

import numpy as np


def finite_difference_jacobian(func, x, eps=1.0e-5):
    """Central finite-difference Jacobian of a vector-valued map."""
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(func(x), dtype=float)
    jac = np.empty((y0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        jac[:, i] = (np.asarray(func(x + dx)) - np.asarray(func(x - dx))) / (2.0 * eps)
    return jac


class ExKF:
    """Minimal extended Kalman filter.

    The forecast covariance uses a tangent-linear approximation of the
    discrete forecast map. If ``jacobian`` is omitted, a finite-difference
    approximation is used.
    """

    def __init__(self, M, H, R, Q=None, jacobian=None):
        """
        Args:
        - M: callable ``(x, dt) -> x`` for the nonlinear forecast map.
        - H: linear observation matrix.
        - R: observation-error covariance matrix.
        - Q: model-error covariance matrix. Defaults to zero.
        - jacobian: optional callable ``(x, dt) -> F`` for the forecast map.
        """
        self.M = M
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Q = None if Q is None else np.asarray(Q, dtype=float)
        self.jacobian = jacobian
        self.x = []
        self.x_f = []
        self.P_f = []
        self.P_a = []

    def initialize(self, x_0, P_0):
        self.x_a = np.asarray(x_0, dtype=float).copy()
        self.P = np.asarray(P_0, dtype=float).copy()
        self.Nx = self.x_a.size
        if self.Q is None:
            self.Q = np.zeros((self.Nx, self.Nx))

    def forecast(self, dt):
        if self.jacobian is None:
            F = finite_difference_jacobian(lambda x: self.M(x, dt), self.x_a)
        else:
            F = np.asarray(self.jacobian(self.x_a, dt), dtype=float)
        self.x_f_state = np.asarray(self.M(self.x_a, dt), dtype=float)
        self.P = F @ self.P @ F.T + self.Q
        self.x_f.append(self.x_f_state.copy())
        self.P_f.append(self.P.copy())

    def update(self, y_obs):
        y_obs = np.asarray(y_obs, dtype=float)
        innovation = y_obs - self.H @ self.x_f_state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_a = self.x_f_state + K @ innovation
        ident = np.eye(self.Nx)
        self.P = (
            (ident - K @ self.H) @ self.P @ (ident - K @ self.H).T + K @ self.R @ K.T
        )
        self.x.append(self.x_a.copy())
        self.P_a.append(self.P.copy())
