import numpy as np
from numpy.linalg import inv


# ============================
#  3DVar
# ============================
class Var3D:
    """Fixed-B analysis with composable forecasts between observations.

    Every forecast starts at the latest forecast or analysis state. B and the
    gain remain fixed until reinitialization; initialize resets the history.
    """
    def __init__(self, M, H, R):
        """
        Args:
        - M (x, dt) -> x: model dynamics
        - H (Ny, Nx):observation operator
        - R (Ny, Ny): covariance of observation noise

        """
        self.M = M
        self.H = H
        self.R = R
        self.x = []

    def initialize(self, x_0, B):
        """
        Args:
        - x_0: initial state
        - B: Background error covariance matrix
        """
        self.x_a = np.asarray(x_0, dtype=float).copy()
        self._current_state = self.x_a.copy()
        self.x_f = self.x_a.copy()
        self.x = []
        self.B = B
        self._calc_kalman_gain()

    # Forecast step
    def forecast(self, dt):
        self.x_f = np.asarray(self.M(self._current_state, dt), dtype=float)
        self._current_state = self.x_f.copy()

    # Analysis step
    def update(self, y_obs):
        self.x_a = self.x_f + self.K @ (y_obs - self.H @ self.x_f)
        self._current_state = self.x_a.copy()

        # store
        self.x.append(self.x_a.copy())

    # Calculate Kalman gain
    def _calc_kalman_gain(self):
        H = self.H
        B = self.B
        R = self.R
        self.K = B @ H.T @ inv(H @ B @ H.T + R)
