from numpy.linalg import inv


# ============================
#  3DVar
# ============================
class Var3D:
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
        self.x_a = x_0
        self.B = B
        self._calc_kalman_gain()

    # Forecast step
    def forecast(self, dt):
        self.x_f = self.M(self.x_a, dt)

    # Analysis step
    def update(self, y_obs):
        self.x_a = self.x_f + self.K @ (y_obs - self.H @ self.x_f)

        # store
        self.x.append(self.x_a)

    # Calculate Kalman gain
    def _calc_kalman_gain(self):
        H = self.H
        B = self.B
        R = self.R
        self.K = B @ H.T @ inv(H @ B @ H.T + R)
