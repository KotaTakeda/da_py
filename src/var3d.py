from numpy.linalg import inv


# ============================
#  3DVar
# ============================
class Var3D:
    def __init__(self, M, H, R, x_0, B, cut_obs_size=0, obs_type=1):
        self.M = M
        self.H = H
        self.R = R
        self.B = B
        self.x_a = x_0
        self.x = []
        self.cut_obs_size = cut_obs_size
        self.obs_type = obs_type
        self._calc_Kalman_gain()

    # 予報/時間発展
    def forecast(self, dt):
        # 予報
        self.x_f = self.M(self.x_a, dt)  # 保存しておく

        # if log:
        #     self.x.append(self.x_f)

    # 更新/解析
    def update(self, y_obs):
        # x 更新
        self.x_a = self.x_f + self.K @ (y_obs - self.H @ self.x_f)

        # 更新した値を保存
        self.x.append(self.x_a)

    def _calc_Kalman_gain(self):
        H = self.H
        B = self.B
        R = self.R
        self.K = B @ H.T @ inv(H @ B @ H.T + R)
