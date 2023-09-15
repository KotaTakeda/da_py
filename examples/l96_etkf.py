import os
import numpy as np
import matplotlib.pyplot as plt
from da.l96 import gen_l96
from da.scheme import rk4
from da.loss import loss_rms
from da.visualize import plot_loss
from da.etkf import ETKF

# Lorenz96の設定
J = 40
F = 8
lorenz = gen_l96(F)

# 同化step
# 時間発展は0.01ごとに行う
Dt = 0.05 # 6hに相当

# モデルの遷移関数(非線形)
# 0.01ずつ時間発展させる
# Dtは同化step
def M(x, Dt):
    for i in range(int(Dt/0.01)):
        x = rk4(lorenz, 0, x, 0.01)
    return x

# 単位行列
I = np.eye(J)

# 観測
H = I

# モデル誤差共分散, 最初は完全モデルを仮定
Q = np.zeros((J, J))


# Generate test data
if os.path.exists("x_true.npy"):
    x_true = np.load("x_true.npy")
else:
    N = 360*20*2  # 2年分に相当
    dt = 0.01
    x0 = F*np.ones(J)
    x0[19] *= 1.001
    x_true = np.zeros((N, len(x0)))
    x = x0
    x_true[0] = x[:]

    for n in range(1, N):
        t = n*dt
        x = rk4(lorenz, t, x, dt)
        x_true[n] = x[:]

    x_true = x_true[360*20:][::5]  # 1年分を捨て，6h毎に取り出す

# da settings
# 観測値: 観測誤差共分散, 後で定数倍の変化をさせる.
r = 1.0
R = r*I
noise = np.random.normal(loc=0, scale=r, size=x_true.shape)
y = x_true + noise

# 初期値
seed = 10
np.random.seed(seed)
x_0 = x_true[np.random.randint(len(x_true)-1)]
P_0 = 25*I

# RUN DA
etkf = ETKF(M, H, R, x_0, P_0, m=20, alpha=1.1, seed=seed, addaptive=False)
for y_obs in y:
    etkf.forecast(Dt)
    etkf.update(y_obs)

x_assim = etkf.x

# Plot
fig, ax = plt.subplots(figsize=(10, 2))
plot_loss(x_true, y, loss_rms, ax=ax, label='obs', lw=0.3)
plot_loss(x_true, x_assim, loss_rms, ax=ax, label='etkf', lw=0.3)
ax.legend()
# ax.set_ylim([0, 2])
plt.show()