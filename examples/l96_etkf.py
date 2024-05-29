"""
Lorenz 96 + ETKF

obsevation: linear and Gauss noise
"""
import numpy as np
import matplotlib.pyplot as plt
from da.l96 import lorenz96
from da.scheme import rk4
from da.loss import loss_rms
from da.visualize import plot_loss
from da.etkf import ETKF

# Lorenz96の設定
J = 40
F = 8

# Generate test data
# time step size
dt = 0.01
# number of time step, 2 years : 360*20*2
N = 360*20*2 

# initial state near the statinary point
x0 = F*np.ones(J) # the statinary point
x0[19] *= 1.001 # perturb

# use my modules
p = (F,) # parameters
scheme = rk4

# モデルの遷移関数(非線形)
# 0.01ずつ時間発展させる
# Dtは同化step
def M(x, t):
    n = int(t/dt)
    for i in range(n):
        x = rk4(lorenz96, 0, x, p, dt)
    if t - n*dt > 0:
        x = rk4(lorenz96, 0, x, p, t - n*dt)
    return x


x_true = np.zeros((N,len(x0)))
x = x0
x_true[0] = x[:]

for n in range(1,N):
    t = n*dt
    x = scheme(lorenz96, t, x, p, dt)
    x_true[n] = x[:]

obs_per = 5
Dt = obs_per * dt
x_true = x_true[360*20:][::obs_per]  # 1年分を捨て，6h毎に取り出す

# 観測値: 観測誤差共分散, 後で定数倍の変化をさせる.
H = np.eye(J)
r = 1.0
R = r*np.eye(J)
# noise = np.random.normal(loc=0, scale=np.sqrt(r), size=x_true.shape)
noise = np.random.multivariate_normal(mean=np.zeros(J), cov=R, size=len(x_true))
y = (H@x_true.T).T + noise

# アンサンブル数
m = 20

# 初期値
seed = 10
np.random.seed(seed)
x_0 = x_true[np.random.randint(len(x_true)-1)]
P_0 = 25*np.eye(J)
X_0 = x_0 + np.random.multivariate_normal(np.zeros_like(x_0), P_0, m)  # (m, dim_x)

# RUN DA
etkf = ETKF(M, H, R, alpha=1.1)
etkf.initialize(X_0)
for y_obs in y:
    etkf.forecast(Dt)
    etkf.update(y_obs)

x_a = etkf.x

# Plot
fig, ax = plt.subplots(figsize=(10, 2))
plot_loss((H@(x_true.T)).T, y, loss_rms, ax=ax, label='obs', lw=0.3)
plot_loss(x_true, x_a, loss_rms, ax=ax, label='etkf', lw=0.3)
ax.legend()
# ax.set_ylim([0, 2])
plt.show()