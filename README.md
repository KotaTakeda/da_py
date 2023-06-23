# topological data assimilation

## requirements

```
pip install --upgrade pip
pip install -r requirements.txt
```

### TDA

- persim
- gudhi
- pot
- tensorflow-macos
- tensorflow-metal
- (torch)
- (eagerpy)

## 基本同化設定

```py
import numpy as np
# Lorenz96の設定
J = 40
F = 8
lorenz = gen_l96(F)

# 同化step
# 時間発展は0.01ごとに行う
dt = 0.05

# モデルの遷移関数(非線形)
# 0.01ずつ時間発展させる
# dtは同化step
def M(x, dt):
    for i in range(int(dt/0.01)):
        x = rk4(lorenz, 0, x, 0.01)
    return x

# 単位行列
I = np.eye(J)

# 観測
H = I

# モデル誤差共分散, 最初は完全モデルを仮定
Q = np.zeros((J, J))

# 観測誤差共分散, 後で定数倍の変化をさせる.
r = 1.0
R = r*I

# 観測値と真値
x_true = np.load('../data/x_true.npy')
y = np.load('../data/x_obs.npy')
if not np.isclose(r, 1.0):
    y = x_true + np.random.normal(loc=0, scale=r, size=x_true.shape) # R = r*I

# KFの初期値
np.random.seed(1)
x_0 = x_true[np.random.randint(len(x_true)-1)]
```
