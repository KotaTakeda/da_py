import numpy as np


def rk4(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + k1 * dt / 2)
    k3 = f(t + dt / 2, y + k2 * dt / 2)
    k4 = f(t + dt, y + k3 * dt)
    yt = y + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return yt


# Euler
def euler(f, t, x, dt):
    return x + f(t, x) * dt


# Lorenz96
# F, J を指定した関数lorenz(t,x)を返す
def gen_l96(F):
    return lambda t, x: lorenz96_np(t, x, F)


def lorenz96(t, x, F):
    dx = np.zeros(len(x))
    for j in range(len(x)):
        dx[j] = (x[(j + 1) % 40] - x[(j - 2) % 40]) * x[(j - 1) % 40] - x[j]
    dx += F
    return dx


def lorenz96_np(t, x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
