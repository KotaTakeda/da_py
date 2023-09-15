import numpy as np

# Lorenz96
# F, J を指定した関数lorenz(t,x)を返す
def gen_l96(F):
    """
    Args:
        - F, float: external force
    """
    return lambda t, x: lorenz96_np(t, x, F)


def lorenz96(t, x, F):
    dx = np.zeros(len(x))
    for j in range(len(x)):
        dx[j] = (x[(j + 1) % 40] - x[(j - 2) % 40]) * x[(j - 1) % 40] - x[j]
    dx += F
    return dx


def lorenz96_np(t, x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
