import numpy as np


# Lorenz63
# b, r, s を指定した関数lorenz(t,x)を返す
def gen_l63(s, r, b):
    return lambda t, x: lorenz63(t, x, s, r, b)


def lorenz63(t, x, s=10, r=28, b=8/3):
    return np.array(
        [
            s * (x[1] - x[0]),
            x[0] * (r - x[2]) - x[1],
            x[0] * x[1] - b * x[2],
        ]
    )

