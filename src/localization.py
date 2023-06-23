import numpy as np


# localization用の関数
def calc_dist(i, j, J=40):
    """NOTE: 1次元周期境界領域を前提"""
    return min([abs(i - j), abs(i + J - j)])


def polynomial(x, coefs):
    return np.array([coefs[i] * x ** (i) for i in range(len(coefs))]).sum()


def gaspari_cohn(d, c):
    x = d / c
    if d < c:
        return polynomial(x, [1, 0, -5 / 3, 5 / 8, 1 / 2, -1 / 4])
    elif c < d and d < 2 * c:
        return polynomial(x, [-2 / 3, 4, -5, 5 / 3, 5 / 8, -1 / 2, 1 / 12]) / x
    else:
        return 0
