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


# Kevin Hayden, Eric Olson, and Edriss S. Titi. Discrete data assimilation
# in the lorenz and 2d navier-stokes equations. 
# PHYSICA D-NONLINEAR PHENOMENA, 240(18):1416–1425, SEP 1 2011.
def atr_radious_bound(s=10, b=8/3, r=28):
    assert s > 0
    assert b > 1
    assert r > 0
    rho = b*(r+s)/(2*(b-1)**0.5)
    return rho


def max_lyapunov_exponent_l63(s=10, b=8/3, r=28):
    rho = atr_radious_bound(s, b, r)
    beta = rho - 1
    return beta
