import numpy as np


# Henon
# a, b を指定した関数henon(x)を返す
def gen_henon(a=1.4, b=0.3):
    return lambda x: henon(x, a, b)


def henon(x, a, b):
    return np.array(
        [
            1 - a * x[0]**2 + x[1],
            b*x[0],
        ]
    )


# Kevin Hayden, Eric Olson, and Edriss S. Titi. Discrete data assimilation
# in the lorenz and 2d navier-stokes equations. 
# PHYSICA D-NONLINEAR PHENOMENA, 240(18):1416–1425, SEP 1 2011.
# def atr_radious_bound(a, b):
#     return rho


# def max_lyapunov_exponent_henon(a, b):
#     return beta
