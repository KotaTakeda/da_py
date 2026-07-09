import numpy as np

# Lorenz96
# F, J を指定した関数lorenz(t,x)を返す
# def gen_l96(F):
#     """
#     Args:
#         - F, float: external force
#     """
#     return lambda t, x: lorenz96_np(t, x, F)


# def lorenz96(t, x, F):
#     """
#     """
#     dx = np.zeros(len(x))
#     for j in range(len(x)):
#         dx[j] = (x[(j + 1) % 40] - x[(j - 2) % 40]) * x[(j - 1) % 40] - x[j]
#     dx += F
#     return dx


def lorenz96(t, x, F):
    """
    F (Nx, ): external force
    """
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def two_thirds_observation(Nx):
    """Periodic 2/3 partial-observation operator for Lorenz-96.

    For each consecutive triple ``(x[3j], x[3j+1], x[3j+2])`` the first two
    components are observed and the third is left unobserved, so
    ``Ny = 2 * Nx / 3``. Requires ``Nx`` divisible by 3.

    Returns ``(H, observed_indices)`` where ``H`` has shape ``(Ny, Nx)`` and
    ``observed_indices`` lists the observed state components in order.
    """
    if Nx % 3 != 0:
        raise ValueError("2/3 observation pattern requires Nx divisible by 3")
    observed = np.array([i for i in range(Nx) if i % 3 != 2])
    H = np.zeros((observed.size, Nx))
    H[np.arange(observed.size), observed] = 1.0
    return H, observed
