from scipy.optimize import newton


# 4th order Runge Kutta
def rk4(f, t, x, p, dt):
    """
    Arguments
        f: callable(t, x, *p)
        t: time
        x: state at t
        p: extra parameters for f
        dt: time step size

    Return
        xt: state at (t + dt)

    """
    k1 = f(t, x, *p)
    k2 = f(t + dt / 2, x + k1 * dt / 2, *p)
    k3 = f(t + dt / 2, x + k2 * dt / 2, *p)
    k4 = f(t + dt, x + k3 * dt, *p)
    xt = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return xt


# Euler
def euler(f, t, x, p, dt):
    return x + f(t, x, *p) * dt


def implicit_midpoint(f, t, x, p, dt):
    def implicit_eq(x_next):
        return x + dt * f(t + dt/2, (x + x_next) / 2, *p) - x_next

    x_next_guess = x + dt * f(t + dt/2, x, *p)
    x_next = newton(implicit_eq, x_next_guess)
    return x_next
