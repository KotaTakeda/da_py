from scipy.optimize import newton


# 4th order Runge Kutta
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


def implicit_midpoint(f, t, x, dt):
    def implicit_eq(x_next):
        return x + dt * f(t + dt/2, (x + x_next) / 2) - x_next

    x_next_guess = x + dt * f(t + dt/2, x)
    x_next = newton(implicit_eq, x_next_guess)
    return x_next
