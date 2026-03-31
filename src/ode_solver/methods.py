"""
src/ode_solver/methods.py
=========================
Runge-Kutta step functions: Euler, Midpoint, Heun, RK4.

Each step function has the same signature:
    step(f, t, y, h) -> float
where:
    f  : callable(t, y) -> float   right-hand side of dy/dt = f(t, y)
    t  : float                     current time
    y  : float                     current solution value
    h  : float                     step size
"""


def euler_step(f, t, y, h):
    """1st-order Euler method."""
    return y + h * float(f(t, y))


def midpoint_step(f, t, y, h):
    """2nd-order Midpoint (explicit) method."""
    k1 = float(f(t,       y))
    k2 = float(f(t + h/2, y + h*k1/2))
    return y + h * k2


def heun_step(f, t, y, h):
    """2nd-order Heun (predictor-corrector) method."""
    k1 = float(f(t,     y))
    k2 = float(f(t + h, y + h*k1))
    return y + h * (k1 + k2) / 2


def rk4_step(f, t, y, h):
    """4th-order classical Runge-Kutta method."""
    k1 = float(f(t,       y))
    k2 = float(f(t + h/2, y + h*k1/2))
    k3 = float(f(t + h/2, y + h*k2/2))
    k4 = float(f(t + h,   y + h*k3))
    return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6


# Registry: name -> step function
RK_METHODS = {
    "euler":    euler_step,
    "midpoint": midpoint_step,
    "heun":     heun_step,
    "rk4":      rk4_step,
}
