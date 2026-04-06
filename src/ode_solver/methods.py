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


def rk10_step(f, t, y, h):
    """
    10th-order Runge-Kutta method (Hairer, 1978).
    Local error O(h^11). 17 stages.
    """
    k1  = float(f(t, y))
    k2  = float(f(t + h*(1/10),
                  y + h*(1/10*k1)))
    k3  = float(f(t + h*(2/15),
                  y + h*(1/15*k1 + 1/15*k2)))
    k4  = float(f(t + h*(3/10),
                  y + h*(3/40*k1 + 9/40*k3)))
    k5  = float(f(t + h*(2/5),
                  y + h*(66/125*k1 - 84/125*k2 + 84/125*k3 + 12/125*k4)))
    k6  = float(f(t + h*(4/5),
                  y + h*(-19/28*k1 + 19/7*k2 - 107/28*k3 + 11/7*k4 + 5/4*k5)))
    k7  = float(f(t + h*(7/10),
                  y + h*(151/1008*k1 + 475/1008*k3 + 17/504*k4 - 25/504*k5 + 5/504*k6)))
    k8  = float(f(t + h*(4/5),
                  y + h*(151/1008*k1 + 475/1008*k3 + 17/504*k4 - 25/504*k5 + 5/504*k6)))
    k9  = float(f(t + h*(9/10),
                  y + h*(4523/12800*k1 - 9/200*k2 + 2835/12800*k3 + 243/12800*k4
                       + 891/6400*k5  + 81/12800*k6 - 81/1280*k7 + 81/1280*k8)))
    k10 = float(f(t + h*(1/2),
                  y + h*(-7601/153600*k1 + 12075/153600*k3 - 12267/153600*k4
                       +  6750/153600*k5 -  4617/153600*k6 + 13851/153600*k7
                       +  4617/153600*k8 - 12393/153600*k9)))
    k11 = float(f(t + h*(1/10),
                  y + h*(13/240*k1 - 25/240*k5 + 25/240*k8 - 13/240*k10)))
    k12 = float(f(t + h*(3/10),
                  y + h*(13873/622080*k1  -  7945/622080*k3  +  7945/622080*k4
                       +   134/622080*k5  -  6478/622080*k6  +  6478/622080*k7
                       -   134/622080*k8  -  7945/622080*k9  +  7945/622080*k10
                       - 13873/622080*k11)))
    k13 = float(f(t + h*(1/2),
                  y + h*(-8885/622080*k1  + 16375/622080*k3  - 16375/622080*k4
                       -  2930/622080*k5  + 10018/622080*k6  - 10018/622080*k7
                       +  2930/622080*k8  + 16375/622080*k9  - 16375/622080*k10
                       +  8885/622080*k11)))
    k14 = float(f(t + h*(7/10),
                  y + h*(13873/622080*k1  -  7945/622080*k3  +  7945/622080*k4
                       +   134/622080*k5  -  6478/622080*k6  +  6478/622080*k7
                       -   134/622080*k8  -  7945/622080*k9  +  7945/622080*k10
                       - 13873/622080*k11)))
    k15 = float(f(t + h*(9/10),
                  y + h*(4523/12800*k1  - 9/200*k2    + 2835/12800*k3
                       +  243/12800*k4  + 891/6400*k5  +   81/12800*k6
                       -   81/1280*k7   + 81/1280*k8   + 2835/12800*k9
                       -  243/12800*k10 + 4523/12800*k11 + 9/200*k14)))
    k16 = float(f(t + h,
                  y + h*(-7/1440*k1 + 125/1440*k5 - 125/1440*k8 + 7/1440*k10)))
    k17 = float(f(t + h,
                  y + h*(7/1440*k1 - 125/1440*k5 + 125/1440*k8
                       - 7/1440*k10 + k16)))

    return y + h * (
          7/90  * k1
        + 32/90 * k5
        + 32/90 * k8
        +  7/90 * k10
        + 12/90 * k17
    )


# Registry: name -> step function
RK_METHODS = {
    "euler": euler_step,
    "heun": heun_step,
    "midpoint": midpoint_step,
    "rk4": rk4_step,
    "rk10": rk10_step   # only for reference
}