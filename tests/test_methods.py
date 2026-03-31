"""
tests/test_methods.py
=====================
Unit tests for all RK step functions.
Validates each method against the simple ODE: dy/dt = -y, y(0) = 1
Exact solution: y(t) = exp(-t)
"""

import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.ode_solver.methods import euler_step, midpoint_step, heun_step, rk4_step

# dy/dt = -y  =>  exact: exp(-t)
f  = lambda t, y: -y
h  = 0.1
t0 = 0.0
y0 = 1.0
t1 = t0 + h
exact_y1 = math.exp(-h)   # ≈ 0.90484


def _integrate(step_fn, n_steps=10):
    """Run step_fn for n_steps and return final y."""
    t, y = t0, y0
    for _ in range(n_steps):
        y = step_fn(f, t, y, h)
        t += h
    return y


class TestEulerStep:
    def test_single_step_type(self):
        result = euler_step(f, t0, y0, h)
        assert isinstance(result, float)

    def test_single_step_direction(self):
        # Euler on -y should give y < 1
        result = euler_step(f, t0, y0, h)
        assert result < y0

    def test_single_step_accuracy(self):
        result = euler_step(f, t0, y0, h)
        assert abs(result - exact_y1) < 0.02   # 1st-order, loose tolerance

    def test_ten_steps(self):
        result = _integrate(euler_step)
        exact  = math.exp(-1.0)
        assert abs(result - exact) < 0.05


class TestMidpointStep:
    def test_single_step_type(self):
        result = midpoint_step(f, t0, y0, h)
        assert isinstance(result, float)

    def test_single_step_accuracy(self):
        result = midpoint_step(f, t0, y0, h)
        assert abs(result - exact_y1) < 0.005   # 2nd-order

    def test_better_than_euler(self):
        euler   = abs(euler_step(f, t0, y0, h) - exact_y1)
        midpt   = abs(midpoint_step(f, t0, y0, h) - exact_y1)
        assert midpt < euler


class TestHeunStep:
    def test_single_step_type(self):
        result = heun_step(f, t0, y0, h)
        assert isinstance(result, float)

    def test_single_step_accuracy(self):
        result = heun_step(f, t0, y0, h)
        assert abs(result - exact_y1) < 0.005   # 2nd-order

    def test_ten_steps(self):
        result = _integrate(heun_step)
        exact  = math.exp(-1.0)
        assert abs(result - exact) < 0.005


class TestRK4Step:
    def test_single_step_type(self):
        result = rk4_step(f, t0, y0, h)
        assert isinstance(result, float)

    def test_single_step_accuracy(self):
        result = rk4_step(f, t0, y0, h)
        assert abs(result - exact_y1) < 1e-6    # 4th-order, very tight

    def test_ten_steps(self):
        result = _integrate(rk4_step)
        exact  = math.exp(-1.0)
        assert abs(result - exact) < 1e-6

    def test_best_among_all(self):
        results = {
            "euler":    abs(euler_step(f, t0, y0, h)   - exact_y1),
            "midpoint": abs(midpoint_step(f, t0, y0, h) - exact_y1),
            "heun":     abs(heun_step(f, t0, y0, h)    - exact_y1),
            "rk4":      abs(rk4_step(f, t0, y0, h)     - exact_y1),
        }
        assert results["rk4"] == min(results.values())
