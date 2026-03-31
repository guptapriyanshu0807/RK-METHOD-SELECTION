"""
tests/test_solver.py
====================
Unit tests for solver utilities:
  - l2_error
  - select_best
  - build_dataframe
"""

import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.ode_solver.solver import l2_error, select_best, build_dataframe

# -----------------------------------------------------------------------
SAMPLE_PROBLEMS = [
    {"ode_type": "linear", "f_expression": "-y", "t0": 0, "y0": 1,
     "tf": 1, "exact_solution": "exp(-t)"},
]


class TestL2Error:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert l2_error(y, y, h=0.1) == 0.0

    def test_known_value(self):
        y_exact = np.array([0.0, 0.0, 0.0])
        y_num   = np.array([1.0, 1.0, 1.0])
        # sqrt(h * 3 * 1) = sqrt(0.5 * 3) = sqrt(1.5)
        expected = math.sqrt(0.5 * 3)
        assert abs(l2_error(y_exact, y_num, h=0.5) - expected) < 1e-12

    def test_nonnegative(self):
        y_exact = np.random.rand(10)
        y_num   = np.random.rand(10)
        assert l2_error(y_exact, y_num, h=0.1) >= 0.0


class TestSelectBest:
    def test_fastest_within_threshold(self):
        l2 = {"euler": 1e-7, "rk4": 1e-9}
        cpu = {"euler": 1.0,  "rk4": 5.0}
        # Both below threshold; euler is fastest
        assert select_best(l2, cpu, threshold=1e-5) == "euler"

    def test_most_accurate_fallback(self):
        l2 = {"euler": 1e-3, "rk4": 1e-4}
        cpu = {"euler": 1.0,  "rk4": 5.0}
        # None below threshold=1e-5; rk4 is most accurate
        assert select_best(l2, cpu, threshold=1e-5) == "rk4"

    def test_single_method(self):
        l2  = {"euler": 1e-2}
        cpu = {"euler": 2.0}
        assert select_best(l2, cpu, threshold=1e-5) == "euler"

    def test_all_below_threshold_picks_fastest(self):
        l2  = {"a": 1e-8, "b": 1e-8, "c": 1e-8}
        cpu = {"a": 3.0,  "b": 1.0,  "c": 2.0}
        assert select_best(l2, cpu, threshold=1e-5) == "b"


class TestBuildDataframe:
    def test_row_count(self):
        df = build_dataframe(SAMPLE_PROBLEMS, h=0.1)
        assert len(df) == 1

    def test_columns_present(self):
        df = build_dataframe(SAMPLE_PROBLEMS, h=0.1)
        assert "f_expression" in df.columns
        assert "num_steps" in df.columns
        assert "rk4_l2_error" in df.columns

    def test_num_steps_calculated(self):
        df = build_dataframe(SAMPLE_PROBLEMS, h=0.1)
        assert df.loc[0, "num_steps"] == 10   # (1-0)/0.1 = 10

    def test_error_columns_nan(self):
        df = build_dataframe(SAMPLE_PROBLEMS, h=0.1)
        assert np.isnan(df.loc[0, "euler_l2_error"])
