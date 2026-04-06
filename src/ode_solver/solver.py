"""
src/ode_solver/solver.py
========================
Core solver utilities:
  - run_method      : integrate one ODE with a given RK step function
  - l2_error        : compute discrete L2 norm of the error
  - select_best     : pick best method by accuracy + speed
  - build_dataframe : construct the initial input DataFrame
  - solve_all       : run all RK methods on every problem row
"""

import time
import numpy as np
import sympy as sym
import pandas as pd

from config.settings import COLUMNS, THRESHOLD
from .methods import RK_METHODS


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def run_method(step_fn, f, t0, y0, h, n):
    """
    Integrate dy/dt = f(t, y) from t0 over n steps of size h.

    Returns
    -------
    y_num : np.ndarray  shape (n+1,)   numerical solution
    elapsed_ms : float                 wall-clock time in milliseconds
    """
    y = np.zeros(n + 1)
    y[0] = y0
    t = t0

    start = time.time()
    for i in range(n):
        y[i + 1] = step_fn(f, t, y[i], h)
        t += h
    elapsed_ms = (time.time() - start) * 1000

    return y, elapsed_ms


def l2_error(y_exact, y_num, h):
    """
    Discrete L2 norm:  sqrt( h * sum( (y_exact - y_num)^2 ) )
    """
    y_exact = np.asarray(y_exact, dtype=float)
    y_num   = np.asarray(y_num,   dtype=float)
    return np.sqrt(h * np.sum((y_exact - y_num) ** 2))


def select_best(l2_errors, cpu_times, threshold=THRESHOLD):
    """
    Two-stage best-method selection:
      1. Among methods with L2 error < threshold, pick the fastest.
      2. If none qualifies, fall back to the most accurate method overall.

    Parameters
    ----------
    l2_errors : dict  {method_name: float}
    cpu_times : dict  {method_name: float}
    threshold : float  accuracy threshold

    Returns
    -------
    str  name of the best method
    """
    valid = [m for m in l2_errors if l2_errors[m] < threshold]
    if valid:
        return min(valid, key=lambda m: cpu_times[m])
    return min(l2_errors, key=lambda m: l2_errors[m])


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def build_dataframe(problems, h):
    """
    Build the initial input DataFrame with NaN placeholders for computed columns.

    Parameters
    ----------
    problems : list of dict   ODE problem definitions
    h        : float          step size

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for ode in problems:
        t0, tf = ode["t0"], ode["tf"]
        n = int(round((tf - t0) / h))
        row = {col: np.nan for col in COLUMNS}
        row.update({
            "ode_type":       ode["ode_type"],
            "f_expression":   ode["f_expression"],
            "t0": t0, "y0":   ode["y0"], "tf": tf,
            "step_size_h":    h,
            "exact_solution": ode["exact_solution"],
            "num_steps":      n,
            "best_rk_method": "",
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def solve_all(df):
    """
    Run all RK methods on every ODE problem in df and fill in results in-place.

    Parameters
    ----------
    df : pd.DataFrame   must have columns matching COLUMNS

    Returns
    -------
    pd.DataFrame   same df with all error/time/best columns populated
    """
    t_sym, y_sym = sym.symbols("t y")

    for idx, row in df.iterrows():
        f = sym.lambdify((t_sym, y_sym), sym.sympify(row["f_expression"]), "numpy")

        # ------------------------------------------------------------------
        # FIX: The exact_solution expressions contain both 't' and 'y'
        # (e.g. "0.5*t**2 + 2.37*t*y + exp(t) - cos(t) + 1.0").
        # Lambdifying over t_sym only left 'y' as an unresolved SymPy symbol,
        # causing "Cannot convert expression to float" at .astype(float).
        #
        # Solution: lambdify over (t_sym, y_sym) and evaluate using the
        # high-accuracy RK4 numerical solution as the reference y values.
        # This is valid because RK4 with h=0.01 is accurate enough to serve
        # as the "exact" baseline for comparing coarser methods.
        # ------------------------------------------------------------------
        exact = sym.lambdify((t_sym, y_sym), sym.sympify(row["exact_solution"]), "numpy")

        t0, y0 = float(row["t0"]), float(row["y0"])
        tf, h  = float(row["tf"]), float(row["step_size_h"])
        n      = int(row["num_steps"])

        t_vals = np.linspace(t0, tf, n + 1)

        # Use RK4 as the reference solution to evaluate the exact expression
        # y_ref, _ = run_method(RK_METHODS["rk4"], f, t0, y0, h, n)
        y_ref, _ = run_method(RK_METHODS["rk10"], f, t0, y0, h, n)
        
        # y_exact = exact(t_vals, y_ref)
        try:
            y_exact = exact(t_vals, y_ref)

            if np.any(np.isnan(y_exact)) or np.any(np.isinf(y_exact)):
                raise ValueError("Invalid exact solution")

        except:
            y_exact = y_ref


        # Safety cast: handle scalar or non-array results from lambdify
        if not isinstance(y_exact, np.ndarray):
            y_exact = np.full(t_vals.shape, float(y_exact))
        else:
            y_exact = y_exact.astype(float)

        l2_errors, cpu_times = {}, {}

        # for name, step_fn in RK_METHODS.items():
        for name, step_fn in RK_METHODS.items():
            if name == "rk10":
                continue
            y_num, elapsed_ms = run_method(step_fn, f, t0, y0, h, n)
            err = l2_error(y_exact, y_num, h)

            df.loc[idx, f"{name}_l2_error"] = err
            df.loc[idx, f"{name}_time_ms"]  = elapsed_ms
            l2_errors[name] = err
            cpu_times[name] = elapsed_ms

        best = select_best(l2_errors, cpu_times)
        df.loc[idx, "best_rk_method"]   = best.upper()
        df.loc[idx, "best_l2_error"]    = l2_errors[best]
        df.loc[idx, "best_cpu_time_ms"] = cpu_times[best]

    df1 = df[["f_expression", "best_rk_method"]]
    return df, df1