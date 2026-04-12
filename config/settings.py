"""
config/settings.py
==================
Global configuration: file paths, solver parameters, and output columns.
"""

# --- File Paths ---
INPUT_CSV  = "data/final_ode_Problem.csv"
OUTPUT_CSV = "data/final_ode_Problem_results.csv"
MODEL_CSV = "data/model_ode.csv"
FUNCTION_VALUE = "data/function_value.csv"
# --- Solver Parameters ---
STEP_SIZE = 0.01       # Integration step size h
THRESHOLD = 1e-5      # L2 error threshold for "good enough" accuracy

# --- DataFrame Columns ---
COLUMNS = [
    "ode_type", "f_expression", "t0", "y0", "tf", "step_size_h",
    "exact_solution", "num_steps",
    "euler_l2_error",    "euler_time_ms",
    "midpoint_l2_error", "midpoint_time_ms",
    "heun_l2_error",     "heun_time_ms",
    "rk4_l2_error",      "rk4_time_ms",
    "best_l2_error", "best_cpu_time_ms", "best_rk_method",
]
