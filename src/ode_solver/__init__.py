# src/ode_solver/__init__.py
from .problems import ODE_PROBLEMS
from .methods  import RK_METHODS, euler_step, midpoint_step, heun_step, rk4_step 
from .solver   import build_dataframe, solve_all, run_method, l2_error, select_best 
from .generate_dataset import generate_function_dataset
