from pathlib import Path
from functools import lru_cache
import random

import numpy as np
import sympy as sym

t = sym.symbols("t")
y = sym.Function("y")


def generate_linear_constant_odes(
    n=1000,
    f_range=(-10.0, 10.0),
    y0_range=(0, 5),
    t0=0,
    tf=1,
    decimal_places=1,
):
    cases = []

    for _ in range(n):
        f_val = round(random.uniform(*f_range), decimal_places)
        y0_val = round(random.uniform(*y0_range), decimal_places)

        cases.append(
            {
                "ode_type": "linear",
                "f_expression": str(f_val),
                "t0": t0,
                "y0": y0_val,
                "tf": tf,
                "exact_solution": f"{f_val}*t + {y0_val}",
            }
        )

    return cases


@lru_cache(maxsize=None)
def get_exact_solution(f_expr, y0_value, t0=0):
    try:
        s = sym.symbols("s")
        f_sym = sym.sympify(
            f_expr,
            locals={
                "t": t,
                "y": y(t),
                "exp": sym.exp,
                "sin": sym.sin,
                "cos": sym.cos,
                "tan": sym.tan,
            },
        )
        f_sym = sym.nsimplify(f_sym, rational=True)
        y0_sym = sym.nsimplify(y0_value, rational=True)

        a_coeff = sym.diff(f_sym, y(t))
        forcing = f_sym - a_coeff * y(t)

        if y(t) not in forcing.atoms(sym.Function):
            if a_coeff == 0:
                exact = y0_sym + sym.integrate(forcing.subs(t, s), (s, t0, t))
            else:
                integral_term = sym.integrate(
                    sym.exp(-a_coeff * (s - t0)) * forcing.subs(t, s),
                    (s, t0, t),
                )
                exact = sym.exp(a_coeff * (t - t0)) * (y0_sym + integral_term)
            return str(sym.expand(exact))

        ode = sym.Eq(sym.diff(y(t), t), f_sym)
        sol = sym.dsolve(ode, ics={y(t0): y0_sym})
        return str(sym.expand(sol.rhs))
    except Exception:
        return None


a_vals = np.linspace(-5, 5, 20)
b_vals = np.linspace(-3, 3, 15)
y0_vals = np.linspace(1, 10, 10)
tf_vals = [1, 2, 3]


def generate_simple(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        b = float(random.choice(b_vals))
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + {round(b, 2)}"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "simple",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def generate_poly(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        power = random.choice([1, 2, 3])
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + t**{power}"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "polynomial",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def generate_trig(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        trig = random.choice(["sin(t)", "cos(t)"])
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + {trig}"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "trigonometric",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def generate_poly_trig(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        power = random.choice([1, 2])
        trig = random.choice(["sin(t)", "cos(t)"])
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + t**{power} + {trig}"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "poly_trig",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def generate_exp(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + exp(t)"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "exponential",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def generate_mixed(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        b = float(random.choice(b_vals))
        power = random.choice([1, 2])
        trig = random.choice(["sin(t)", "cos(t)"])
        y0_val = float(random.choice(y0_vals))
        tf_val = float(random.choice(tf_vals))

        f_expr = f"{round(a, 2)}*y + {round(b, 2)} + t**{power} + {trig} + exp(t)"
        exact = get_exact_solution(f_expr, round(y0_val, 2))

        data.append(
            {
                "ode_type": "mixed_all",
                "f_expression": f_expr,
                "t0": 0,
                "y0": round(y0_val, 2),
                "tf": tf_val,
                "exact_solution": exact,
            }
        )
    return data


def build_ode_problems(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    return (
        generate_linear_constant_odes(1000)
        + generate_simple(1000)
        + generate_poly(1000)
        + generate_trig(1000)
        + generate_poly_trig(1000)
        + generate_exp(1000)
        + generate_mixed(1000)
    )


def write_problems_file(output_path=None, seed=42, problems=None):
    if problems is None:
        problems = build_ode_problems(seed=seed)
    if output_path is None:
        output_path = Path(__file__).with_name("problems.py")
    else:
        output_path = Path(output_path)

    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write('"""\n')
        file_obj.write("src/ode_solver/problems.py\n")
        file_obj.write('"""\n\n')
        file_obj.write("ODE_PROBLEMS = [\n")
        for problem in problems:
            file_obj.write(f"    {problem},\n")
        file_obj.write("]\n")

    return problems

if __name__ == "__main__":
    write_problems_file(problems=build_ode_problems())
