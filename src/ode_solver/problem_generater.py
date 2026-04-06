import numpy as np
import random
from sympy import symbols, Function, Eq, dsolve, sympify

t = symbols('t')
y = Function('y')

# ─────────────────────────────────────────
# Exact solution generator using SymPy
# ─────────────────────────────────────────

import random

def generate_linear_constant_odes(
    n=1000,
    f_range=(-10.0, 10.0),
    y0_range=(0, 5),
    t0=0,
    tf=1,
    decimal_places=1
):
    cases = []

    for _ in range(n):
        f_val = round(random.uniform(*f_range), decimal_places)
        y0 = round(random.uniform(*y0_range), decimal_places)

        f_expr = str(f_val)
        exact = f"{f_val}*t + {y0}"

        case = {
            "ode_type": "linear",
            "f_expression": f_expr,
            "t0": t0,
            "y0": y0,
            "tf": tf,
            "exact_solution": exact
        }
        cases.append(case)

    return cases


if __name__ == "__main__":
    cases = generate_linear_constant_odes(n=1000)





def get_exact_solution(f_expr, y0, t0=0):
    try:
        f_sym = sympify(f_expr)
        ode = Eq(y(t).diff(t), f_sym)
        sol = dsolve(ode, ics={y(t0): y0})
        return str(sol.rhs)
    except:
        return None


# ─────────────────────────────────────────
# Parameter ranges
# ─────────────────────────────────────────
a_vals = np.linspace(-5, 5, 20)
b_vals = np.linspace(-3, 3, 15)
y0_vals = np.linspace(1, 10, 10)
tf_vals = [1, 2, 3]


# ─────────────────────────────────────────
# Generators for each category
# ─────────────────────────────────────────
def generate_simple(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        b = float(random.choice(b_vals))
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + {round(b,2)}"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "simple",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


def generate_poly(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        power = random.choice([1, 2, 3])
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + t**{power}"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "polynomial",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


def generate_trig(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        trig = random.choice(["sin(t)", "cos(t)","tan(t)"])
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + {trig}"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "trigonometric",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


def generate_poly_trig(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        power = random.choice([1, 2])
        trig = random.choice(["sin(t)", "cos(t)"])
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + t**{power} + {trig}"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "poly_trig",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


def generate_exp(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + exp(t)"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "exponential",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


def generate_mixed(n):
    data = []
    for _ in range(n):
        a = float(random.choice(a_vals))
        b = float(random.choice(b_vals))
        power = random.choice([1, 2])
        trig = random.choice(["sin(t)", "cos(t)"])
        y0 = float(random.choice(y0_vals))
        tf = float(random.choice(tf_vals))

        f_expr = f"{round(a,2)}*y + {round(b,2)} + t**{power} + {trig} + exp(t)"
        exact = get_exact_solution(f_expr, y0)

        data.append({
            "ode_type": "mixed_all",
            "f_expression": f_expr,
            "t0": 0,
            "y0": round(y0,2),
            "tf": tf,
            "exact_solution": exact
        })
    return data


# ─────────────────────────────────────────
# Generate full dataset
# ─────────────────────────────────────────
ODE_PROBLEMS = (
    generate_simple(1000) +
    generate_poly(1000) +
    generate_trig(1000) +
    generate_poly_trig(1000) +
    generate_exp(1000) +
    generate_mixed(1000)
)


# ─────────────────────────────────────────
# Save to file (same format as yours)
# ─────────────────────────────────────────
with open("problems.py", "w") as f:
    f.write("ODE_PROBLEMS = [\n")
    for p in ODE_PROBLEMS:
        f.write(f"    {p},\n")
    f.write("]\n")