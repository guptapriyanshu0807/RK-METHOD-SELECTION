# First we are create a virtual envoriment .
    Create a virtual envoriment simply follow two steps:
    1. Create a Envoriment :-
    
    Windows: python -m venv .venv
    macOS/Linux: python3 -m venv .venv
    
    2.Activate the envoriment  
    
    Windows: .venv\Scripts\activate
    macOS/Linux: source .venv/bin/activate
### Install requirments.txt file 

    pip install -r requirements.txt




# ODE Solver — Runge-Kutta Method Comparison

Compares four Runge-Kutta methods (Euler, Midpoint, Heun, RK4) across a benchmark
suite of 60+ linear ODEs. For each problem the solver picks the best method based
on accuracy (L2 error) and speed (CPU time).

---

## Project Structure

```
ode_solver_project/
│
├── main.py                        # Entry point — run this
│
├── config/
│   ├── __init__.py
│   └── settings.py                # Global constants: paths, step size, threshold
│
├── src/
│   └── ode_solver/
│       ├── __init__.py
│       ├── problems.py            # 60+ ODE problem definitions
│       ├── methods.py             # RK step functions (Euler, Midpoint, Heun, RK4)
│       └── solver.py              # Core solver logic + DataFrame utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_methods.py            # Unit tests for RK step functions
│   └── test_solver.py             # Unit tests for solver utilities
│
├── data/                          # Auto-generated CSV files (git-ignored)
│   ├── final_ode_Problem2.csv
│   └── final_ode_Problem_results2.csv
│
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

This will:
1. Build the input CSV in `data/`
2. Run all four RK methods on every ODE problem
3. Select the best method per problem (accuracy + speed)
4. Save results to `data/final_ode_Problem_results2.csv`
5. Print a summary of which method won most often

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

Edit `config/settings.py` to change:

| Parameter    | Default  | Description                              |
|-------------|----------|------------------------------------------|
| `STEP_SIZE`  | `0.01`   | Integration step size h                  |
| `THRESHOLD`  | `1e-5`   | L2 error threshold for "good enough"     |
| `INPUT_CSV`  | `data/…` | Path where the input CSV is saved        |
| `OUTPUT_CSV` | `data/…` | Path where results CSV is saved          |

---

## Methods

| Method   | Order | Function calls/step |
|---------|-------|---------------------|
| Euler    | 1st   | 1                   |
| Midpoint | 2nd   | 2                   |
| Heun     | 2nd   | 2                   |
| RK4      | 4th   | 4                   |

**Best-method selection rule:**
- If any method achieves L2 error < `THRESHOLD`, the fastest among those is chosen.
- Otherwise, the most accurate method overall is selected.

---

## ODE Problem Categories

- Basic linear (constant / y-dependent RHS)
- Exponential growth & decay (`k*y`)
- Linear with polynomial / trig / exponential forcing
- Trigonometric and polynomial coefficients
- Rational coefficients (integrating-factor type)
- High growth-rate cases (`n*y`, n up to 10)
