"""
main.py
=======
Entry point for the ODE Solver project.

Usage:
    python main.py
"""

import sys
import os
import pandas as pd
# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config.settings    import INPUT_CSV, OUTPUT_CSV, STEP_SIZE , MODEL_CSV , FUNCTION_VALUE
from src.ode_solver     import ODE_PROBLEMS, build_dataframe, solve_all ,generate_function_dataset , model_RK_

def main():
    print("=" * 55)
    print("  ODE Solver: Runge-Kutta Method Comparison")
    print("=" * 55)

    # 1. Build input DataFrame and save
    print(f"\n[1/5] Building dataset ({len(ODE_PROBLEMS)} problems, h={STEP_SIZE}) ...")
    df = build_dataframe(ODE_PROBLEMS, STEP_SIZE)
    df.to_csv(INPUT_CSV, index=False)
    print(f"      Input saved  → {INPUT_CSV}")

    # 2. Solve all problems
    print("\n[2/5] Running RK methods (Euler, Midpoint, Heun, RK4) ...")
    df,df1= solve_all(df)

    # 3. Save results
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[3/5] Results saved → {OUTPUT_CSV}")

    #4. Save results
    df1.to_csv(MODEL_CSV, index=False)
    print(f"\n[4/5] Results saved → {MODEL_CSV}")
    # Summary


    df = generate_function_dataset(df1)
    df.to_csv(FUNCTION_VALUE,index = False)
    print(f"\n[5/5] Results saved → {FUNCTION_VALUE}")
    print(f"")
    
    #Summary
    print("\n--- Best Method Distribution ---")
    print(df["best_rk_method"].value_counts().to_string())
    print("\nDone.")

    data = pd.read_csv("data/function_value.csv")
    acc = model_RK_(data)
    print(f"MOdel accuracy: {acc}")

if __name__ == "__main__":
    main()
