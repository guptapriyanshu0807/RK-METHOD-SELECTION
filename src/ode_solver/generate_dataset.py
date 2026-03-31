import numpy as np
import pandas as pd
import sympy as sym

def generate_function_dataset(df):
    """
    Generate dataset by evaluating f_expression at n_points between 0 and 1.
    
    Parameters:
    df : pandas DataFrame (must contain 'f_expression' and 'best_rk_method')
    output_file : name of csv file to save
    n_points : number of t values
    
    Returns:
    result_df : Generated DataFrame
    """
    output_file="function_values.csv"
    n_points=100
    # Keep required columns
    df_two = df[["f_expression", "best_rk_method"]].copy()

    # Generate t values
    t_values = np.linspace(0, 1, n_points)
    y_values = np.linspace(0, 1, n_points)
    t_columns = [f"t{i}" for i in range(n_points)]

    # Create empty dataframe
    result_df = pd.DataFrame(columns=["f_expression"] + t_columns + ["best_rk_method"])

    # Define symbolic variables
    t, y = sym.symbols("t y")

    for idx, row in df_two.iterrows():
        
        expr = sym.sympify(row["f_expression"])
        f = sym.lambdify((t, y), expr, "numpy")
        
        values = np.array(f(t_values, y_values))

        # Handle constant function case
        if values.shape == ():
            values = np.full(n_points, values)

        new_row = [row["f_expression"]] + list(values) + [row["best_rk_method"]]
        result_df.loc[idx] = new_row

    # Save to CSV
    # result_df.to_csv(output_file, index=False)

    # print(f"CSV created successfully: {output_file}")

    return result_df