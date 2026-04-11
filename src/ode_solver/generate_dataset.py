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
    required_columns = ["f_expression", "best_rk_method"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metadata_columns = [
        col for col in [
            "ode_type",
            "t0",
            "y0",
            "tf",
            "step_size_h",
            "num_steps",
            "best_l2_error",
            "best_cpu_time_ms",
            "euler_l2_error",
            "euler_time_ms",
            "midpoint_l2_error",
            "midpoint_time_ms",
            "heun_l2_error",
            "heun_time_ms",
            "rk4_l2_error",
            "rk4_time_ms",
        ]
        if col in df.columns
    ]
    df_two = df[required_columns + metadata_columns].copy()

    # Generate t values
    t_values = np.linspace(0, 1, n_points)
    y_values = np.linspace(0, 1, n_points)
    sample_columns = [f"sample_{i}" for i in range(n_points)]
    feature_columns = [
        "value_mean",
        "value_std",
        "value_min",
        "value_max",
        "value_abs_mean",
        "value_abs_max",
    ]

    # Create empty dataframe
    result_df = pd.DataFrame(
        columns=["f_expression"] + metadata_columns + sample_columns + feature_columns + ["best_rk_method"]
    )

    # Define symbolic variables
    t, y = sym.symbols("t y")

    for idx, row in df_two.iterrows():
        
        expr = sym.sympify(row["f_expression"])
        f = sym.lambdify((t, y), expr, "numpy")
        
        values = np.asarray(f(t_values, y_values), dtype=float)

        # Handle constant function case
        if values.shape == ():
            values = np.full(n_points, values)
        else:
            values = values.reshape(-1)

        stats = [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.mean(np.abs(values))),
            float(np.max(np.abs(values))),
        ]

        metadata_values = [row[col] for col in metadata_columns]
        new_row = [row["f_expression"]] + metadata_values + list(values) + stats + [row["best_rk_method"]]
        result_df.loc[idx] = new_row

    # Save to CSV
    # result_df.to_csv(output_file, index=False)

    # print(f"CSV created successfully: {output_file}")

    return result_df
