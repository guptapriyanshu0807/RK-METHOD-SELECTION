from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sympy as sym
from tensorflow import keras

from config.settings import STEP_SIZE
from .methods import RK_METHODS
from .solver import evaluate_exact_solution, l2_error, run_method, select_best


SAMPLE_POINTS = 100
METHOD_NAMES = ("euler", "midpoint", "heun", "rk4")


def _build_callable(f_expression: str):
    t_sym, y_sym = sym.symbols("t y")
    expr = sym.sympify(f_expression)
    return sym.lambdify((t_sym, y_sym), expr, "numpy")


def build_feature_record(
    f_expression: str,
    t0: float,
    y0: float,
    tf: float,
    step_size: float = STEP_SIZE,
    exact_solution: str | None = None,
    ode_type: str = "custom",
):
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    num_steps = int(round((tf - t0) / step_size))
    if num_steps <= 0:
        raise ValueError("The given time interval and step size produce no integration steps")

    f = _build_callable(f_expression)
    t_vals = np.linspace(t0, tf, num_steps + 1)
    y_ref, _ = run_method(RK_METHODS["rk10"], f, t0, y0, step_size, num_steps)

    reference_source = "rk10_reference"
    if exact_solution:
        try:
            y_exact = evaluate_exact_solution(exact_solution, t_vals)
            reference_source = "exact_solution"
        except ValueError:
            y_exact = y_ref
    else:
        y_exact = y_ref

    method_metrics = {}
    l2_errors = {}
    cpu_times = {}

    for name in METHOD_NAMES:
        y_num, elapsed_ms = run_method(RK_METHODS[name], f, t0, y0, step_size, num_steps)
        err = l2_error(y_exact, y_num, step_size)
        method_metrics[name] = {
            "solution": y_num,
            "l2_error": float(err),
            "time_ms": float(elapsed_ms),
        }
        l2_errors[name] = float(err)
        cpu_times[name] = float(elapsed_ms)

    best_method = select_best(l2_errors, cpu_times).upper()

    sample_t = np.linspace(0, 1, SAMPLE_POINTS)
    sample_y = np.linspace(0, 1, SAMPLE_POINTS)
    sampled_values = np.asarray(f(sample_t, sample_y), dtype=float)
    if sampled_values.shape == ():
        sampled_values = np.full(SAMPLE_POINTS, float(sampled_values))
    else:
        sampled_values = sampled_values.reshape(-1)
        if sampled_values.size != SAMPLE_POINTS:
            sampled_values = np.resize(sampled_values, SAMPLE_POINTS)

    record = {
        "f_expression": f_expression,
        "ode_type": ode_type,
        "t0": float(t0),
        "y0": float(y0),
        "tf": float(tf),
        "step_size_h": float(step_size),
        "num_steps": int(num_steps),
        "best_l2_error": float(l2_errors[best_method.lower()]),
        "best_cpu_time_ms": float(cpu_times[best_method.lower()]),
        "best_rk_method": best_method,
    }

    for name in METHOD_NAMES:
        record[f"{name}_l2_error"] = method_metrics[name]["l2_error"]
        record[f"{name}_time_ms"] = method_metrics[name]["time_ms"]

    for i, value in enumerate(sampled_values):
        record[f"sample_{i}"] = float(value)

    record["value_mean"] = float(np.mean(sampled_values))
    record["value_std"] = float(np.std(sampled_values))
    record["value_min"] = float(np.min(sampled_values))
    record["value_max"] = float(np.max(sampled_values))
    record["value_abs_mean"] = float(np.mean(np.abs(sampled_values)))
    record["value_abs_max"] = float(np.max(np.abs(sampled_values)))

    return {
        "record": record,
        "reference_source": reference_source,
        "t_values": t_vals,
        "y_reference": y_exact,
        "method_metrics": method_metrics,
    }


def prepare_model_input(record: dict, feature_names):
    frame = pd.DataFrame([{name: record.get(name, 0.0) for name in feature_names}])
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


@lru_cache(maxsize=1)
def load_prediction_artifacts(model_dir: str = "model"):
    model_path = Path(model_dir)
    keras_path = model_path / "model.keras"
    if keras_path.exists():
        model = keras.models.load_model(keras_path)
    else:
        model = joblib.load(model_path / "model.joblib")

    scaler = joblib.load(model_path / "scaler.joblib")
    encoder = joblib.load(model_path / "encoder.joblib")
    return model, scaler, encoder


def predict_best_method(
    f_expression: str,
    t0: float,
    y0: float,
    tf: float,
    step_size: float = STEP_SIZE,
    exact_solution: str | None = None,
    ode_type: str = "custom",
):
    analysis = build_feature_record(
        f_expression=f_expression,
        t0=t0,
        y0=y0,
        tf=tf,
        step_size=step_size,
        exact_solution=exact_solution,
        ode_type=ode_type,
    )

    model, scaler, encoder = load_prediction_artifacts()
    feature_names = list(getattr(scaler, "feature_names_in_", []))
    model_input = prepare_model_input(analysis["record"], feature_names)
    transformed = scaler.transform(model_input)

    probabilities = model.predict(transformed, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_method = str(encoder.inverse_transform([predicted_index])[0])

    predicted_key = predicted_method.lower()
    predicted_solution = analysis["method_metrics"][predicted_key]["solution"]

    return {
        "input": {
            "f_expression": f_expression,
            "t0": float(t0),
            "y0": float(y0),
            "tf": float(tf),
            "step_size": float(step_size),
            "exact_solution": exact_solution,
        },
        "reference_source": analysis["reference_source"],
        "predicted_best_method": predicted_method,
        "benchmark_best_method": analysis["record"]["best_rk_method"],
        "class_probabilities": {
            str(label): float(prob)
            for label, prob in zip(encoder.classes_, probabilities)
        },
        "method_metrics": {
            name.upper(): {
                "l2_error": metrics["l2_error"],
                "time_ms": metrics["time_ms"],
            }
            for name, metrics in analysis["method_metrics"].items()
        },
        "solution": {
            "method": predicted_method,
            "t_values": [float(v) for v in analysis["t_values"]],
            "y_values": [float(v) for v in predicted_solution],
        },
    }
