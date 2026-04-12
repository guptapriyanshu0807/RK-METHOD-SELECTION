import numpy as np

from src.ode_solver.inference import build_feature_record, prepare_model_input


class TestInferenceHelpers:
    def test_build_feature_record_returns_expected_shape(self):
        result = build_feature_record(
            f_expression="-y",
            t0=0.0,
            y0=1.0,
            tf=1.0,
            step_size=0.1,
            exact_solution="exp(-t)",
        )

        record = result["record"]
        assert record["best_rk_method"] in {"EULER", "HEUN", "MIDPOINT", "RK4"}
        assert len(result["t_values"]) == 11
        assert set(result["method_metrics"].keys()) == {"euler", "midpoint", "heun", "rk4"}
        assert np.isfinite(record["value_mean"])
        assert "sample_99" in record

    def test_prepare_model_input_numeric_conversion(self):
        record = {
            "ode_type": "custom",
            "t0": 0.0,
            "y0": 1.0,
            "tf": 1.0,
            "step_size_h": 0.1,
            "num_steps": 10,
            "sample_0": 1.0,
            "value_mean": 1.0,
        }
        frame = prepare_model_input(record, ["ode_type", "t0", "sample_0", "value_mean"])
        assert frame.shape == (1, 4)
        assert frame.loc[0, "ode_type"] == 0.0
