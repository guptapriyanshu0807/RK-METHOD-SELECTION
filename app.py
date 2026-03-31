from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

app = FastAPI(
    title="ODE Solver — RK Method Predictor",
    description="Given dy/dx values at sample points, predicts the best Runge-Kutta method for solving a 1st order linear ODE.",
    version="1.0.0"
)

# ── Load trained model ──────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


# ── Schemas ─────────────────────────────────────────────────────────
class ODEInput(BaseModel):
    dydx_values: List[float]

class PredictionOutput(BaseModel):
    best_rk_method: str
    confidence: float
    all_probabilities: dict


# ── Routes ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Welcome to the ODE RK Method Predictor API 🚀",
        "usage": "POST /predict with a list of dy/dx values",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: ODEInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check model file.")

    try:
        X = np.array(data.dydx_values).reshape(1, -1)

        probabilities = model.predict_proba(X)[0]
        classes = model.classes_

        best_idx = np.argmax(probabilities)
        best_method = classes[best_idx]
        confidence = round(float(probabilities[best_idx]), 4)

        all_probs = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }

        return PredictionOutput(
            best_rk_method=str(best_method),
            confidence=confidence,
            all_probabilities=all_probs
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))