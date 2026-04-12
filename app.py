from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from config.settings import STEP_SIZE
from src.ode_solver.inference import predict_best_method


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="ODE RK Method Selector",
    description="Local web app and API for predicting the best Runge-Kutta method for a user-supplied ODE.",
    version="1.0.0",
)


class ODERequest(BaseModel):
    f_expression: str = Field(..., examples=["t + y"])
    t0: float = 0.0
    y0: float = 1.0
    tf: float = 1.0
    step_size: float = STEP_SIZE
    exact_solution: str | None = None
    ode_type: str = "custom"

    @model_validator(mode="after")
    def validate_range(self):
        if self.tf <= self.t0:
            raise ValueError("tf must be greater than t0")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        return self


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: ODERequest):
    try:
        return predict_best_method(
            f_expression=payload.f_expression,
            t0=payload.t0,
            y0=payload.y0,
            tf=payload.tf,
            step_size=payload.step_size,
            exact_solution=payload.exact_solution,
            ode_type=payload.ode_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
