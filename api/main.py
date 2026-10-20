"""REST API for compatibility & longevity predictions.

    uvicorn api.main:app --reload      # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relcompat import __version__  # noqa: E402
from relcompat.models.predict import Predictor  # noqa: E402
from relcompat.schemas import CoupleInput, PredictionOut  # noqa: E402

app = FastAPI(
    title="Relationship Compatibility & Longevity API",
    description="Predicts couple compatibility and expected relationship duration. "
    "Models are trained on synthetic data and are for educational use only.",
    version=__version__,
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor(os.environ.get("RELCOMPAT_MODELS_DIR"))


def predictor_dependency() -> Predictor:
    try:
        return get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    try:
        get_predictor()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return {"status": "ok", "models_loaded": loaded, "version": __version__}


@app.get("/model-info")
def model_info(predictor: Predictor = Depends(predictor_dependency)) -> dict:
    metrics = predictor.metrics()
    return {
        "models": predictor.model_versions,
        "test_metrics": {t: info["test_metrics"] for t, info in metrics.get("tasks", {}).items()},
    }


@app.post("/predict", response_model=PredictionOut)
def predict(couple: CoupleInput, predictor: Predictor = Depends(predictor_dependency)) -> PredictionOut:
    return predictor.predict(couple)
