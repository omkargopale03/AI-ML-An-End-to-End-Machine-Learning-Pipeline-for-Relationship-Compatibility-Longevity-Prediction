"""Load trained artifacts and score a couple."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from relcompat.config import load_config
from relcompat.features.build_features import build_features, couple_to_frame
from relcompat.models.explain import top_factors
from relcompat.schemas import CoupleInput, PredictionOut

MODEL_FILES = {
    "compatibility": "compatibility.joblib",
    "longevity_reg": "longevity_reg.joblib",
    "longevity_cls": "longevity_cls.joblib",
}


def compatibility_label(score: float) -> str:
    if score >= 65:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Low"


class Predictor:
    """Loads the three model bundles once and serves predictions."""

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else load_config().models_dir
        missing = [f for f in MODEL_FILES.values() if not (self.models_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model files {missing} not found in {self.models_dir}. "
                "Run `python run_pipeline.py` first."
            )
        self.bundles = {task: joblib.load(self.models_dir / f) for task, f in MODEL_FILES.items()}

    @property
    def model_versions(self) -> dict[str, str]:
        return {
            task: f"{b['model_name']} v{b['version']} ({b['trained_at']})"
            for task, b in self.bundles.items()
        }

    def metrics(self) -> dict:
        path = self.models_dir / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def predict(self, couple: CoupleInput | dict) -> PredictionOut:
        if isinstance(couple, dict):
            couple = CoupleInput.model_validate(couple)
        X = build_features(couple_to_frame(couple))
        comp, reg, cls = (self.bundles[t] for t in MODEL_FILES)

        p_compatible = float(comp["model"].predict_proba(X)[0, 1])
        lo, hi = reg.get("target_range", (0.0, float("inf")))
        years = min(max(float(reg["model"].predict(X)[0]), lo), hi)
        score = round(100 * p_compatible, 1)
        return PredictionOut(
            compatibility_score=score,
            compatible=p_compatible >= 0.5,
            compatibility_label=compatibility_label(score),
            expected_years=round(years, 1),
            lasts_5_years_probability=round(float(cls["model"].predict_proba(X)[0, 1]), 3),
            top_factors=top_factors(comp["model"], X, comp["background"], k=5),
            longevity_factors=top_factors(reg["model"], X, reg["background"], k=3),
            model_versions=self.model_versions,
        )
