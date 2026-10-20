"""Model selection: cross-validated hyperparameter search over candidate models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from relcompat import __version__
from relcompat.config import Config
from relcompat.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_preprocessor,
)

logger = logging.getLogger(__name__)

# task name -> (target column, problem type)
TASKS = {
    "compatibility": ("compatible", "classification"),
    "longevity_reg": ("longevity_years", "regression"),
    "longevity_cls": ("lasts_5_years", "classification"),
}
SCORING = {"classification": "roc_auc", "regression": "neg_root_mean_squared_error"}

_TREE_GRID = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 8, 12, 16],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": ["sqrt", 0.5],
}
_XGB_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
}


def candidates(problem: str, seed: int) -> dict[str, tuple[object, dict]]:
    """Candidate estimators with their hyperparameter search spaces."""
    if problem == "classification":
        return {
            "logistic_regression": (LogisticRegression(max_iter=2000), {"C": loguniform(1e-2, 1e2)}),
            "random_forest": (RandomForestClassifier(random_state=seed, n_jobs=1), _TREE_GRID),
            "xgboost": (
                XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss"),
                _XGB_GRID,
            ),
        }
    return {
        "ridge": (Ridge(), {"alpha": loguniform(1e-2, 1e3)}),
        "random_forest": (RandomForestRegressor(random_state=seed, n_jobs=1), _TREE_GRID),
        "xgboost": (XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist"), _XGB_GRID),
    }


def _build_search(estimator, grid: dict, problem: str, cfg: Config) -> RandomizedSearchCV:
    pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
    prefix = "model__"
    if problem == "regression":
        # Durations are right-skewed, so models are fit on log1p(years).
        pipeline = TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
        prefix = "regressor__model__"
        cv = KFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    else:
        cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    space = {prefix + k: v for k, v in grid.items()}
    n_options = np.prod([len(v) if isinstance(v, list) else np.inf for v in space.values()])
    return RandomizedSearchCV(
        pipeline,
        space,
        n_iter=int(min(cfg.n_iter, n_options)),
        scoring=SCORING[problem],
        cv=cv,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
        refit=True,
    )


@dataclass
class TaskResult:
    task: str
    target: str
    problem: str
    best_name: str
    best_model: object
    fitted: dict[str, object] = field(default_factory=dict)  # candidate -> best estimator
    cv_scores: dict[str, float] = field(default_factory=dict)  # candidate -> mean CV score
    best_params: dict[str, dict] = field(default_factory=dict)


def train_task(task: str, X: pd.DataFrame, y: pd.Series, cfg: Config) -> TaskResult:
    target, problem = TASKS[task]
    result = TaskResult(task, target, problem, best_name="", best_model=None)
    for name, (estimator, grid) in candidates(problem, cfg.seed).items():
        search = _build_search(estimator, grid, problem, cfg)
        search.fit(X, y)
        # Report RMSE as a positive number; ROC-AUC as is.
        score = -search.best_score_ if problem == "regression" else search.best_score_
        result.fitted[name] = search.best_estimator_
        result.cv_scores[name] = round(float(score), 4)
        result.best_params[name] = {
            k.split("__")[-1]: (v.item() if hasattr(v, "item") else v)
            for k, v in search.best_params_.items()
        }
        logger.info("  %-20s CV %s = %.4f", name, SCORING[problem], score)

    pick = min if problem == "regression" else max
    result.best_name = pick(result.cv_scores, key=result.cv_scores.get)
    result.best_model = result.fitted[result.best_name]
    logger.info("  -> selected %s", result.best_name)
    return result


def make_bundle(
    result: TaskResult, background: pd.DataFrame, y_train: pd.Series, test_metrics: dict
) -> dict:
    """Self-describing artifact saved with joblib."""
    return {
        "model": result.best_model,
        "model_name": result.best_name,
        "task": result.task,
        "target": result.target,
        "problem": result.problem,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "background": background,
        # Regression predictions are clamped to the range seen in training.
        "target_range": (float(y_train.min()), float(y_train.max())),
        "cv_score": result.cv_scores[result.best_name],
        "best_params": result.best_params[result.best_name],
        "test_metrics": test_metrics,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
