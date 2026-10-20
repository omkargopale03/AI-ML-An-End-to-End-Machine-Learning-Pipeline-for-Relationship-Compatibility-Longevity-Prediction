"""End-to-end pipeline: generate -> validate -> features -> train -> evaluate -> save."""

from __future__ import annotations

import json
import logging
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from relcompat.config import Config
from relcompat.data.generate import generate_couples
from relcompat.data.validate import validate
from relcompat.features.build_features import build_features
from relcompat.models import evaluate as ev
from relcompat.models.train import TASKS, make_bundle, train_task
from relcompat.schemas import TARGETS

logger = logging.getLogger(__name__)


def run_pipeline(cfg: Config, make_plots: bool = True) -> dict:
    start = time.perf_counter()
    for path in (cfg.raw_path.parent, cfg.processed_dir, cfg.models_dir, cfg.figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    logger.info("[1/5] Generating %d synthetic couples (seed=%d)", cfg.n_samples, cfg.seed)
    raw = generate_couples(cfg.n_samples, cfg.seed, cfg.missing_rate, cfg.outlier_rate)
    raw.to_csv(cfg.raw_path, index=False)

    logger.info("[2/5] Validating and cleaning")
    raw = validate(pd.read_csv(cfg.raw_path))

    logger.info("[3/5] Building pairwise features")
    X = build_features(raw)
    y = raw[TARGETS]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y["compatible"], random_state=cfg.seed
    )
    X_train.join(y_train).to_csv(cfg.processed_dir / "train.csv", index=False)
    X_test.join(y_test).to_csv(cfg.processed_dir / "test.csv", index=False)
    background = X_train.sample(min(200, len(X_train)), random_state=cfg.seed)

    logger.info("[4/5] Training and model selection")
    summary: dict[str, dict] = {}
    for task, (target, problem) in TASKS.items():
        logger.info("Task %s (%s on %s)", task, problem, target)
        result = train_task(task, X_train, y_train[target], cfg)
        candidate_scores = {
            name: {"cv": result.cv_scores[name], "test": ev.evaluate(model, X_test, y_test[target], problem)}
            for name, model in result.fitted.items()
        }
        test_metrics = candidate_scores[result.best_name]["test"]
        joblib.dump(make_bundle(result, background, y_train[target], test_metrics), cfg.models_dir / f"{task}.joblib")
        summary[task] = {
            "target": target,
            "problem": problem,
            "selected": result.best_name,
            "best_params": result.best_params[result.best_name],
            "test_metrics": test_metrics,
            "candidates": candidate_scores,
        }

        if make_plots:
            plot = ev.plot_classifier if problem == "classification" else ev.plot_regressor
            plot(result.best_model, X_test, y_test[target], task, cfg.figures_dir)
            importance = ev.plot_importance(result.best_model, X_test, background, task, cfg.figures_dir)
            summary[task]["top_features"] = importance.head(10).round(4).to_dict()

    logger.info("[5/5] Writing reports")
    metrics = {
        "n_samples": int(len(raw)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "seed": cfg.seed,
        "elapsed_seconds": round(time.perf_counter() - start, 1),
        "tasks": summary,
    }
    (cfg.models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ev.write_comparison(summary, cfg.reports_dir / "model_comparison.md")
    return metrics
