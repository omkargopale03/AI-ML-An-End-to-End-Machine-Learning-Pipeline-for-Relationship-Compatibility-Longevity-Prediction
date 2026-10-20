"""Hold-out metrics, diagnostic plots and the model comparison report."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from relcompat.features.build_features import CATEGORICAL_FEATURES, FEATURE_DESCRIPTIONS  # noqa: E402
from relcompat.models.explain import feature_contributions  # noqa: E402

logger = logging.getLogger(__name__)


def classification_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def regression_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": root_mean_squared_error(y, pred),
        "r2": r2_score(y, pred),
    }


def evaluate(model, X, y, problem: str) -> dict[str, float]:
    fn = classification_metrics if problem == "classification" else regression_metrics
    return {k: round(float(v), 4) for k, v in fn(model, X, y).items()}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_classifier(model, X, y, task: str, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{task}: confusion matrix")
    RocCurveDisplay.from_estimator(model, X, y, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    axes[1].set_title(f"{task}: ROC curve")
    _save(fig, fig_dir / f"{task}_diagnostics.png")


def plot_regressor(model, X, y, task: str, fig_dir: Path) -> None:
    pred = model.predict(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y, pred, s=6, alpha=0.35)
    lim = [0, max(np.max(y), np.max(pred))]
    axes[0].plot(lim, lim, "--", color="grey", linewidth=1)
    axes[0].set(xlabel="Actual years", ylabel="Predicted years", title=f"{task}: predicted vs actual")
    axes[1].scatter(pred, y - pred, s=6, alpha=0.35)
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted years", ylabel="Residual", title=f"{task}: residuals")
    _save(fig, fig_dir / f"{task}_diagnostics.png")


def plot_importance(model, X, background, task: str, fig_dir: Path, n_rows: int = 800) -> pd.Series:
    """Mean |contribution| per feature (global SHAP importance) plus a beeswarm plot."""
    sample = X.sample(min(n_rows, len(X)), random_state=0)
    contrib = feature_contributions(model, sample, background)
    importance = contrib.abs().mean().sort_values(ascending=False)

    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([FEATURE_DESCRIPTIONS.get(f, f) for f in top.index], top.values, color="#4C72B0")
    ax.set(xlabel="Mean |contribution|", title=f"{task}: global feature importance")
    _save(fig, fig_dir / f"{task}_importance.png")

    try:
        import shap

        coded = sample.copy()
        for col in CATEGORICAL_FEATURES:
            coded[col] = coded[col].astype("category").cat.codes
        shap.summary_plot(contrib.values, coded, show=False, max_display=15)
        fig = plt.gcf()
        fig.suptitle(f"{task}: SHAP summary", y=1.0)
        _save(fig, fig_dir / f"{task}_shap_summary.png")
    except Exception as exc:  # plotting is a nice-to-have
        logger.warning("SHAP summary plot skipped for %s: %s", task, exc)
    return importance


def write_comparison(results: dict[str, dict], path: Path) -> None:
    """Markdown table of CV and test scores for every candidate model."""
    lines = ["# Model comparison", ""]
    for task, info in results.items():
        lines += [f"## {task} ({info['problem']}, target `{info['target']}`)", ""]
        metric_names = list(next(iter(info["candidates"].values()))["test"].keys())
        cv_label = "CV ROC-AUC" if info["problem"] == "classification" else "CV RMSE"
        lines.append("| model | " + " | ".join([cv_label, *metric_names]) + " |")
        lines.append("|" + "---|" * (len(metric_names) + 2))
        for name, scores in info["candidates"].items():
            marker = " **(selected)**" if name == info["selected"] else ""
            cells = [f"{scores['cv']:.4f}", *(f"{scores['test'][m]:.4f}" for m in metric_names)]
            lines.append(f"| {name}{marker} | " + " | ".join(cells) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
