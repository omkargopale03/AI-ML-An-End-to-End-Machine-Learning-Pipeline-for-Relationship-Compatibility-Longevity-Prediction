"""Per-feature contributions (SHAP-style) for the fitted model pipelines.

Contributions are computed on the transformed (scaled / one-hot) matrix and then
summed back onto the engineered feature names, so a one-hot encoded
`attachment_pair` shows up as a single factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBClassifier, XGBRegressor, DMatrix

from relcompat.features.build_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
)


def _unwrap(model):
    """Return (preprocessor, estimator) from a Pipeline, optionally target-wrapped."""
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    return model.named_steps["preprocess"], model.named_steps["model"]


def _source_feature(column: str) -> str:
    kind, name = column.split("__", 1)
    if kind == "num":
        return name
    return next(f for f in CATEGORICAL_FEATURES if name.startswith(f"{f}_"))


def feature_contributions(
    model, X: pd.DataFrame, background: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Signed contribution of each engineered feature to each row's raw model output."""
    pre, est = _unwrap(model)
    Xt = pre.transform(X)
    columns = pre.get_feature_names_out()

    if isinstance(est, (XGBClassifier, XGBRegressor)):
        values = est.get_booster().predict(DMatrix(Xt), pred_contribs=True)[:, :-1]
    elif hasattr(est, "coef_"):
        # Linear model: contribution = coef * (x - background mean).
        ref = pre.transform(background).mean(axis=0) if background is not None else 0.0
        values = (Xt - ref) * np.ravel(est.coef_)
    else:
        import shap

        values = shap.TreeExplainer(est).shap_values(Xt, check_additivity=False)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes) -> positive class
            values = values[..., 1]

    contrib = pd.DataFrame(values, columns=columns, index=X.index)
    grouped = contrib.T.groupby([_source_feature(c) for c in columns]).sum().T
    return grouped.reindex(columns=ALL_FEATURES, fill_value=0.0)


def _display_value(value):
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 2)
    return value


def top_factors(model, X_row: pd.DataFrame, background=None, k: int = 5) -> list[dict]:
    """The k features with the largest absolute contribution for a single row."""
    contrib = feature_contributions(model, X_row, background).iloc[0]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:k])
    return [
        {
            "feature": feature,
            "description": FEATURE_DESCRIPTIONS.get(feature, feature),
            "value": _display_value(X_row.iloc[0][feature]),
            "impact": round(float(impact), 4),
            "effect": "increases" if impact >= 0 else "decreases",
        }
        for feature, impact in top.items()
    ]
