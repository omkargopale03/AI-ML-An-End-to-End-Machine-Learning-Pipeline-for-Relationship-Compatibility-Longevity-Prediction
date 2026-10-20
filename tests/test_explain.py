import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from relcompat.features.build_features import ALL_FEATURES, build_features, make_preprocessor
from relcompat.models.explain import feature_contributions, top_factors
from relcompat.models.train import candidates


def _fit(estimator, X, y, regression=False):
    model = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
    if regression:
        model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return model.fit(X, y)


@pytest.mark.parametrize("problem", ["classification", "regression"])
@pytest.mark.parametrize("name", [0, 1, 2])  # linear, random forest, xgboost
def test_contributions_for_every_candidate(small_df, problem, name):
    X = build_features(small_df)
    y = small_df["compatible" if problem == "classification" else "longevity_years"]
    estimator, _ = list(candidates(problem, seed=0).values())[name]
    if hasattr(estimator, "n_estimators"):
        estimator.set_params(n_estimators=20)
    model = _fit(estimator, X, y, regression=problem == "regression")

    contrib = feature_contributions(model, X.head(10), background=X.head(100))
    assert list(contrib.columns) == ALL_FEATURES
    assert contrib.shape == (10, len(ALL_FEATURES))
    assert np.isfinite(contrib.values).all()

    factors = top_factors(model, X.head(1), background=X.head(100), k=4)
    assert len(factors) == 4
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)
