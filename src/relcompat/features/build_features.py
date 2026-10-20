"""Turn raw couple records (p1_*/p2_* columns) into symmetric pairwise features.

The same `build_features` function is used for training and for serving, so a
prediction request goes through exactly the transformations the model saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from relcompat.schemas import (
    BIG_FIVE,
    COUPLE_FIELDS,
    DRINKING_LEVELS,
    EDUCATION_LEVELS,
    PARTNER_FIELDS,
    CoupleInput,
)

NUMERIC_FEATURES = [
    "age_gap", "mean_age", "education_diff", "mean_education", "income_diff",
    "mean_income", "religiosity_diff", "same_religion", "children_match",
    *[f"{trait}_diff" for trait in BIG_FIVE], "personality_distance",
    "mean_agreeableness", "mean_conscientiousness", "combined_neuroticism",
    "hobby_jaccard", "shared_hobbies", "political_distance", "communication_mean",
    "communication_min", "same_love_language", "smoker_mismatch", "drinking_diff",
    "lifestyle_mismatch", "log_distance_km", "months_dating", "cohabiting",
]
CATEGORICAL_FEATURES = ["attachment_pair", "conflict_pair", "how_met"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "age_gap": "Age difference between partners",
    "mean_age": "Couple's average age",
    "education_diff": "Difference in education level",
    "mean_education": "Couple's average education level",
    "income_diff": "Difference in income bracket",
    "mean_income": "Couple's average income bracket",
    "religiosity_diff": "Difference in how religious each partner is",
    "same_religion": "Sharing the same religion",
    "children_match": "Agreement on wanting children",
    "openness_diff": "Difference in openness",
    "conscientiousness_diff": "Difference in conscientiousness",
    "extraversion_diff": "Difference in extraversion",
    "agreeableness_diff": "Difference in agreeableness",
    "neuroticism_diff": "Difference in neuroticism",
    "personality_distance": "Overall personality (Big Five) distance",
    "mean_agreeableness": "Couple's average agreeableness",
    "mean_conscientiousness": "Couple's average conscientiousness",
    "combined_neuroticism": "Combined neuroticism of both partners",
    "hobby_jaccard": "Overlap in hobbies",
    "shared_hobbies": "Number of shared hobbies",
    "political_distance": "Distance in political views",
    "communication_mean": "Average communication skill",
    "communication_min": "Weaker partner's communication skill",
    "same_love_language": "Sharing the same love language",
    "smoker_mismatch": "Only one partner smokes",
    "drinking_diff": "Difference in drinking habits",
    "lifestyle_mismatch": "Number of lifestyle mismatches",
    "log_distance_km": "Geographic distance between partners",
    "months_dating": "Months spent dating",
    "cohabiting": "Living together",
    "attachment_pair": "Combination of attachment styles",
    "conflict_pair": "Combination of conflict-resolution styles",
    "how_met": "How the couple met",
}


def _pair(a: pd.Series, b: pd.Series) -> pd.Series:
    """Order-independent label for two categorical values, e.g. 'anxious|secure'."""
    a, b = a.fillna("unknown").astype(str), b.fillna("unknown").astype(str)
    return pd.Series(
        ["|".join(sorted(pair)) for pair in zip(a, b)], index=a.index, dtype="object"
    )


def _hobby_set(value) -> frozenset:
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(value)
    if isinstance(value, str) and value:
        return frozenset(value.split(";"))
    return frozenset()


def _children_match(a: pd.Series, b: pd.Series) -> pd.Series:
    def score(x, y):
        if not isinstance(x, str) or not isinstance(y, str):
            return np.nan
        if "unsure" in (x, y):
            return 0.5
        return 1.0 if x == y else 0.0

    return pd.Series([score(x, y) for x, y in zip(a, b)], index=a.index, dtype=float)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the model feature table from raw couple records."""
    p1 = lambda f: raw[f"p1_{f}"]  # noqa: E731
    p2 = lambda f: raw[f"p2_{f}"]  # noqa: E731
    num = lambda s: pd.to_numeric(s, errors="coerce").astype(float)  # noqa: E731
    edu_rank = {level: i for i, level in enumerate(EDUCATION_LEVELS)}
    drink_rank = {level: i for i, level in enumerate(DRINKING_LEVELS)}

    out = pd.DataFrame(index=raw.index)
    out["age_gap"] = (num(p1("age")) - num(p2("age"))).abs()
    out["mean_age"] = (num(p1("age")) + num(p2("age"))) / 2
    e1, e2 = p1("education").map(edu_rank), p2("education").map(edu_rank)
    out["education_diff"] = (e1 - e2).abs()
    out["mean_education"] = (e1 + e2) / 2
    out["income_diff"] = (num(p1("income_bracket")) - num(p2("income_bracket"))).abs()
    out["mean_income"] = (num(p1("income_bracket")) + num(p2("income_bracket"))) / 2
    out["religiosity_diff"] = (num(p1("religiosity")) - num(p2("religiosity"))).abs()
    out["same_religion"] = (p1("religion") == p2("religion")).astype(float)
    out["children_match"] = _children_match(p1("wants_children"), p2("wants_children"))

    diffs = []
    for trait in BIG_FIVE:
        diff = (num(p1(trait)) - num(p2(trait))).abs()
        out[f"{trait}_diff"] = diff
        diffs.append(diff)
    # min_count keeps the distance NaN (to be imputed) when every trait is missing.
    out["personality_distance"] = np.sqrt(
        pd.concat(diffs, axis=1).pow(2).sum(axis=1, min_count=1)
    )
    out["mean_agreeableness"] = (num(p1("agreeableness")) + num(p2("agreeableness"))) / 2
    out["mean_conscientiousness"] = (
        num(p1("conscientiousness")) + num(p2("conscientiousness"))
    ) / 2
    out["combined_neuroticism"] = num(p1("neuroticism")) + num(p2("neuroticism"))

    h1, h2 = p1("hobbies").map(_hobby_set), p2("hobbies").map(_hobby_set)
    shared = [len(a & b) for a, b in zip(h1, h2)]
    union = [len(a | b) for a, b in zip(h1, h2)]
    out["shared_hobbies"] = np.array(shared, dtype=float)
    out["hobby_jaccard"] = [s / u if u else 0.0 for s, u in zip(shared, union)]

    out["political_distance"] = (
        num(p1("political_leaning")) - num(p2("political_leaning"))
    ).abs()
    c1, c2 = num(p1("communication_score")), num(p2("communication_score"))
    out["communication_mean"] = (c1 + c2) / 2
    out["communication_min"] = np.fmin(c1, c2)
    out["same_love_language"] = (p1("love_language") == p2("love_language")).astype(float)

    smoker_mismatch = (p1("smoker").astype(bool) != p2("smoker").astype(bool)).astype(float)
    drinking_diff = (p1("drinking").map(drink_rank) - p2("drinking").map(drink_rank)).abs()
    out["smoker_mismatch"] = smoker_mismatch
    out["drinking_diff"] = drinking_diff
    out["lifestyle_mismatch"] = smoker_mismatch + (drinking_diff.fillna(0) >= 2)

    # log1p compresses the long tail of long-distance (and outlier) couples.
    out["log_distance_km"] = np.log1p(num(raw["distance_km"]).clip(lower=0))
    out["months_dating"] = num(raw["months_dating"])
    out["cohabiting"] = raw["cohabiting"].astype(bool).astype(float)

    out["attachment_pair"] = _pair(p1("attachment_style"), p2("attachment_style"))
    out["conflict_pair"] = _pair(p1("conflict_style"), p2("conflict_style"))
    out["how_met"] = raw["how_met"].fillna("unknown").astype(str)
    return out[ALL_FEATURES]


def couple_to_frame(couple: CoupleInput) -> pd.DataFrame:
    """Flatten a validated CoupleInput into a one-row raw dataframe."""
    row = {}
    for prefix, partner in (("p1", couple.partner_1), ("p2", couple.partner_2)):
        data = partner.model_dump()
        for field in PARTNER_FIELDS:
            value = data[field]
            row[f"{prefix}_{field}"] = ";".join(value) if field == "hobbies" else value
    for field in COUPLE_FIELDS:
        row[field] = getattr(couple, field)
    return pd.DataFrame([row])


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
