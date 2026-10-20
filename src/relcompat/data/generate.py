"""Reproducible synthetic dataset of couples with compatibility & longevity labels.

Labels come from a hidden latent "relationship quality" score built from the
pairwise features plus noise, so the models have a real (but imperfect) signal
to learn. The weights below encode common findings from relationship research:
shared values and life goals, similar agreeableness/conscientiousness, low
neuroticism, secure attachment and constructive conflict styles all help.
Gender is generated but deliberately plays no part in the labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from relcompat.features.build_features import build_features
from relcompat.schemas import (
    ATTACHMENT_STYLES,
    BIG_FIVE,
    CHILDREN_PREFS,
    CONFLICT_STYLES,
    DRINKING_LEVELS,
    EDUCATION_LEVELS,
    GENDERS,
    HOBBIES,
    HOW_MET,
    LOVE_LANGUAGES,
    RELIGIONS,
)

# Columns that receive injected missing values (labels are computed beforehand).
MISSABLE_COLUMNS = [
    *[f"{p}_{f}" for p in ("p1", "p2") for f in
      ("income_bracket", "communication_score", "love_language", *BIG_FIVE)],
    "distance_km", "months_dating",
]

ATTACHMENT_EFFECTS = {
    "secure|secure": 0.8,
    "anxious|avoidant": -1.2,
    "avoidant|avoidant": -0.4,
    "anxious|anxious": -0.3,
}
CONFLICT_EFFECTS = {
    "collaborative|collaborative": 0.6,
    "collaborative|compromising": 0.4,
    "compromising|compromising": 0.3,
    "competing|competing": -0.8,
    "avoiding|competing": -0.6,
    "avoiding|avoiding": -0.4,
}


def _choice(rng, options, n, p=None):
    return rng.choice(np.array(options, dtype=object), size=n, p=p)


def _trait(rng, n, mean=5.5, sd=1.8):
    return np.round(np.clip(rng.normal(mean, sd, n), 1, 10), 1)


def _hobbies(rng, n):
    picks = rng.random((n, len(HOBBIES))) < 0.28
    return [";".join(h for h, keep in zip(HOBBIES, row) if keep) for row in picks]


def _correlated_choice(rng, base, options, p_same, n):
    """Copy partner 1's value with probability p_same (assortative mating)."""
    other = _choice(rng, options, n)
    return np.where(rng.random(n) < p_same, base, other)


def _sample_partners(rng, n) -> dict[str, np.ndarray]:
    cols: dict[str, np.ndarray] = {}
    age1 = np.clip(rng.normal(32, 8, n), 18, 75)
    big_gap = rng.random(n) < 0.08
    age2 = np.clip(age1 + rng.normal(0, 4, n) + big_gap * rng.normal(0, 14, n), 18, 80)
    cols["p1_age"], cols["p2_age"] = age1.round().astype(int), age2.round().astype(int)

    religion1 = _choice(rng, RELIGIONS, n, p=[0.3, 0.3, 0.1, 0.12, 0.05, 0.05, 0.08])
    children1 = _choice(rng, CHILDREN_PREFS, n, p=[0.55, 0.2, 0.25])
    politics1 = rng.integers(1, 8, n)
    for prefix in ("p1", "p2"):
        first = prefix == "p1"
        cols[f"{prefix}_gender"] = _choice(rng, GENDERS, n, p=[0.48, 0.48, 0.04])
        cols[f"{prefix}_education"] = _choice(rng, EDUCATION_LEVELS, n, p=[0.3, 0.4, 0.22, 0.08])
        cols[f"{prefix}_income_bracket"] = rng.integers(1, 6, n)
        cols[f"{prefix}_religion"] = (
            religion1 if first else _correlated_choice(rng, religion1, RELIGIONS, 0.5, n)
        )
        cols[f"{prefix}_religiosity"] = rng.integers(1, 6, n)
        cols[f"{prefix}_wants_children"] = (
            children1 if first else _correlated_choice(rng, children1, CHILDREN_PREFS, 0.55, n)
        )
        for trait in BIG_FIVE:
            cols[f"{prefix}_{trait}"] = _trait(rng, n)
        cols[f"{prefix}_love_language"] = _choice(rng, LOVE_LANGUAGES, n)
        cols[f"{prefix}_attachment_style"] = _choice(
            rng, ATTACHMENT_STYLES, n, p=[0.55, 0.2, 0.18, 0.07]
        )
        cols[f"{prefix}_hobbies"] = np.array(_hobbies(rng, n), dtype=object)
        cols[f"{prefix}_communication_score"] = _trait(rng, n, mean=6.0, sd=1.7)
        cols[f"{prefix}_conflict_style"] = _choice(
            rng, CONFLICT_STYLES, n, p=[0.25, 0.3, 0.15, 0.17, 0.13]
        )
        cols[f"{prefix}_political_leaning"] = (
            politics1 if first
            else np.clip(politics1 + np.round(rng.normal(0, 1.8, n)), 1, 7).astype(int)
        )
        cols[f"{prefix}_smoker"] = rng.random(n) < 0.15
        cols[f"{prefix}_drinking"] = _choice(rng, DRINKING_LEVELS, n, p=[0.25, 0.55, 0.2])
    return cols


def latent_quality(features: pd.DataFrame) -> np.ndarray:
    """Noise-free relationship quality score (log-odds scale) from clean features."""
    f = features
    z = (
        -0.08 * f["age_gap"]
        + 0.5 * f["same_religion"]
        - 0.3 * f["religiosity_diff"]
        + np.select([f["children_match"] == 1, f["children_match"] == 0], [0.8, -1.5], 0.0)
        - 0.3 * f["political_distance"]
        - 0.25 * f["agreeableness_diff"]
        - 0.2 * f["conscientiousness_diff"]
        - 0.06 * f["personality_distance"]
        + 0.15 * (f["mean_agreeableness"] - 5.5)
        - 0.12 * (f["combined_neuroticism"] - 11)
        + 1.6 * f["hobby_jaccard"]
        + 0.18 * (f["communication_min"] - 5)
        + 0.3 * f["same_love_language"]
        - 0.6 * f["smoker_mismatch"]
        - 0.3 * f["drinking_diff"]
        - 0.08 * f["log_distance_km"]
    )
    attachment = f["attachment_pair"]
    z += attachment.map(ATTACHMENT_EFFECTS).fillna(0.0)
    z += np.where(attachment.str.contains("secure") & (attachment != "secure|secure"), 0.3, 0.0)
    z += np.where(attachment.str.contains("fearful"), -0.5, 0.0)
    z += f["conflict_pair"].map(CONFLICT_EFFECTS).fillna(0.0)
    return z.to_numpy(dtype=float)


def generate_couples(
    n: int = 10_000,
    seed: int = 42,
    missing_rate: float = 0.03,
    outlier_rate: float = 0.005,
) -> pd.DataFrame:
    """Generate `n` synthetic couples with targets compatible / longevity_years / lasts_5_years."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(_sample_partners(rng, n))
    df["distance_km"] = np.round(rng.lognormal(mean=2.3, sigma=1.3, size=n), 1)
    df["how_met"] = _choice(rng, HOW_MET, n, p=[0.4, 0.25, 0.12, 0.1, 0.13])
    df["months_dating"] = np.clip(rng.gamma(2.0, 9.0, n), 1, 240).round().astype(int)
    df["cohabiting"] = rng.random(n) < (0.15 + 0.6 * np.clip(df["months_dating"] / 48, 0, 1))

    features = build_features(df)
    z = latent_quality(features)
    # Center for class balance, sharpen the logistic link and add noise; with these
    # settings an oracle that knows the noise-free score reaches ROC-AUC ~0.89.
    z = 1.8 * (z - np.median(z)) + rng.normal(0, 0.6, n)
    df["compatible"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    log_years = (
        1.55
        + 0.25 * z
        + 0.012 * (features["mean_age"] - 32)
        + 0.25 * features["cohabiting"]
        + 0.06 * (features["mean_income"] - 3)
        + 0.004 * features["months_dating"]
        + rng.normal(0, 0.3, n)
    )
    df["longevity_years"] = np.clip(np.exp(log_years), 0.1, 40).round(2)
    df["lasts_5_years"] = (df["longevity_years"] >= 5).astype(int)

    # Imperfections for the cleaning stage: missing values and distance outliers.
    for col in MISSABLE_COLUMNS:
        df[col] = df[col].astype(object if df[col].dtype == object else float)
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    outliers = rng.random(n) < outlier_rate
    df.loc[outliers, "distance_km"] = df.loc[outliers, "distance_km"] * 50
    return df
