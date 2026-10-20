"""Schema checks and basic cleaning for the raw couples dataset."""

from __future__ import annotations

import logging

import pandas as pd

from relcompat.schemas import COUPLE_FIELDS, PARTNER_FIELDS, TARGETS

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    *[f"{p}_{f}" for p in ("p1", "p2") for f in PARTNER_FIELDS], *COUPLE_FIELDS, *TARGETS,
]
# (min, max) for numeric columns; NaN is allowed and imputed later.
RANGES = {
    **{f"{p}_age": (18, 90) for p in ("p1", "p2")},
    **{f"{p}_income_bracket": (1, 5) for p in ("p1", "p2")},
    **{f"{p}_religiosity": (1, 5) for p in ("p1", "p2")},
    **{f"{p}_political_leaning": (1, 7) for p in ("p1", "p2")},
    "longevity_years": (0, 60),
}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema, drop invalid/duplicate rows and return a cleaned copy."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if df[TARGETS].isna().any().any():
        raise DataValidationError("Target columns contain missing values")
    if not set(df["compatible"].unique()) <= {0, 1}:
        raise DataValidationError("'compatible' must be binary")

    clean = df.drop_duplicates().copy()
    invalid = pd.Series(False, index=clean.index)
    for col, (lo, hi) in RANGES.items():
        values = pd.to_numeric(clean[col], errors="coerce")
        invalid |= values.notna() & ~values.between(lo, hi)
    clean = clean[~invalid]

    dropped = len(df) - len(clean)
    if dropped:
        logger.info("Dropped %d invalid or duplicate rows", dropped)
    missing_share = clean.isna().mean()
    for col, share in missing_share[missing_share > 0].items():
        logger.debug("%s: %.1f%% missing", col, 100 * share)
    return clean.reset_index(drop=True)
