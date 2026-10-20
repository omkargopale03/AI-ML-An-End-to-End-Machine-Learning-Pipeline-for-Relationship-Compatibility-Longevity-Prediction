import numpy as np
import pandas as pd

from relcompat.features.build_features import (
    ALL_FEATURES,
    build_features,
    couple_to_frame,
    make_preprocessor,
)
from relcompat.schemas import EXAMPLE_COUPLE, CoupleInput


def _swap_partners(df: pd.DataFrame) -> pd.DataFrame:
    swap = {"p1_": "p2_", "p2_": "p1_"}
    return df.rename(columns={c: swap[c[:3]] + c[3:] for c in df.columns if c[:3] in swap})


def test_feature_columns(small_df):
    X = build_features(small_df)
    assert list(X.columns) == ALL_FEATURES
    assert len(X) == len(small_df)


def test_features_are_symmetric(small_df):
    pd.testing.assert_frame_equal(build_features(small_df), build_features(_swap_partners(small_df)))


def test_preprocessor_output_has_no_nans(small_df):
    X = build_features(small_df)
    assert X.isna().any().any()  # injected missing values propagate...
    Xt = make_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()  # ...and are imputed


def test_couple_to_frame_matches_training_format():
    couple = CoupleInput.model_validate(EXAMPLE_COUPLE)
    row = build_features(couple_to_frame(couple)).iloc[0]
    assert row["age_gap"] == 2
    assert row["shared_hobbies"] == 2
    assert row["hobby_jaccard"] == 0.5
    assert row["attachment_pair"] == "anxious|secure"
    assert row["children_match"] == 0.5
