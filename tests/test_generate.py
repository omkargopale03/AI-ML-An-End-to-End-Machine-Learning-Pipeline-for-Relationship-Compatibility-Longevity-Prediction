import pandas as pd
import pytest

from relcompat.data.generate import MISSABLE_COLUMNS, generate_couples
from relcompat.data.validate import REQUIRED_COLUMNS, DataValidationError, validate


def test_shape_and_columns(small_df):
    assert len(small_df) == 800
    assert set(REQUIRED_COLUMNS) <= set(small_df.columns)


def test_deterministic_for_seed():
    pd.testing.assert_frame_equal(generate_couples(200, seed=1), generate_couples(200, seed=1))
    assert not generate_couples(200, seed=1).equals(generate_couples(200, seed=2))


def test_label_ranges(small_df):
    assert set(small_df["compatible"].unique()) <= {0, 1}
    assert 0.3 < small_df["compatible"].mean() < 0.7
    assert small_df["longevity_years"].between(0, 40).all()
    assert (small_df["lasts_5_years"] == (small_df["longevity_years"] >= 5)).all()


def test_missing_values_injected_only_in_missable_columns(small_df):
    with_missing = set(small_df.columns[small_df.isna().any()])
    assert with_missing
    assert with_missing <= set(MISSABLE_COLUMNS)


def test_validate_drops_out_of_range_rows(small_df):
    bad = small_df.copy()
    bad.loc[0, "p1_age"] = 150
    assert len(validate(bad)) == len(small_df) - 1


def test_validate_rejects_missing_columns(small_df):
    with pytest.raises(DataValidationError):
        validate(small_df.drop(columns=["compatible"]))
