import pytest

from relcompat.models.predict import Predictor, compatibility_label
from relcompat.schemas import EXAMPLE_COUPLE, PredictionOut


def test_prediction_schema(models_dir):
    result = Predictor(models_dir).predict(EXAMPLE_COUPLE)
    assert isinstance(result, PredictionOut)
    assert 0 <= result.compatibility_score <= 100
    assert result.compatible == (result.compatibility_score >= 50)
    assert result.expected_years >= 0
    assert 0 <= result.lasts_5_years_probability <= 1
    assert len(result.top_factors) == 5
    assert len(result.longevity_factors) == 3
    assert set(result.model_versions) == {"compatibility", "longevity_reg", "longevity_cls"}


def test_mismatched_couple_scores_lower(models_dir):
    predictor = Predictor(models_dir)
    bad = {
        **EXAMPLE_COUPLE,
        "partner_1": {**EXAMPLE_COUPLE["partner_1"], "attachment_style": "anxious",
                      "conflict_style": "avoiding"},
        "partner_2": {**EXAMPLE_COUPLE["partner_2"], "wants_children": "no", "religion": "muslim",
                      "religiosity": 5, "political_leaning": 7, "attachment_style": "avoidant",
                      "hobbies": ["gaming"], "smoker": True, "conflict_style": "competing"},
    }
    assert predictor.predict(bad).compatibility_score < predictor.predict(EXAMPLE_COUPLE).compatibility_score


def test_missing_models_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path)


@pytest.mark.parametrize("score,label", [(10, "Low"), (50, "Moderate"), (80, "High")])
def test_compatibility_label(score, label):
    assert compatibility_label(score) == label
