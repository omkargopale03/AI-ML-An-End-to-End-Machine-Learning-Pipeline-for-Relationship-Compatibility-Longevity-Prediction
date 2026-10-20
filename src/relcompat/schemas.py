"""Domain vocabulary and pydantic models shared by the pipeline, API and app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GENDERS = ["male", "female", "non_binary"]
EDUCATION_LEVELS = ["high_school", "bachelors", "masters", "phd"]
RELIGIONS = ["none", "christian", "muslim", "hindu", "jewish", "buddhist", "other"]
CHILDREN_PREFS = ["yes", "no", "unsure"]
LOVE_LANGUAGES = ["words", "acts", "gifts", "time", "touch"]
ATTACHMENT_STYLES = ["secure", "anxious", "avoidant", "fearful"]
CONFLICT_STYLES = ["collaborative", "compromising", "accommodating", "avoiding", "competing"]
DRINKING_LEVELS = ["never", "social", "regular"]
HOW_MET = ["online", "friends", "work", "school", "other"]
HOBBIES = [
    "travel", "music", "sports", "reading", "gaming", "cooking",
    "art", "outdoors", "movies", "fitness", "tech", "volunteering",
]
BIG_FIVE = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Per-partner columns in the raw dataset, stored as p1_<field> / p2_<field>.
PARTNER_FIELDS = [
    "age", "gender", "education", "income_bracket", "religion", "religiosity",
    "wants_children", *BIG_FIVE, "love_language", "attachment_style", "hobbies",
    "communication_score", "conflict_style", "political_leaning", "smoker", "drinking",
]
COUPLE_FIELDS = ["distance_km", "how_met", "months_dating", "cohabiting"]
TARGETS = ["compatible", "longevity_years", "lasts_5_years"]

Trait = Field(ge=1, le=10, description="Score on a 1-10 scale")


class PartnerProfile(BaseModel):
    age: int = Field(ge=18, le=90)
    gender: Literal["male", "female", "non_binary"]
    education: Literal["high_school", "bachelors", "masters", "phd"]
    income_bracket: int = Field(ge=1, le=5, description="1 = lowest, 5 = highest")
    religion: Literal["none", "christian", "muslim", "hindu", "jewish", "buddhist", "other"]
    religiosity: int = Field(ge=1, le=5, description="1 = not religious, 5 = very devout")
    wants_children: Literal["yes", "no", "unsure"]
    openness: float = Trait
    conscientiousness: float = Trait
    extraversion: float = Trait
    agreeableness: float = Trait
    neuroticism: float = Trait
    love_language: Literal["words", "acts", "gifts", "time", "touch"]
    attachment_style: Literal["secure", "anxious", "avoidant", "fearful"]
    hobbies: list[
        Literal["travel", "music", "sports", "reading", "gaming", "cooking",
                "art", "outdoors", "movies", "fitness", "tech", "volunteering"]
    ] = Field(default_factory=list)
    communication_score: float = Trait
    conflict_style: Literal["collaborative", "compromising", "accommodating", "avoiding", "competing"]
    political_leaning: int = Field(ge=1, le=7, description="1 = very liberal, 7 = very conservative")
    smoker: bool = False
    drinking: Literal["never", "social", "regular"]


class CoupleInput(BaseModel):
    # Resolved lazily so the example can live at the bottom of the module.
    model_config = ConfigDict(json_schema_extra=lambda schema: schema.update(examples=[EXAMPLE_COUPLE]))

    partner_1: PartnerProfile
    partner_2: PartnerProfile
    distance_km: float = Field(ge=0, le=20_000, description="Distance between homes in km")
    how_met: Literal["online", "friends", "work", "school", "other"]
    months_dating: int = Field(ge=0, le=600)
    cohabiting: bool = False


class Factor(BaseModel):
    feature: str
    description: str
    value: float | str | None
    impact: float = Field(description="Signed contribution to the model output")
    effect: Literal["increases", "decreases"]


class PredictionOut(BaseModel):
    compatibility_score: float = Field(ge=0, le=100)
    compatible: bool
    compatibility_label: Literal["Low", "Moderate", "High"]
    expected_years: float = Field(ge=0)
    lasts_5_years_probability: float = Field(ge=0, le=1)
    top_factors: list[Factor]
    longevity_factors: list[Factor]
    model_versions: dict[str, str]


EXAMPLE_COUPLE: dict = {
    "partner_1": {
        "age": 29, "gender": "female", "education": "masters", "income_bracket": 3,
        "religion": "none", "religiosity": 1, "wants_children": "yes",
        "openness": 7.5, "conscientiousness": 7.0, "extraversion": 5.5,
        "agreeableness": 8.0, "neuroticism": 3.5, "love_language": "time",
        "attachment_style": "secure", "hobbies": ["travel", "reading", "cooking"],
        "communication_score": 8.0, "conflict_style": "collaborative",
        "political_leaning": 3, "smoker": False, "drinking": "social",
    },
    "partner_2": {
        "age": 31, "gender": "male", "education": "bachelors", "income_bracket": 3,
        "religion": "none", "religiosity": 2, "wants_children": "unsure",
        "openness": 6.5, "conscientiousness": 6.5, "extraversion": 6.0,
        "agreeableness": 6.0, "neuroticism": 6.0, "love_language": "acts",
        "attachment_style": "anxious", "hobbies": ["travel", "cooking", "outdoors"],
        "communication_score": 7.0, "conflict_style": "compromising",
        "political_leaning": 5, "smoker": False, "drinking": "regular",
    },
    "distance_km": 5, "how_met": "friends", "months_dating": 18, "cohabiting": True,
}
