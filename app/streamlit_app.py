"""Interactive front end: enter two partner profiles, get predictions + explanations.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from relcompat.config import load_config  # noqa: E402
from relcompat.models.predict import Predictor  # noqa: E402
from relcompat.schemas import (  # noqa: E402
    ATTACHMENT_STYLES,
    CHILDREN_PREFS,
    CONFLICT_STYLES,
    DRINKING_LEVELS,
    EDUCATION_LEVELS,
    EXAMPLE_COUPLE,
    GENDERS,
    HOBBIES,
    HOW_MET,
    LOVE_LANGUAGES,
    RELIGIONS,
    CoupleInput,
)

st.set_page_config(page_title="Relationship Compatibility", page_icon="💞", layout="wide")
CFG = load_config()


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor(CFG.models_dir)


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def partner_form(key: str, defaults: dict) -> dict:
    def select(name, options, label=None):
        return st.selectbox(label or _label(name), options, index=options.index(defaults[name]),
                            format_func=_label, key=f"{key}_{name}")

    def slider(name, lo, hi, step=0.5, label=None, help=None):
        return st.slider(label or _label(name), lo, hi, defaults[name], step=step, key=f"{key}_{name}", help=help)

    p = {}
    c1, c2 = st.columns(2)
    with c1:
        p["age"] = st.number_input("Age", 18, 90, defaults["age"], key=f"{key}_age")
        p["gender"] = select("gender", GENDERS)
        p["education"] = select("education", EDUCATION_LEVELS)
        p["income_bracket"] = st.slider("Income bracket", 1, 5, defaults["income_bracket"], key=f"{key}_income")
        p["religion"] = select("religion", RELIGIONS)
        p["religiosity"] = st.slider("Religiosity", 1, 5, defaults["religiosity"], key=f"{key}_religiosity")
        p["wants_children"] = select("wants_children", CHILDREN_PREFS, "Wants children")
        p["political_leaning"] = st.slider(
            "Political leaning", 1, 7, defaults["political_leaning"], key=f"{key}_politics",
            help="1 = very liberal, 7 = very conservative",
        )
    with c2:
        p["love_language"] = select("love_language", LOVE_LANGUAGES)
        p["attachment_style"] = select("attachment_style", ATTACHMENT_STYLES)
        p["conflict_style"] = select("conflict_style", CONFLICT_STYLES)
        p["drinking"] = select("drinking", DRINKING_LEVELS)
        p["smoker"] = st.checkbox("Smoker", defaults["smoker"], key=f"{key}_smoker")
        p["communication_score"] = slider("communication_score", 1.0, 10.0, label="Communication skill")
        p["hobbies"] = st.multiselect("Hobbies", HOBBIES, defaults["hobbies"], format_func=_label,
                                      key=f"{key}_hobbies")
    with st.expander("Big Five personality (1-10)"):
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            p[trait] = slider(trait, 1.0, 10.0)
    return p


def gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d6336c"},
            "steps": [
                {"range": [0, 40], "color": "#f8d7da"},
                {"range": [40, 65], "color": "#fff3cd"},
                {"range": [65, 100], "color": "#d1e7dd"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def factor_chart(factors, title: str) -> go.Figure:
    df = pd.DataFrame([f.model_dump() for f in factors])[::-1]
    colors = ["#2f9e44" if v >= 0 else "#e03131" for v in df["impact"]]
    fig = go.Figure(go.Bar(x=df["impact"], y=df["description"], orientation="h", marker_color=colors,
                           hovertext=[f"value: {v}" for v in df["value"]]))
    fig.update_layout(title=title, height=80 + 45 * len(df), margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="← lowers  |  raises →")
    return fig


st.title("💞 Relationship Compatibility & Longevity Predictor")
st.caption("End-to-end ML pipeline demo. Models are trained on **synthetic data** — "
           "for learning purposes only, not for real relationship decisions.")

try:
    predictor = load_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

predict_tab, metrics_tab, about_tab = st.tabs(["Predict", "Model performance", "About"])

with predict_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Partner 1")
        partner_1 = partner_form("p1", EXAMPLE_COUPLE["partner_1"])
    with right:
        st.subheader("Partner 2")
        partner_2 = partner_form("p2", EXAMPLE_COUPLE["partner_2"])

    st.subheader("The relationship")
    c1, c2, c3, c4 = st.columns(4)
    distance = c1.number_input("Distance between homes (km)", 0.0, 20000.0, float(EXAMPLE_COUPLE["distance_km"]))
    how_met = c2.selectbox("How you met", HOW_MET, index=HOW_MET.index(EXAMPLE_COUPLE["how_met"]),
                           format_func=_label)
    months = c3.number_input("Months dating", 0, 600, EXAMPLE_COUPLE["months_dating"])
    cohabiting = c4.checkbox("Living together", EXAMPLE_COUPLE["cohabiting"])

    if st.button("Predict", type="primary", width="stretch"):
        couple = CoupleInput(partner_1=partner_1, partner_2=partner_2, distance_km=distance,
                             how_met=how_met, months_dating=months, cohabiting=cohabiting)
        result = predictor.predict(couple)

        st.divider()
        g, m = st.columns([1, 1])
        with g:
            st.markdown(f"### Compatibility: **{result.compatibility_label}**")
            st.plotly_chart(gauge(result.compatibility_score), width="stretch")
        with m:
            st.markdown("### Longevity")
            st.metric("Expected duration", f"{result.expected_years:.1f} years")
            st.metric("Chance of lasting 5+ years", f"{100 * result.lasts_5_years_probability:.0f}%")
        f1, f2 = st.columns(2)
        f1.plotly_chart(factor_chart(result.top_factors, "What drives compatibility"), width="stretch")
        f2.plotly_chart(factor_chart(result.longevity_factors, "What drives longevity"), width="stretch")

with metrics_tab:
    metrics = predictor.metrics()
    st.write(f"Trained on {metrics.get('n_train', '?')} couples, evaluated on {metrics.get('n_test', '?')} held-out couples.")
    for task, info in metrics.get("tasks", {}).items():
        st.subheader(f"{task} — selected: `{info['selected']}`")
        rows = {name: {"cv": s["cv"], **s["test"]} for name, s in info["candidates"].items()}
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
        figs = [CFG.figures_dir / f"{task}_{kind}.png" for kind in ("diagnostics", "importance")]
        cols = st.columns(2)
        for col, fig in zip(cols, figs):
            if fig.exists():
                col.image(str(fig))

with about_tab:
    st.markdown("""
**Pipeline:** synthetic data generation → validation & cleaning → pairwise feature engineering →
cross-validated hyperparameter search over Logistic/Ridge, Random Forest and XGBoost →
hold-out evaluation → SHAP-style explanations → FastAPI + Streamlit serving.

**Targets**
- *Compatibility* — binary classifier; the score is the predicted probability × 100.
- *Expected duration* — regression on relationship length in years (fit on log scale).
- *Lasts 5+ years* — binary classifier.

**Factor charts** show each feature's signed contribution to this couple's prediction relative to an
average couple: green bars push the prediction up, red bars push it down.
""")
