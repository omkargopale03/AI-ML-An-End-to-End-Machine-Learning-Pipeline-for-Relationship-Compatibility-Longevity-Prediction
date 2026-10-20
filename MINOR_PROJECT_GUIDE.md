# Minor Project Guide: Relationship Compatibility & Longevity Prediction

This guide is based on the actual code, config, notebook and result files in this repository. When something could not be found in the project, it says **"Not found"**.

- **GitHub repo:** https://github.com/omkargopale03/AI-ML-An-End-to-End-Machine-Learning-Pipeline-for-Relationship-Compatibility-Longevity-Prediction
- **Note on freshness:** the metric numbers come from `models/metrics.json` and `reports/model_comparison.md`, which were produced by the last full pipeline run. The test suite and the live example prediction were **not re-run** while this guide was written. Run `python -m pytest -q` before recording to confirm that the tests pass.

---

## 1. BASICS

**Full project title:**
An End-to-End Machine Learning Pipeline for Relationship Compatibility & Longevity Prediction (labelled "Mini Project 2 — AI/ML" in the README).

**Summary:**
The project takes the profiles of two partners (age, education, religion, personality, hobbies, habits and more) plus some facts about the couple (distance, how they met, months dating, living together). It then predicts three things: (1) a **compatibility score from 0 to 100**, (2) the **expected relationship length in years**, and (3) the **probability that the relationship lasts 5 or more years**. Each prediction comes with an explanation: the top factors that pushed the score up or down. The full pipeline covers creating the data, cleaning it, building features, training and comparing three model types, evaluating them, explaining them, and serving the best models through a **FastAPI REST API** and a **Streamlit web app**. All data is **synthetic** (computer-generated).

**Problem it solves and who benefits:**
- The project shows how to build a complete, reproducible ML system, not just a single model in a notebook. That covers data generation, validation, feature engineering, model selection, evaluation, explainability, an API, a UI, tests and Docker.
- **Main beneficiaries:** students and learners of ML engineering, and examiners or reviewers who want to see good ML practice from start to end.
- **Not for real couples:** the README states clearly that the data is synthetic, that the label rules are simplified assumptions, and that predictions "should never be used to judge or advise real couples."

**Why this approach or technology was chosen (from the code and README):**
- **Synthetic data with a known hidden rule:** the true pattern is known, so we can check that the model learns it. The SHAP importance plot recovers the rules hidden in the generator.
- **Symmetric pairwise features:** compatibility is about the *pair*, so features like "age gap" or "hobby overlap" are used instead of raw individual values. Swapping partner 1 and partner 2 gives exactly the same features.
- **The same feature function for training and serving:** `build_features` is used in both places, so the API cannot drift away from what the model learned.
- **Three model families compared fairly:** Logistic/Ridge regression, Random Forest and XGBoost are each tuned with cross-validation, and the winner is picked by CV score, not test score.
- **Explainability:** each prediction shows its top factors, so the output is not a "black box."
- **FastAPI + Streamlit:** FastAPI gives a REST API with automatic input validation and docs. Streamlit gives a quick interactive web UI in pure Python.

---

## 2. TOOLS AND TECHNOLOGIES

Versions marked "installed" are the ones found in the local Python environment. `requirements.txt` gives only minimum versions (shown in brackets).

| Category | Tool or library (version) | What it is used for |
|---|---|---|
| Language | Python 3.13.5 installed (project needs ≥3.10 per `pyproject.toml`; Docker image uses 3.11-slim) | Whole project |
| Data handling | pandas 2.3.3 (≥2.1) | Tables of couples, features, CSV read/write |
| Numerics | numpy 2.4.1 (≥1.26) | Random data generation, math |
| Statistics | scipy 1.17.1 (≥1.11) | `loguniform` distribution for hyperparameter search |
| Machine learning | scikit-learn 1.9.1 (≥1.5) | Logistic Regression, Ridge, Random Forest, pipelines, imputation, scaling, one-hot encoding, cross-validation, RandomizedSearchCV, metrics |
| Machine learning | xgboost 3.4.1 (≥2.0) | XGBoost classifier/regressor candidates and their exact feature contributions |
| Explainability | shap 0.54.1 (≥0.45) | TreeExplainer for Random Forest; SHAP summary (beeswarm) plots |
| Plots | matplotlib 3.11.2 (≥3.8) | Confusion matrix, ROC, residual and importance plots |
| Plots | seaborn 0.13.2 (≥0.13) | Charts in the EDA notebook |
| Interactive charts | plotly 7.1.0 (≥5.20) | Gauge and factor bar charts in the Streamlit app |
| Config | PyYAML 6.0.3 (≥6.0) | Reads `config/config.yaml` |
| Data validation | pydantic 2.12.5 (≥2.5) | Input/output schemas (`CoupleInput`, `PredictionOut`) with range checks |
| Model saving | joblib 1.6.0 (≥1.3) | Saves and loads trained model bundles (`.joblib`) |
| API framework | FastAPI 0.135.1 (≥0.110) | REST API (`/health`, `/model-info`, `/predict`) |
| API server | uvicorn 0.42.0 (≥0.29) | Runs the FastAPI app |
| Web UI | Streamlit 1.54.0 (≥1.40) | Interactive web app |
| Testing | pytest 9.1.1 (≥8.0) | 27 automated tests |
| Testing | httpx 0.28.1 (≥0.27) | Needed by FastAPI's `TestClient` in API tests |
| Notebook | jupyter (≥1.0), nbconvert 7.17.1 (≥7.0) | EDA notebook; `make notebook` re-executes it |
| Packaging | setuptools (≥68) via `pyproject.toml` | Installs `relcompat` as a package (`pip install -e .`) |
| Containers | Docker (`Dockerfile`), Docker Compose (`docker-compose.yml`) | Builds an image with trained models; runs API + app together |
| Automation | Make (`Makefile`) | Short commands: install, train, test, api, app, etc. |
| Version control | Git + GitHub | Source hosting |
| Database | Not found | Data is stored as CSV files; models as `.joblib` files |
| External APIs / keys | Not found | No API keys or outside services are used |
| Cloud hosting | Not found | Runs locally or in Docker only |
| CI/CD | Not found | No GitHub Actions or other CI config |

---

## 3. PROJECT STRUCTURE

| File / folder | What it does |
|---|---|
| `README.md` | Project overview, architecture diagram, results, how to run, ethics |
| `requirements.txt` | List of Python libraries to install |
| `pyproject.toml` | Package definition (`relcompat` v1.0.0) and pytest settings |
| `config/config.yaml` | Settings: seed 42, 10,000 samples, 3% missing, 0.5% outliers, 20% test, 5 CV folds, 8 search iterations, paths |
| `run_pipeline.py` | Command-line entry point that runs the full training pipeline |
| `src/relcompat/config.py` | Loads the YAML into a `Config` object; supports overrides |
| `src/relcompat/schemas.py` | Allowed values (religions, hobbies, etc.), pydantic input/output models, and the built-in `EXAMPLE_COUPLE` |
| `src/relcompat/pipeline.py` | Runs all stages in order: generate → validate → features → split → train → evaluate → save |
| `src/relcompat/data/generate.py` | Creates the synthetic couples dataset and its labels |
| `src/relcompat/data/validate.py` | Checks columns and ranges; drops duplicate and invalid rows |
| `src/relcompat/features/build_features.py` | Turns two partner profiles into 33 pairwise features; builds the preprocessing step |
| `src/relcompat/models/train.py` | Candidate models, hyperparameter search, picks the best model, builds the saved bundle |
| `src/relcompat/models/evaluate.py` | Computes metrics, draws plots, writes the model comparison report |
| `src/relcompat/models/explain.py` | Computes each feature's contribution to a prediction; returns the top factors |
| `src/relcompat/models/predict.py` | `Predictor` class: loads the 3 models and scores a couple |
| `api/main.py` | FastAPI REST service |
| `app/streamlit_app.py` | Streamlit web app (Predict / Model performance / About tabs) |
| `notebooks/01_eda.ipynb` | Exploratory data analysis with charts and takeaways |
| `tests/` | 27 pytest tests (data, features, explanations, prediction, API) |
| `data/raw/couples.csv` | Generated raw data (10,000 rows × 47 columns, ~2.8 MB) |
| `data/processed/train.csv`, `test.csv` | Feature tables + targets after the split (8,000 / 2,000 rows) |
| `models/*.joblib` | Trained models: `compatibility`, `longevity_reg`, `longevity_cls` |
| `models/metrics.json` | All metrics, chosen models, best parameters, top features |
| `reports/model_comparison.md` | Table comparing all candidate models |
| `reports/figures/*.png` | 9 figures: diagnostics, importance and SHAP summary for each of the 3 tasks |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Container build and run setup |
| `Makefile` | Shortcut commands |
| `.gitignore` | Keeps generated data, models and figures out of Git |

**Step-by-step flow (training):**
1. `run_pipeline.py` reads `config/config.yaml` (plus any command-line overrides).
2. `generate_couples()` creates 10,000 synthetic couples and saves them to `data/raw/couples.csv`.
3. `validate()` reads the CSV back, checks the schema and ranges, and removes bad or duplicate rows.
4. `build_features()` turns each couple into 33 pairwise features.
5. The data is split 80/20 (stratified on `compatible`) into 8,000 training and 2,000 test couples, which are saved to `data/processed/`.
6. For each of the 3 tasks, `train_task()` tunes 3 candidate models with 5-fold cross-validation and keeps the best one by CV score.
7. Every candidate is scored on the test set. Plots and SHAP figures are saved to `reports/figures/`.
8. The best model of each task is saved as a `.joblib` bundle. `metrics.json` and `model_comparison.md` are written.

**Step-by-step flow (prediction request):**
1. The user fills in the form in Streamlit or sends JSON to `POST /predict`.
2. Pydantic (`CoupleInput`) checks the input, e.g. age 18–90 and traits 1–10. Bad input gets error 422.
3. `couple_to_frame()` flattens it into one row. `build_features()` makes the same 33 features used in training.
4. `Predictor.predict()` runs the 3 models and computes the top factors with `top_factors()`.
5. A `PredictionOut` is returned: score, label, years, 5-year probability, top factors and model versions.

---

## 4. HOW IT WORKS (METHODS)

### 4.1 Data (input)
- **Source:** synthetic, generated by `src/relcompat/data/generate.py`. No real people's data is used.
- **Size:** 10,000 couples (set in `config.yaml`), 47 columns. That is 20 fields per partner (×2 = 40), 4 couple fields and 3 targets. File: `data/raw/couples.csv` (CSV, ~2.8 MB).
- **Per-partner fields:** age, gender, education, income bracket (1–5), religion, religiosity (1–5), wants children (yes/no/unsure), Big Five personality (openness, conscientiousness, extraversion, agreeableness, neuroticism, each 1–10), love language, attachment style, hobbies (from 12), communication score (1–10), conflict style, political leaning (1–7), smoker, drinking.
- **Couple fields:** distance in km, how they met, months dating, cohabiting.
- **Targets:** `compatible` (0/1), `longevity_years` (number), `lasts_5_years` (0/1, equal to years ≥ 5).

**How the data is generated (key ideas):**
- **Realistic pairing:** partner 2 copies partner 1's religion 50% of the time and children preference 55% of the time. Political views are close, and ages are close, with an 8% chance of a big gap. This mimics "assortative mating."
- **Hidden "relationship quality" score** (`latent_quality()`): a weighted sum of the pairwise features. Weights include +0.8 for agreeing on children and −1.5 for disagreeing, +1.6 × hobby overlap, +0.5 for the same religion, −0.3 × political distance, and −0.6 if only one partner smokes. There are also bonuses and penalties for attachment pairs (secure+secure +0.8, anxious+avoidant −1.2) and conflict-style pairs (competing+competing −0.8).
- The score is centred, sharpened (×1.8) and given random noise. `compatible` is then drawn with a logistic (sigmoid) probability. The code notes that a perfect "oracle" would reach only about 0.89 ROC-AUC.
- `longevity_years` = exp(1.55 + 0.25 × quality + small effects of age, cohabiting, income and months dating + noise), clipped to 0.1–40 years.
- **Gender is generated but does not affect the labels at all.**
- **Deliberate imperfections:** about 3% missing values in selected columns, and 0.5% of rows get a distance 50× too large (outliers). These test the cleaning steps.

### 4.2 Cleaning and validation (`data/validate.py`)
- Stops with an error if required columns are missing, if a target has missing values, or if `compatible` is not 0/1.
- Removes duplicate rows.
- Removes rows outside these ranges: ages 18–90, income 1–5, religiosity 1–5, politics 1–7, longevity 0–60. Missing values are allowed here and filled in later.
- Result on the real run: **0 rows dropped** (from the EDA notebook).

### 4.3 Feature engineering (`features/build_features.py`)
There are 33 features, and all of them are **symmetric** (the order of the partners does not matter):
- **30 numeric:** age gap, mean age, education difference and mean, income difference and mean, religiosity difference, same religion, children match (1 = agree, 0 = disagree, 0.5 if either is unsure), the 5 personality differences, overall personality distance, mean agreeableness, mean conscientiousness, combined neuroticism, hobby Jaccard overlap, number of shared hobbies, political distance, mean and minimum communication, same love language, smoker mismatch, drinking difference, lifestyle mismatch count, log of distance (`log1p`, which shrinks outliers), months dating, cohabiting.
- **3 categorical:** attachment pair (e.g. "anxious|secure"), conflict pair, how they met. Pairs are sorted alphabetically so the order does not matter.

### 4.4 Preprocessing (`make_preprocessor()`)
- Numeric columns: fill missing values with the **median**, then **standard scale**.
- Categorical columns: fill missing values with "unknown", then **one-hot encode** (unknown categories are ignored).
- This step sits *inside* each model pipeline. It is therefore fitted only on training folds, which avoids data leakage.

### 4.5 Models and training (`models/train.py`)
There are three tasks:

| Task | Target | Type | Candidates |
|---|---|---|---|
| `compatibility` | `compatible` | Classification | Logistic Regression, Random Forest, XGBoost |
| `longevity_reg` | `longevity_years` | Regression | Ridge, Random Forest, XGBoost |
| `longevity_cls` | `lasts_5_years` | Classification | Logistic Regression, Random Forest, XGBoost |

- **Tuning:** `RandomizedSearchCV` with 8 random settings per candidate and 5-fold CV (stratified for classification).
- **Search spaces:** C or alpha drawn from a log-uniform range for the linear models. For Random Forest: number of trees, depth, minimum leaf size and max features. For XGBoost: number of trees, depth, learning rate, subsample, colsample and min child weight.
- **Scoring:** ROC-AUC for classification, RMSE for regression.
- **Selection:** the model with the best **cross-validation** score wins. The test set is not used for choosing, which keeps it a fair final check.
- **Regression on a log scale:** relationship durations are right-skewed, so the regressors learn `log1p(years)` through `TransformedTargetRegressor` and convert back with `expm1`.
- **Saved bundle:** each `.joblib` holds the model, its name, feature lists, 200 "background" training rows (used for explanations), the range of the training target (predictions are clamped to it), CV score, best parameters, test metrics, version and training time.

### 4.6 Evaluation (`models/evaluate.py`)
- Classification: accuracy, precision, recall, F1 (at a 0.5 threshold) and ROC-AUC, plus confusion matrix and ROC curve plots.
- Regression: MAE, RMSE and R², plus predicted-vs-actual and residual plots.
- Global importance: the mean absolute contribution of each feature over 800 test rows, saved as a bar chart and a SHAP beeswarm plot.

### 4.7 Explainability (`models/explain.py`)
- **XGBoost:** exact contributions from XGBoost's built-in `pred_contribs`.
- **Random Forest:** SHAP `TreeExplainer`.
- **Linear models:** contribution = coefficient × (value − average of the background rows).
- One-hot columns are added back together, so "attachment_pair" shows as one factor. `top_factors()` returns the k biggest contributions with a plain-English description and "increases" or "decreases".

### 4.8 Serving (`models/predict.py`, `api/main.py`, `app/streamlit_app.py`)
- `Predictor` loads the 3 bundles once.
- Compatibility score = probability × 100. "Compatible" = probability ≥ 0.5.
- Label: **High** if ≥65, **Moderate** if ≥40, otherwise **Low**.
- Returns the top 5 compatibility factors and the top 3 longevity factors.
- The API and the Streamlit app both use this same `Predictor` class. The Streamlit app calls it directly, not through the API.

---

## 5. FEATURES

| # | Feature | How it is implemented | How a user uses it |
|---|---|---|---|
| 1 | One-command training pipeline | `run_pipeline.py` → `relcompat.pipeline.run_pipeline()` | `python run_pipeline.py` |
| 2 | Configurable runs | `config/config.yaml`, `Config.with_overrides()`; CLI flags `--config`, `--n-samples`, `--n-iter`, `--cv-folds`, `--seed`, `--quick` (3000 samples, 3 folds, 2 iterations), `--no-plots` | e.g. `python run_pipeline.py --quick` |
| 3 | Synthetic data generator (reproducible by seed) | `data/generate.py: generate_couples()` | Runs automatically in the pipeline |
| 4 | Data validation and cleaning | `data/validate.py: validate()` | Runs automatically |
| 5 | Symmetric pairwise features | `features/build_features.py: build_features()` | Runs automatically (training and prediction) |
| 6 | Automatic model selection (3 models × 3 tasks) | `models/train.py: train_task()` | Runs automatically; results in `metrics.json` |
| 7 | Evaluation metrics and plots | `models/evaluate.py` | See `reports/figures/` and `reports/model_comparison.md` |
| 8 | Per-couple explanations (top factors) | `models/explain.py: top_factors()` | Shown in the app charts and the API `top_factors` field |
| 9 | Compatibility score, label, years, 5-year probability | `models/predict.py: Predictor.predict()` | Through the app or the API |
| 10 | REST API with auto docs | `api/main.py`: `GET /health`, `GET /model-info`, `POST /predict` | Open `http://localhost:8000/docs` |
| 11 | Input validation (422 for bad input) | pydantic models in `schemas.py` | Sending e.g. age 12 returns error 422 |
| 12 | Clear error if models are missing (503) | `predictor_dependency()` in `api/main.py` | API returns 503 with a "run the pipeline first" message |
| 13 | Streamlit "Predict" tab | `app/streamlit_app.py`: `partner_form()`, `gauge()`, `factor_chart()` | Fill both partners (pre-filled with the example couple), click **Predict** |
| 14 | Streamlit "Model performance" tab | Same file, reads `metrics.json` and figure PNGs | Click the tab to see the comparison tables and plots |
| 15 | Streamlit "About" tab | Same file | Explains the pipeline and how to read the charts |
| 16 | EDA notebook | `notebooks/01_eda.ipynb` | Open in Jupyter, or `make notebook` to re-run it |
| 17 | Automated tests (27) | `tests/` | `python -m pytest -q` |
| 18 | Docker deployment | `Dockerfile`, `docker-compose.yml` (trains models during the build) | `docker compose up --build` |
| 19 | Makefile shortcuts | `Makefile` | `make install`, `make train`, `make quick`, `make test`, `make api`, `make app`, `make notebook`, `make docker`, `make clean` |

---

## 6. HOW TO RUN IT

Run these from the project root folder:

```bash
# 1. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)

# 2. Install
pip install -r requirements.txt
pip install -e .

# 3. Train everything (about 2 minutes; last run took 102.7 s)
python run_pipeline.py
#    or a fast smoke run:  python run_pipeline.py --quick

# 4. Run the tests
python -m pytest -q

# 5. Start the REST API  ->  http://localhost:8000/docs
uvicorn api.main:app --reload

# 6. Start the web app (in a second terminal)  ->  http://localhost:8501
streamlit run app/streamlit_app.py
```

**Docker option:**
```bash
docker compose up --build      # API on http://localhost:8000, app on http://localhost:8501
```

**Environment variables / keys:**
- `RELCOMPAT_MODELS_DIR` (optional): tells the API where to find the models. The default is `models/`.
- API keys: **Not found / not needed.**

**Note:** the `models/` folder already contains trained models, so steps 5 and 6 work without step 3. If the models are missing, the API returns error 503 and the app shows an error asking you to run the pipeline.

---

## 7. DEMO PLAN FOR A SCREEN RECORDING (about 5 minutes)

**Prepare before recording:**
- Install everything (section 6) and run `python run_pipeline.py --quick` once to make sure everything works. Then run the full `python run_pipeline.py` so the real models and figures are in place. *(Note: the quick run overwrites the models with smaller ones, so always finish with the full run.)*
- Run `python -m pytest -q` once to confirm all tests pass.
- Open two terminals: one running `uvicorn api.main:app --reload`, one running `streamlit run app/streamlit_app.py`.
- Open browser tabs: `http://localhost:8501`, `http://localhost:8000/docs`, and the GitHub README.
- Open `reports/figures/compatibility_shap_summary.png` in an image viewer.
- Set browser zoom to about 90% so the whole form fits. Close unrelated tabs and notifications.

| Time | Show | Do / click | Say |
|---|---|---|---|
| 0:00–0:30 | GitHub README (architecture diagram) | Scroll to the mermaid flowchart | "This is an end-to-end ML pipeline that predicts relationship compatibility and how long a relationship may last. All data is synthetic. The goal is to show complete ML engineering, from data to a deployed app." |
| 0:30–1:15 | VS Code: `generate.py`, `build_features.py` | Briefly show `latent_quality()` and the feature list | "We generate 10,000 couples with 20 attributes per partner. Labels come from a hidden quality score plus noise. We turn each couple into 33 symmetric pair features, like age gap, hobby overlap and agreement on children." |
| 1:15–1:45 | Terminal | Show the end of the `python run_pipeline.py` output (the "Selected models" table) | "The pipeline tunes Logistic Regression, Random Forest and XGBoost with 5-fold cross-validation for each of three tasks, and picks the best by CV score." |
| 1:45–2:45 | Streamlit **Predict** tab | The form is pre-filled with the example couple. Click **Predict** | "Here are two partners. After clicking Predict we see a compatibility gauge, expected years, and the chance of lasting 5+ years. For the example couple the app shows 94.7%, label *High*, 13.9 years, and 100% chance of 5+ years (the API returns 0.996; the app rounds it). The green and red bars show which factors raise or lower the score." |
| 2:45–3:15 | Streamlit Predict tab | Change Partner 2: wants children → **No**, attachment → **Avoidant**, conflict style → **Competing**, tick **Smoker**, hobbies → only **Gaming**. Click **Predict** again | "If we create clear mismatches, like disagreeing on children, an avoidant attachment and different hobbies, the score drops, and the factor chart shows why." (The test `test_mismatched_couple_scores_lower` checks a similar mismatched couple scores lower. Try this exact change once before recording to see your real numbers.) |
| 3:15–3:50 | Streamlit **Model performance** tab | Scroll through the three tables and the plots | "Logistic regression won compatibility with ROC-AUC 0.881. Ridge won duration with R² 0.719. Logistic regression won the 5-year task with ROC-AUC 0.926. These are on 2,000 held-out couples." |
| 3:50–4:10 | SHAP summary image | Show `compatibility_shap_summary.png` | "The most important factors are agreement on children, attachment-style pairing, agreeableness gap and religious alignment. These are exactly the rules hidden in the generator, so the model learned real structure." |
| 4:10–4:40 | `http://localhost:8000/docs` | Open **POST /predict** → *Try it out* → *Execute* (the example is pre-filled) | "The same models are served as a REST API built with FastAPI. Input is validated automatically, and the response is JSON with the score and top factors." |
| 4:40–5:00 | Terminal | Run `python -m pytest -q` | "There are 27 automated tests covering data, features, explanations, prediction and the API. It can also run with one Docker command. Thank you." |

**Expected outputs:** the example values above are **from the README**. Your numbers may differ slightly after retraining, so read the real numbers off the screen.

---

## 8. RESULTS AND OUTCOMES

**Data used in the final run (`models/metrics.json`):** 10,000 couples, 8,000 train, 2,000 test, seed 42. The full pipeline took **102.7 seconds**.

**EDA facts (`notebooks/01_eda.ipynb` outputs):**
- Raw data shape: (10000, 47)
- 49.0% of couples are compatible; 54.8% last 5+ years (balanced classes)
- Missing values per column: between 1.84% and 3.32%
- Rows dropped by validation: 0

### Selected models (hold-out test set)

| Task | Selected model | Best params | Key metrics |
|---|---|---|---|
| Compatibility | Logistic Regression | C = 0.3149 | Accuracy 0.7865, Precision 0.7867, Recall 0.7747, F1 0.7807, **ROC-AUC 0.8812** |
| Longevity (years) | Ridge (log target) | alpha = 0.0195 | **MAE 1.8426 yrs, RMSE 2.8603 yrs, R² 0.7185** |
| Lasts 5+ years | Logistic Regression | C = 2.4810 | Accuracy 0.8375, Precision 0.8480, Recall 0.8535, F1 0.8507, **ROC-AUC 0.9257** |

### All candidates (`reports/model_comparison.md`)

**Compatibility (classification)**

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression **(selected)** | 0.8765 | 0.7865 | 0.7867 | 0.7747 | 0.7807 | 0.8812 |
| Random Forest | 0.8515 | 0.7525 | 0.7500 | 0.7431 | 0.7465 | 0.8395 |
| XGBoost | 0.8698 | 0.7775 | 0.7718 | 0.7757 | 0.7738 | 0.8737 |

**Longevity in years (regression)**

| Model | CV RMSE | MAE | RMSE | R² |
|---|---|---|---|---|
| Ridge **(selected)** | 2.8724 | 1.8426 | 2.8603 | 0.7185 |
| Random Forest | 3.9295 | 2.5701 | 4.0464 | 0.4367 |
| XGBoost | 3.0789 | 1.9444 | 3.0794 | 0.6737 |

**Lasts 5+ years (classification)**

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression **(selected)** | 0.9263 | 0.8375 | 0.8480 | 0.8535 | 0.8507 | 0.9257 |
| Random Forest | 0.8916 | 0.7985 | 0.7837 | 0.8682 | 0.8238 | 0.8854 |
| XGBoost | 0.9143 | 0.8270 | 0.8250 | 0.8645 | 0.8443 | 0.9150 |

### Top 10 features (mean |contribution|)

| Rank | Compatibility | Longevity (years) | Lasts 5+ years |
|---|---|---|---|
| 1 | children_match 1.0332 | children_match 0.2250 | children_match 1.3975 |
| 2 | attachment_pair 0.7435 | attachment_pair 0.1639 | attachment_pair 1.0218 |
| 3 | agreeableness_diff 0.5244 | agreeableness_diff 0.1184 | agreeableness_diff 0.7374 |
| 4 | religiosity_diff 0.4889 | religiosity_diff 0.1079 | religiosity_diff 0.6746 |
| 5 | same_religion 0.4075 | cohabiting 0.0960 | cohabiting 0.5982 |
| 6 | smoker_mismatch 0.3995 | conscientiousness_diff 0.0902 | conscientiousness_diff 0.5773 |
| 7 | conscientiousness_diff 0.3950 | political_distance 0.0900 | combined_neuroticism 0.5482 |
| 8 | hobby_jaccard 0.3618 | same_religion 0.0869 | same_religion 0.5440 |
| 9 | political_distance 0.3518 | smoker_mismatch 0.0846 | political_distance 0.5321 |
| 10 | combined_neuroticism 0.3452 | hobby_jaccard 0.0811 | smoker_mismatch 0.5295 |

### Key findings
- **Simple linear models beat Random Forest and XGBoost on all 3 tasks.** The README explains why: the hidden generating rule is mostly additive (on the log-odds scale). XGBoost was a close second.
- **Close to the best possible:** the code notes that a perfect "oracle" that knows the noise-free score reaches about 0.89 ROC-AUC. The compatibility model reaches 0.881.
- **Sanity check passed:** the top features match the strongest rules in the generator (children, attachment, agreeableness, religion).

### Sample output (from README, for `EXAMPLE_COUPLE`, truncated)
```json
{
  "compatibility_score": 94.7,
  "compatible": true,
  "compatibility_label": "High",
  "expected_years": 13.9,
  "lasts_5_years_probability": 0.996,
  "top_factors": [
    {"feature": "hobby_jaccard", "description": "Overlap in hobbies", "value": 0.5, "impact": 0.9116, "effect": "increases"},
    {"feature": "children_match", "description": "Agreement on wanting children", "value": 0.5, "impact": -0.7774, "effect": "decreases"}
  ]
}
```

### Tests
27 pytest tests (README and pytest cache both list 27). They check that:
- **Data:** the data has the right shape, the same seed gives the same data, label ranges are valid, missing values appear only in the intended columns, and validation drops bad rows and rejects missing columns (6 tests).
- **Features:** the feature columns are correct, features stay identical when partners are swapped, there are no NaNs after preprocessing, and API input gives the same features as training data (4 tests).
- **Explanations:** these work for all 3 model types × 2 problem types (6 tests).
- **Prediction:** the output schema is valid, a mismatched couple scores lower, missing models raise an error, and the labels Low/Moderate/High are correct (6 tests).
- **API:** `/health`, `/predict`, 422 for invalid input, `/model-info`, and 503 without models (5 tests).

Latest pass/fail result: **not re-run while writing this guide.** Run `python -m pytest -q` and screenshot the result.

---

## 9. SCREENSHOTS TO TAKE

1. **Streamlit Predict result:** the gauge with label, expected years and 5-year chance for the example couple.
2. **Factor charts:** "What drives compatibility" and "What drives longevity" (green/red bars).
3. **Streamlit Model performance tab:** the comparison tables for the 3 tasks.
4. **FastAPI `/docs`:** the POST /predict request and its JSON response.
5. **`reports/figures/compatibility_diagnostics.png`:** confusion matrix and ROC curve.
6. **`reports/figures/longevity_reg_diagnostics.png`:** predicted vs actual years and residuals.
7. **`reports/figures/compatibility_shap_summary.png`:** SHAP beeswarm of the most important features.
8. **Terminal: end of `python run_pipeline.py`:** the "Selected models" summary and runtime.
9. **Terminal: `python -m pytest -q`:** all tests passing.
10. **EDA notebook:** the "Correlation with targets" bar chart or the target distribution plots.

---

## 10. LIMITATIONS AND FUTURE SCOPE

**Limitations:**
- **Synthetic data only.** The labels come from hand-chosen weights, not real couples, so the results show that the pipeline works, not real-world truth.
- **Weights are simplified assumptions,** loosely inspired by relationship research but not established science (README).
- **No real-world validation.** Accuracy on real people is unknown.
- **Fixed 0.5 decision threshold.** Probability calibration is not checked.
- **The Streamlit app calls the models directly** instead of through the API.
- **No login or authentication, and no cloud deployment:** Not found.
- **No database:** data and models are files.
- **No CI pipeline or model monitoring:** Not found.
- **Questionnaire-style inputs** cannot capture everything that matters in real relationships.

**Realistic future scope:**
- Train and test on a real, consented, anonymised survey dataset.
- Add probability calibration (calibration curves) and tune the decision threshold.
- Deploy the API and app to a cloud service and connect the app to the API.
- Add a GitHub Actions CI workflow to run tests on every push.
- Track experiments (e.g. MLflow) and save model versions.
- Add fairness checks across groups, and more model types (e.g. LightGBM).
- Store requests and predictions in a database and add monitoring for data drift.

---

## 11. LIKELY VIVA QUESTIONS

1. **Where does the data come from?**
   It is synthetic, created by `generate_couples()` in `data/generate.py`: 10,000 couples with a fixed seed (42). The labels come from a hidden "relationship quality" score plus random noise.

2. **Why use synthetic data?**
   Real relationship data is private and hard to get. With synthetic data we also know the true hidden rule, so we can check that the model learned it, and the importance plots confirm it did.

3. **What are "symmetric" features and why use them?**
   They do not depend on which partner is called partner 1 or partner 2. Examples are the absolute age gap and the sorted attachment pair. Compatibility belongs to the pair, so swapping the partners should not change the answer. `test_features_are_symmetric` checks this.

4. **Which models did you try, and which won?**
   Logistic/Ridge regression, Random Forest and XGBoost. Linear models won all three tasks: ROC-AUC 0.881 for compatibility, R² 0.719 for years, and ROC-AUC 0.926 for 5+ years. The hidden rule is mostly additive, which suits linear models.

5. **How did you choose the best model? Why not use the test score?**
   By the mean 5-fold cross-validation score from `RandomizedSearchCV`. The test set is kept for the final fair check only. Choosing by test score would overfit to the test set.

6. **How do you prevent data leakage?**
   Imputation, scaling and one-hot encoding are inside a scikit-learn `Pipeline`, so they are fitted only on training folds. The test set (20%) is split off before training.

7. **How do you handle missing values and outliers?**
   Numeric features get the median and categorical ones get "unknown" (`SimpleImputer`). Distance outliers are reduced with a `log1p` transform. Rows out of range are removed in `validate()`.

8. **Why is the duration model trained on a log scale?**
   Durations are right-skewed (a few very long relationships). Training on `log1p(years)` with `TransformedTargetRegressor` makes the target more balanced, and predictions are converted back with `expm1`.

9. **How do you explain a single prediction?**
   `explain.py` computes each feature's contribution. XGBoost uses its built-in `pred_contribs`, Random Forest uses SHAP TreeExplainer, and linear models use coefficient × (value − average). One-hot parts are summed back together. The top 5 (compatibility) and top 3 (longevity) are shown.

10. **What is ROC-AUC and why use it?**
    It measures how well the model ranks positive cases above negative ones (1.0 = perfect, 0.5 = random). It does not depend on one threshold, so it was used for model selection. Our compatibility model scores 0.881, close to the ~0.89 a perfect oracle could reach on this noisy data.

*Bonus questions:*
- **How is bad input handled?** Pydantic in `schemas.py` checks types and ranges (e.g. age 18–90). FastAPI returns 422, and returns 503 if the models are missing.
- **Is gender used?** It is generated but excluded from the features and has no effect on the labels. Religion is used only as "same vs different."

---

## 12. REPORT AND VIDEO LINES

**2-line project description:**
An end-to-end machine learning pipeline that predicts a couple's compatibility score, expected relationship duration and chance of lasting 5+ years from two partner profiles, with an explanation of the top factors behind each prediction.
It covers synthetic data generation, validation, symmetric feature engineering, cross-validated model selection (Logistic/Ridge, Random Forest, XGBoost), SHAP explanations, a FastAPI REST API, a Streamlit web app, tests and Docker.

**Tools used (one line):**
Python, pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Plotly, Pydantic, PyYAML, joblib, FastAPI, Uvicorn, Streamlit, Jupyter, pytest, httpx, Docker, Docker Compose, Make, Git, GitHub

**30-second spoken summary:**
"This project is an end-to-end machine learning pipeline for relationship compatibility. You enter the profiles of two partners, like their values, personality, hobbies and habits, and it predicts a compatibility score, how many years the relationship may last, and the chance it lasts five years or more. It also shows which factors raised or lowered the score. I generated 10,000 synthetic couples, built 33 pair-based features, and compared Logistic Regression, Random Forest and XGBoost using cross-validation. The best model reached a ROC-AUC of 0.88 for compatibility. The models are served through a FastAPI API and a Streamlit web app, with 27 automated tests and Docker support."
