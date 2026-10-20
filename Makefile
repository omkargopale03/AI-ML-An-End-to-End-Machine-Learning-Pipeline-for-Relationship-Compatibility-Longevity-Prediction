.PHONY: install train quick test api app notebook docker clean

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python run_pipeline.py

quick:
	python run_pipeline.py --quick

test:
	python -m pytest -q

api:
	uvicorn api.main:app --reload --port 8000

app:
	streamlit run app/streamlit_app.py

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

docker:
	docker compose up --build

clean:
	rm -f data/raw/*.csv data/processed/*.csv models/*.joblib models/metrics.json reports/figures/*.png
