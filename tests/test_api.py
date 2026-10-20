import pytest
from fastapi.testclient import TestClient

from api import main
from relcompat.schemas import EXAMPLE_COUPLE


@pytest.fixture
def client(models_dir, monkeypatch):
    monkeypatch.setenv("RELCOMPAT_MODELS_DIR", str(models_dir))
    main.get_predictor.cache_clear()
    yield TestClient(main.app)
    main.get_predictor.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["models_loaded"] is True


def test_predict(client):
    response = client.post("/predict", json=EXAMPLE_COUPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["compatibility_score"] <= 100
    assert body["compatibility_label"] in {"Low", "Moderate", "High"}


def test_predict_rejects_invalid_input(client):
    bad = {**EXAMPLE_COUPLE, "partner_1": {**EXAMPLE_COUPLE["partner_1"], "age": 12}}
    assert client.post("/predict", json=bad).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert set(body["models"]) == {"compatibility", "longevity_reg", "longevity_cls"}


def test_503_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("RELCOMPAT_MODELS_DIR", str(tmp_path))
    main.get_predictor.cache_clear()
    assert TestClient(main.app).post("/predict", json=EXAMPLE_COUPLE).status_code == 503
    main.get_predictor.cache_clear()
