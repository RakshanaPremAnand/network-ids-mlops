"""
Basic test suite for the NIDS API and preprocessing utilities.

Run:
    pytest tests/ -v

Note: these tests assume a model has already been trained (they will
skip gracefully if models/xgb_nids_model.joblib doesn't exist yet — run
`python -m src.models.train` first, e.g. on the synthetic sample data).
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_PATH

MODEL_READY = MODEL_PATH.exists()


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_endpoint_returns_valid_schema(client):
    import joblib
    from src.config import FEATURE_LIST_PATH
    feature_names = joblib.load(FEATURE_LIST_PATH)
    dummy_row = {f: 1.0 for f in feature_names}

    resp = client.post("/predict", json={"features": dummy_row})
    assert resp.status_code == 200
    body = resp.json()

    assert "predicted_class" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert "class_probabilities" in body
    assert "top_contributing_features" in body
    assert len(body["top_contributing_features"]) <= 5
    assert isinstance(body["is_attack"], bool)


@pytest.mark.skipif(not MODEL_READY, reason="Model not trained yet")
def test_predict_missing_features_defaults_to_zero(client):
    """The API should not crash if a payload is missing some feature keys."""
    resp = client.post("/predict", json={"features": {"Flow Duration": 100.0}})
    assert resp.status_code == 200


def test_predict_rejects_malformed_payload(client):
    resp = client.post("/predict", json={"not_features": {}})
    assert resp.status_code == 422


def test_label_group_map_covers_benign():
    from src.config import LABEL_GROUP_MAP
    assert LABEL_GROUP_MAP["BENIGN"] == "BENIGN"
    assert LABEL_GROUP_MAP["DoS Hulk"] == "DoS"


def test_clean_columns_strips_whitespace():
    import pandas as pd
    from src.data.preprocess import clean_columns
    df = pd.DataFrame({" Label ": [1], "  Flow Duration": [2]})
    cleaned = clean_columns(df)
    assert "Label" in cleaned.columns
    assert "Flow Duration" in cleaned.columns
