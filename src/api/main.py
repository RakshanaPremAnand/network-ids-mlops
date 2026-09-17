"""
Real-time inference API for the NIDS.

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> liveness + model status
    POST /predict  -> classify a single network flow, with SHAP explanation
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_LIST_PATH, PREDICTION_LOG_PATH,
)
from src.api.schemas import FlowFeatures, PredictionResponse, HealthResponse, ShapContribution
from src.explainability.shap_explain import NIDSExplainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Network Intrusion Detection API",
    description="XGBoost + SHAP powered NIDS, CICIDS-2017",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- Load model artifacts at startup --------------------------------------
_state = {"model": None, "label_encoder": None, "feature_names": None, "explainer": None}


@app.on_event("startup")
def load_artifacts():
    if not MODEL_PATH.exists():
        log.warning(f"Model not found at {MODEL_PATH}. Run `python -m src.models.train` first.")
        return
    _state["model"] = joblib.load(MODEL_PATH)
    _state["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)
    _state["feature_names"] = joblib.load(FEATURE_LIST_PATH)
    _state["explainer"] = NIDSExplainer(_state["model"], _state["feature_names"])
    log.info(f"Model loaded. {len(_state['feature_names'])} features, "
             f"{len(_state['label_encoder'].classes_)} classes.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_state["model"] is not None,
        model_version=MODEL_PATH.name if _state["model"] is not None else None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FlowFeatures):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    start = time.perf_counter()
    feature_names = _state["feature_names"]

    row = {f: payload.features.get(f, 0.0) for f in feature_names}
    X = pd.DataFrame([row], columns=feature_names)

    model = _state["model"]
    label_encoder = _state["label_encoder"]

    proba = model.predict_proba(X)[0]
    pred_idx = int(proba.argmax())
    pred_class = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    class_probs = {
        cls: round(float(p), 5) for cls, p in zip(label_encoder.classes_, proba)
    }

    top_features = _state["explainer"].explain(X, pred_idx, top_k=5)

    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(f"Prediction: {pred_class} (conf={confidence:.3f}) in {elapsed_ms:.1f}ms")

    _log_prediction(row, pred_class, confidence)

    return PredictionResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 5),
        class_probabilities=class_probs,
        top_contributing_features=[ShapContribution(**f) for f in top_features],
        is_attack=pred_class != "BENIGN",
    )


def _log_prediction(features: dict, pred_class: str, confidence: float):
    """
    Appends every prediction to a JSONL log. This log is what the drift
    monitor later reads to build the "current" window and compare it
    against the training-time reference distribution.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": pred_class,
        "confidence": confidence,
        **features,
    }
    with open(PREDICTION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
