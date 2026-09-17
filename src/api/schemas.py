"""
Pydantic request/response models for the /predict endpoint.
"""
from typing import Dict, List
from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """
    A single network flow's features, as produced by a CICFlowMeter-style
    feature extractor. Accepts an arbitrary dict of feature_name -> value
    so it stays in sync with whatever columns survived preprocessing,
    without needing 70+ explicit Pydantic fields.
    """
    features: Dict[str, float] = Field(
        ..., description="Mapping of feature column name to numeric value"
    )


class ShapContribution(BaseModel):
    feature: str
    shap_value: float
    input_value: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    top_contributing_features: List[ShapContribution]
    is_attack: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
