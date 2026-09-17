"""
Wraps a SHAP TreeExplainer around the trained XGBoost model so every
prediction served by the API can come with a feature-level explanation
(this is what makes alerts "analyst-reviewable" instead of black-box).
"""
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)


class NIDSExplainer:
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X_row: pd.DataFrame, predicted_class_idx: int, top_k: int = 5):
        """
        Returns the top_k features (by absolute SHAP value) that pushed the
        prediction towards the predicted class, for a single row of input.
        """
        shap_values = self.explainer.shap_values(X_row)

        # xgboost multiclass -> shap_values is a list[n_classes] of (n_rows, n_features)
        # or a single (n_rows, n_features, n_classes) array depending on shap version
        if isinstance(shap_values, list):
            class_shap = np.array(shap_values[predicted_class_idx])[0]
        elif shap_values.ndim == 3:
            class_shap = shap_values[0, :, predicted_class_idx]
        else:
            class_shap = shap_values[0]

        contributions = list(zip(self.feature_names, class_shap.tolist()))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contributions[:top_k]

        return [
            {"feature": name, "shap_value": round(float(val), 5),
             "input_value": round(float(X_row.iloc[0][name]), 5)}
            for name, val in top
        ]


def load_explainer() -> "NIDSExplainer":
    from src.config import MODEL_PATH, FEATURE_LIST_PATH
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_LIST_PATH)
    return NIDSExplainer(model, feature_names)
