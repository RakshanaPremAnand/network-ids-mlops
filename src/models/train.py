"""
Trains the XGBoost NIDS classifier on SMOTE-balanced CICIDS-2017 features,
logs everything to MLflow (params, metrics, confusion matrix, model
artifact), and registers the model in the MLflow Model Registry.

Run:
    python -m src.models.train
"""
import sys
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    XGB_PARAMS, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME,
    MODEL_PATH, LABEL_ENCODER_PATH, FEATURE_LIST_PATH, MODEL_DIR,
)
from src.data.load_data import load_train_test, encode_labels, apply_smote

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - NIDS XGBoost")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(register_if_better: bool = True):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    log.info("Loading processed train/test data...")
    X_train, X_test, y_train, y_test = load_train_test()

    log.info("Encoding labels...")
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)
    class_names = list(label_encoder.classes_)

    log.info("Applying SMOTE to training split...")
    X_train_res, y_train_res = apply_smote(X_train, y_train_enc)

    with mlflow.start_run() as run:
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("n_train_original", len(X_train))
        mlflow.log_param("n_train_after_smote", len(X_train_res))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(class_names))

        log.info("Training XGBoost classifier...")
        model = XGBClassifier(**XGB_PARAMS, num_class=len(class_names))
        model.fit(X_train_res, y_train_res)

        log.info("Evaluating on held-out test set...")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            y_test_enc, y_pred, average="macro", zero_division=0
        )
        f1_weighted = f1_score(y_test_enc, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        report = classification_report(
            y_test_enc, y_pred, target_names=class_names, zero_division=0
        )
        log.info(f"\n{report}")

        report_path = MODEL_DIR / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        cm_path = MODEL_DIR / "confusion_matrix.png"
        plot_confusion_matrix(y_test_enc, y_pred, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        mlflow.xgboost.log_model(model, artifact_path="model")

        # Persist locally too so FastAPI can load without hitting MLflow
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
        joblib.dump(list(X_train.columns), FEATURE_LIST_PATH)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(LABEL_ENCODER_PATH))

        log.info(f"Run complete. Accuracy={acc:.4f}  F1-macro={f1_macro:.4f}")
        log.info(f"MLflow run_id: {run.info.run_id}")

        if register_if_better:
            _maybe_promote_model(run.info.run_id, f1_macro)

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def _maybe_promote_model(run_id: str, new_f1: float):
    """
    Registers the model. If a Production version already exists, only
    promotes this one to Production if its F1-macro is >= the current
    Production model's F1-macro (logged as a run metric).
    """
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    try:
        mv = mlflow.register_model(model_uri, MLFLOW_MODEL_NAME)
    except Exception as e:
        log.warning(f"Could not register model: {e}")
        return

    try:
        prod_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    except Exception:
        prod_versions = []

    should_promote = True
    if prod_versions:
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_f1 = prod_run.data.metrics.get("f1_macro", -1)
        should_promote = new_f1 >= prod_f1
        log.info(f"Current production F1-macro={prod_f1:.4f}, new={new_f1:.4f}, "
                  f"promote={should_promote}")

    if should_promote:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        log.info(f"Promoted model version {mv.version} to Production.")
    else:
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME, version=mv.version, stage="Staging",
        )
        log.info(f"New model version {mv.version} kept in Staging (did not beat production).")


if __name__ == "__main__":
    train()
