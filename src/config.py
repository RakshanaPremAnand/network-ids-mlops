"""
Central configuration for the NIDS MLOps project.
All paths, constants, and thresholds live here so nothing is hard-coded
in multiple places.
"""
import os
from pathlib import Path

# ---- Paths ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
MLRUNS_DIR = ROOT_DIR / "mlruns"

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROCESSED_TRAIN_PATH = DATA_PROCESSED_DIR / "train.parquet"
PROCESSED_TEST_PATH = DATA_PROCESSED_DIR / "test.parquet"
REFERENCE_SAMPLE_PATH = DATA_PROCESSED_DIR / "reference_sample.parquet"

MODEL_PATH = MODEL_DIR / "xgb_nids_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.joblib"

PREDICTION_LOG_PATH = LOG_DIR / "predictions_log.jsonl"
DRIFT_STATUS_PATH = LOG_DIR / "drift_status.json"
DRIFT_REPORT_HTML = LOG_DIR / "drift_report.html"

# ---- MLflow -------------------------------------------------------------
MLFLOW_TRACKING_URI = f"file:{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "network-ids"
MLFLOW_MODEL_NAME = "nids-xgboost"

# ---- Label grouping -----------------------------------------------------
# CICIDS-2017 ships ~15 fine-grained labels. We group them into 7
# analyst-relevant categories (BENIGN + 6 attack families), matching the
# "7 attack categories" referenced in the project abstract.
LABEL_GROUP_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Infiltration",
}

RANDOM_STATE = 42
TARGET_COLUMN = "Label"

# ---- Training -------------------------------------------------------------
TEST_SIZE = 0.2
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.15,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}

# ---- Drift monitoring -----------------------------------------------------
DRIFT_WINDOW_SIZE = 500          # matches "500-record rolling window" in the deck
DRIFT_SHARE_THRESHOLD = 0.3      # fraction of drifted columns that triggers retrain
F1_IMPROVEMENT_MARGIN = 0.0      # new model must be >= old F1 to be promoted

# ---- API --------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
