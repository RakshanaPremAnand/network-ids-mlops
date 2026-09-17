"""
Compares the reference training-data distribution against the most recent
window of live predictions (from logs/predictions_log.jsonl) using
Evidently AI, and writes a drift report (HTML) + a machine-readable
status JSON that the retraining trigger script reads.

Run:
    python -m src.monitoring.drift_monitor
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    REFERENCE_SAMPLE_PATH, PREDICTION_LOG_PATH, DRIFT_REPORT_HTML,
    DRIFT_STATUS_PATH, DRIFT_WINDOW_SIZE, DRIFT_SHARE_THRESHOLD, TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_current_window(window_size: int) -> pd.DataFrame:
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No prediction log at {PREDICTION_LOG_PATH}. Hit the /predict "
            f"endpoint at least {window_size} times first."
        )
    records = []
    with open(PREDICTION_LOG_PATH) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records[-window_size:])
    return df


def run_drift_check() -> dict:
    reference = pd.read_parquet(REFERENCE_SAMPLE_PATH)
    reference_features = reference.drop(columns=[TARGET_COLUMN], errors="ignore")

    current = load_current_window(DRIFT_WINDOW_SIZE)
    drop_cols = [c for c in ["timestamp", "prediction", "confidence"] if c in current.columns]
    current_features = current.drop(columns=drop_cols, errors="ignore")

    # Align columns — only compare features present in both
    common_cols = [c for c in reference_features.columns if c in current_features.columns]
    reference_features = reference_features[common_cols]
    current_features = current_features[common_cols]

    log.info(f"Reference window: {reference_features.shape}, "
             f"Current window: {current_features.shape}, "
             f"comparing {len(common_cols)} shared features")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_features, current_data=current_features)
    report.save_html(str(DRIFT_REPORT_HTML))

    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    n_drifted = drift_metric.get("number_of_drifted_columns", 0)
    n_total = drift_metric.get("number_of_columns", len(common_cols)) or 1
    drift_share = n_drifted / n_total
    dataset_drift = drift_share >= DRIFT_SHARE_THRESHOLD

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_drifted_columns": n_drifted,
        "n_total_columns": n_total,
        "drift_share": round(drift_share, 4),
        "threshold": DRIFT_SHARE_THRESHOLD,
        "drift_detected": dataset_drift,
        "window_size": len(current_features),
        "report_path": str(DRIFT_REPORT_HTML),
    }

    with open(DRIFT_STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)

    log.info(f"Drift check complete: {status}")
    return status


if __name__ == "__main__":
    run_drift_check()
