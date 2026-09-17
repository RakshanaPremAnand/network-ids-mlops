"""
Reads the drift status written by src/monitoring/drift_monitor.py. If
drift was detected, re-runs training (train.py already compares the new
model's F1-macro against the current Production model and only promotes
if it's at least as good — this is the "self-healing" part of the loop).

Run manually:
    python -m scripts.retrain_trigger

In production this would be scheduled (cron / GitHub Actions on a
schedule) to run after each drift_monitor.py execution.
"""
import sys
import json
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DRIFT_STATUS_PATH
from src.models.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    if not DRIFT_STATUS_PATH.exists():
        log.error(f"No drift status found at {DRIFT_STATUS_PATH}. "
                   f"Run `python -m src.monitoring.drift_monitor` first.")
        return

    with open(DRIFT_STATUS_PATH) as f:
        status = json.load(f)

    log.info(f"Drift status: drift_detected={status['drift_detected']}, "
             f"drift_share={status['drift_share']}")

    if not status["drift_detected"]:
        log.info("No significant drift detected. Skipping retrain.")
        return

    log.info("Drift detected — triggering retraining pipeline...")
    metrics = train(register_if_better=True)
    log.info(f"Retraining complete. New model metrics: {metrics}")
    log.info("If the new model's F1-macro beat the current Production "
              "model, it has been auto-promoted in the MLflow Registry.")


if __name__ == "__main__":
    main()
