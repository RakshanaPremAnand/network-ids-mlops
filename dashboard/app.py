"""
Streamlit dashboard: model metrics, drift status, recent predictions,
and a manual "check drift + retrain" button.

Run:
    streamlit run dashboard/app.py
"""
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import mlflow

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, PREDICTION_LOG_PATH,
    DRIFT_STATUS_PATH, DRIFT_REPORT_HTML,
)

st.set_page_config(page_title="NIDS MLOps Dashboard", layout="wide")
st.title("🛡️ Real-Time NIDS — MLOps Dashboard")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---- Model runs ----------------------------------------------------------
st.header("Model Runs (MLflow)")
try:
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                                   order_by=["start_time DESC"])
        if not runs.empty:
            cols = [c for c in runs.columns if c.startswith("metrics.") or c == "start_time"]
            st.dataframe(runs[cols].head(10), use_container_width=True)
            latest = runs.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Accuracy", f"{latest.get('metrics.accuracy', 0):.4f}")
            c2.metric("Latest F1 (macro)", f"{latest.get('metrics.f1_macro', 0):.4f}")
            c3.metric("Total Runs", len(runs))
        else:
            st.info("No training runs yet. Run `python -m src.models.train` first.")
    else:
        st.info("No MLflow experiment found yet.")
except Exception as e:
    st.warning(f"Could not load MLflow data: {e}")

st.divider()

# ---- Drift status ----------------------------------------------------------
st.header("Drift Monitoring (Evidently AI)")
if DRIFT_STATUS_PATH.exists():
    with open(DRIFT_STATUS_PATH) as f:
        status = json.load(f)
    c1, c2, c3 = st.columns(3)
    c1.metric("Drift Detected", "🔴 Yes" if status["drift_detected"] else "🟢 No")
    c2.metric("Drifted Columns", f"{status['n_drifted_columns']}/{status['n_total_columns']}")
    c3.metric("Drift Share", f"{status['drift_share']:.1%}")
    st.caption(f"Last checked: {status['timestamp']}")

    if DRIFT_REPORT_HTML.exists():
        with st.expander("View full Evidently drift report"):
            st.components.v1.html(DRIFT_REPORT_HTML.read_text(encoding="utf-8"), height=800, scrolling=True)
else:
    st.components.v1.html(DRIFT_REPORT_HTML.read_text(encoding="utf-8"), height=800, scrolling=True)

if st.button("🔄 Run drift check now"):
    with st.spinner("Running drift check..."):
        from src.monitoring.drift_monitor import run_drift_check
        try:
            result = run_drift_check()
            st.success(f"Drift check complete: {result}")
            st.rerun()
        except Exception as e:
            st.error(f"Drift check failed: {e}")

st.divider()

# ---- Recent predictions ----------------------------------------------------
st.header("Recent Predictions")
if PREDICTION_LOG_PATH.exists():
    records = [json.loads(line) for line in open(PREDICTION_LOG_PATH).readlines()[-200:]]
    df = pd.DataFrame(records)
    if not df.empty:
        c1, c2 = st.columns(2)
        c1.metric("Total Logged Predictions", len(df))
        c2.metric("Attack Rate (last 200)", f"{(df['prediction'] != 'BENIGN').mean():.1%}")
        st.bar_chart(df["prediction"].value_counts())
        st.dataframe(df[["timestamp", "prediction", "confidence"]].tail(20),
                     use_container_width=True)
else:
    st.info("No predictions logged yet. Send traffic via `python -m scripts.csv_streamer`.")
