"""
Simulates a live network traffic feed by replaying rows from the held-out
test set against the running FastAPI /predict endpoint, one flow at a
time. This is what feeds the prediction log that drift_monitor.py later
analyzes — standing in for a real packet-capture -> feature-extraction
pipeline.

Run (with the API already running in another terminal):
    python -m scripts.csv_streamer --n 600 --delay 0.05
"""
import sys
import time
import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_TEST_PATH, TARGET_COLUMN, API_HOST, API_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main(n: int, delay: float, host: str, port: int):
    df = pd.read_parquet(PROCESSED_TEST_PATH)
    df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    df = df.sample(n=min(n, len(df)), replace=n > len(df)).reset_index(drop=True)

    url = f"http://{host}:{port}/predict"
    log.info(f"Streaming {len(df)} flows to {url} ...")

    ok, failed = 0, 0
    for i, row in df.iterrows():
        payload = {"features": row.to_dict()}
        try:
            resp = requests.post(url, json=payload, timeout=5)
            resp.raise_for_status()
            ok += 1
            if i % 50 == 0:
                log.info(f"[{i+1}/{len(df)}] {resp.json()['predicted_class']}")
        except Exception as e:
            failed += 1
            log.warning(f"Row {i} failed: {e}")
        time.sleep(delay)

    log.info(f"Done. {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="number of flows to stream")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--host", type=str, default=API_HOST if API_HOST != "0.0.0.0" else "127.0.0.1")
    parser.add_argument("--port", type=int, default=API_PORT)
    args = parser.parse_args()
    main(args.n, args.delay, args.host, args.port)
