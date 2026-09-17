"""
Generates a small SYNTHETIC dataset that mimics the structure of
CICIDS-2017 (same style of column names, 7 label categories) purely so
you can run the entire pipeline end-to-end in under a minute *before*
downloading the real ~6GB dataset. Accuracy numbers from this synthetic
run are meaningless — it's a plumbing test, not a real experiment.

Run:
    python -m scripts.generate_sample_data
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_RAW_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags", "SYN Flag Count",
    "ACK Flag Count", "Average Packet Size", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "Active Mean", "Idle Mean",
]

LABELS = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "FTP-Patator", "Bot", "Infiltration"]
# Skewed like real CICIDS-2017: mostly benign, attacks are minority classes
LABEL_WEIGHTS = [0.55, 0.15, 0.12, 0.10, 0.04, 0.03, 0.01]


def generate(n_rows: int = 20000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for col in FEATURE_COLUMNS:
        # Log-normal-ish distributions to mimic skewed network-flow features
        data[col] = rng.lognormal(mean=3, sigma=2, size=n_rows).round(2)

    labels = rng.choice(LABELS, size=n_rows, p=LABEL_WEIGHTS)

    # Nudge feature distributions per label so the model has *something*
    # learnable (purely for plumbing-test purposes)
    df = pd.DataFrame(data)
    for i, lbl in enumerate(labels):
        if lbl != "BENIGN":
            shift = (hash(lbl) % 5 + 1)
            df.loc[i, "Flow Duration"] *= shift
            df.loc[i, "Flow Packets/s"] *= shift * 1.5
            df.loc[i, "SYN Flag Count"] = rng.integers(shift, shift + 3)

    df["Label"] = labels
    # inject a few inf/nan values, same as real CICIDS-2017 quirks
    n_bad = max(1, n_rows // 500)
    bad_idx = rng.choice(n_rows, size=n_bad, replace=False)
    df.loc[bad_idx, "Flow Bytes/s"] = np.inf
    return df


def main():
    df = generate()
    out_path = DATA_RAW_DIR / "SAMPLE-synthetic.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Wrote synthetic sample dataset: {out_path} ({df.shape})")
    log.info("This is FAKE data for testing the pipeline only. "
             "Replace data/raw/ with real CICIDS-2017 CSVs for actual results.")


if __name__ == "__main__":
    main()
