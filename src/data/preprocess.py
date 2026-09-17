"""
Loads raw CICIDS-2017 CSV files from data/raw, cleans them, groups labels
into 7 categories, and writes train/test parquet files to data/processed.

CICIDS-2017 ships as 8 separate CSVs (one per capture day/scenario), e.g.:
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

Just drop all of them, unzipped, into data/raw/ and run this script.
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    DATA_RAW_DIR, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH,
    REFERENCE_SAMPLE_PATH, LABEL_GROUP_MAP, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download CICIDS-2017 and place "
            f"the 8 CSV files there first (see README.md)."
        )
    log.info(f"Found {len(csv_files)} CSV file(s) in {raw_dir}")

    frames = []
    for f in csv_files:
        log.info(f"  loading {f.name}")
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        frames.append(df)

    full = pd.concat(frames, axis=0, ignore_index=True)
    log.info(f"Combined raw shape: {full.shape}")
    return full


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # CICIDS-2017 columns have inconsistent leading/trailing whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def group_labels(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_GROUP_MAP).fillna(df[TARGET_COLUMN])

    known = set(LABEL_GROUP_MAP.values())
    unknown = sorted(set(df[TARGET_COLUMN].unique()) - known)
    if unknown:
        log.warning(f"Unmapped labels found, keeping as-is: {unknown}")
    log.info(f"Label distribution after grouping:\n{df[TARGET_COLUMN].value_counts()}")
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    # Replace inf/-inf (common in Flow Bytes/s, Flow Packets/s) with NaN, then drop
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    log.info(f"Dropped {before - len(df)} rows with NaN/inf values ({len(df)} remain)")

    # Drop exact duplicate rows (CICIDS-2017 has a known duplicate issue)
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Dropped {before - len(df)} duplicate rows ({len(df)} remain)")

    # Drop columns that are constant (zero variance) or non-numeric leakage
    non_feature_cols = [c for c in ["Flow ID", "Source IP", "Src IP", "Destination IP",
                                     "Dst IP", "Timestamp", "SimillarHTTP"] if c in df.columns]
    df = df.drop(columns=non_feature_cols, errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nunique = df[numeric_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        log.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    return df


def main():
    raw = load_raw_csvs(DATA_RAW_DIR)
    raw = clean_columns(raw)

    if TARGET_COLUMN not in raw.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' not found. "
                        f"Columns present: {list(raw.columns)[:10]}...")

    raw = group_labels(raw)
    raw = clean_values(raw)

    X = raw.drop(columns=[TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    train_df.to_parquet(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_parquet(PROCESSED_TEST_PATH, index=False)
    log.info(f"Saved train set {train_df.shape} -> {PROCESSED_TRAIN_PATH}")
    log.info(f"Saved test set  {test_df.shape} -> {PROCESSED_TEST_PATH}")

    # A reference sample of BENIGN + normal traffic distribution, used later
    # as the "reference" window for Evidently AI drift comparisons.
    ref_sample = train_df.sample(n=min(5000, len(train_df)), random_state=RANDOM_STATE)
    ref_sample.to_parquet(REFERENCE_SAMPLE_PATH, index=False)
    log.info(f"Saved reference sample {ref_sample.shape} -> {REFERENCE_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
