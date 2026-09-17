"""
Helpers for loading processed train/test parquet files and applying
SMOTE oversampling to the training split only (never to test data —
that would leak synthetic samples into evaluation).
"""
import sys
import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, TARGET_COLUMN, RANDOM_STATE,
)

log = logging.getLogger(__name__)


def load_train_test():
    train_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
    test_df = pd.read_parquet(PROCESSED_TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def apply_smote(X_train, y_train_enc, sampling_strategy="auto", target_minority_size=50000):
    """
    Oversample minority attack classes so XGBoost doesn't just learn to
    predict BENIGN / DoS for everything. Applied to TRAIN split only.

    Caps oversampling at `target_minority_size` per class instead of fully
    balancing to the majority class size — avoids exploding a 17-sample
    class up to 1.6M+ rows.
    """
    before_counts = pd.Series(y_train_enc).value_counts().to_dict()
    log.info(f"Class distribution before SMOTE: {before_counts}")

    if sampling_strategy == "auto":
        sampling_strategy = {
            cls: max(count, target_minority_size)
            for cls, count in before_counts.items()
        }

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train_enc)
    log.info(f"Class distribution after SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res
