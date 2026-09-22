"""
preprocessing.py
================
Phase 2 - turn the raw DataFrame into the train / validation / test tensors
the network consumes, without leaking test information into the scaler.

Everything downstream (training, MC Dropout, SHAP, LIME, DiCE) works from
the single `Splits` object this file returns, so the row order of the test
set is fixed once here and never changes again. Test-set row position is the
`test_instance_id` used by every later phase.
"""

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class Splits:
    """Everything the rest of the pipeline needs about the data."""

    # Scaled feature frames (column names kept - SHAP, LIME and DiCE need them)
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    # Labels as float32 arrays
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler

    @property
    def feature_names(self) -> list:
        return self.X_train.columns.tolist()

    def as_arrays(self) -> tuple:
        """float32 numpy views of the three feature frames."""
        return (
            self.X_train.to_numpy(dtype=np.float32),
            self.X_val.to_numpy(dtype=np.float32),
            self.X_test.to_numpy(dtype=np.float32),
        )


def split_features_target(df: pd.DataFrame) -> tuple:
    """Separate the 21 features from the binary target."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """70 / 15 / 15 split, class ratio preserved in all three parts."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config.TEST_VAL_FRACTION,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.VAL_SHARE_OF_TEMP,
        stratify=y_temp,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on TRAINING rows only (no leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train[config.SCALED_FEATURES])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Standardise the four continuous features; leave the rest untouched."""
    X_scaled = X.copy()
    X_scaled[config.SCALED_FEATURES] = scaler.transform(X[config.SCALED_FEATURES])
    return X_scaled


def build_splits(df: pd.DataFrame, save_scaler: bool = True) -> Splits:
    """Full Phase 2 pipeline: split -> fit scaler on train -> transform all."""
    X, y = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    scaler = fit_scaler(X_train)
    if save_scaler:
        config.ensure_dirs()
        joblib.dump(scaler, config.SCALER_PATH)

    splits = Splits(
        X_train=apply_scaler(X_train, scaler),
        X_val=apply_scaler(X_val, scaler),
        X_test=apply_scaler(X_test, scaler),
        y_train=y_train.to_numpy(dtype=np.float32),
        y_val=y_val.to_numpy(dtype=np.float32),
        y_test=y_test.to_numpy(dtype=np.float32),
        scaler=scaler,
    )
    check_splits(splits)
    return splits


def check_splits(splits: Splits) -> None:
    """Sanity checks that must hold before training starts."""
    for name, frame in (("train", splits.X_train),
                        ("val", splits.X_val),
                        ("test", splits.X_test)):
        values = frame.to_numpy(dtype=np.float32)
        assert frame.shape[1] == config.N_FEATURES, f"{name}: wrong feature count"
        assert not np.isnan(values).any(), f"{name}: contains NaN"
        assert not np.isinf(values).any(), f"{name}: contains Inf"

    scaled = splits.X_train[config.SCALED_FEATURES]
    assert np.allclose(scaled.mean(), 0, atol=1e-6), "train means are not ~0"
    assert np.allclose(scaled.std(ddof=0), 1, atol=1e-6), "train stds are not ~1"


def load_scaler() -> StandardScaler:
    """Load the scaler saved by build_splits()."""
    return joblib.load(config.resolve_input(config.SCALER_PATH))


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Separates the 21 features from the Diabetes_binary target.
# [x] Splits 70% train / 15% validation / 15% test, stratified on the
#     target, with the fixed seed from config.py.
# [x] Fits StandardScaler on the TRAINING rows only, then applies it to
#     all three splits - so no test information reaches the scaler.
# [x] Scales only BMI, MentHlth, PhysHlth, Age; binary/ordinal codes are
#     left as they are.
# [x] Saves the fitted scaler to project/src/models/scaler.pkl.
# [x] Converts labels to float32 and keeps features as DataFrames (SHAP,
#     LIME and DiCE all need the column names).
# [x] Returns one `Splits` object that fixes the test-set row order, which
#     defines test_instance_id for every later phase.
# [x] check_splits() asserts no NaN/Inf, correct feature count, and that
#     the scaled training features have mean ~0 and std ~1.
# =====================================================================
