"""
data.py
=======
Loading, cleaning and splitting for the three clinical tabular datasets.

Every dataset is reduced to the same object - `Dataset` - so nothing
downstream needs to know which one it is working with. The split is
train / val / calibration / test; the calibration part is held back
exclusively for conformal risk control and is never trained or reported on.
"""

from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


# ---------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------

@dataclass
class Dataset:
    """One prepared dataset, identically shaped for every downstream step."""

    key: str
    name: str
    X: dict                      # split -> DataFrame (scaled, numeric)
    y: dict                      # split -> float32 array
    feature_names: list
    categorical_features: list   # names that came from one-hot / codes
    numeric_features: list
    immutable_features: list
    scaler: StandardScaler
    meta: dict = field(default_factory=dict)

    def array(self, split: str) -> np.ndarray:
        return self.X[split].to_numpy(dtype=np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def summary(self) -> dict:
        return {
            "dataset": self.key,
            "name": self.name,
            "n_features": self.n_features,
            **{f"n_{s}": len(self.y[s]) for s in self.y},
            "positive_rate": float(np.mean(np.concatenate(list(self.y.values())))),
        }


# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

def download_if_missing(key: str) -> Path:
    """Fetch a dataset from its UCI URL the first time it is needed."""
    spec = config.DATASETS[key]
    path = Path(spec["file"])
    if path.exists():
        return path
    if not spec["url"]:
        raise FileNotFoundError(f"{key}: expected the file at {path}")
    config.ensure_dirs()
    urllib.request.urlretrieve(spec["url"], path)
    return path


# ---------------------------------------------------------------------
# Per-dataset cleaning -> (features DataFrame, binary target Series)
# ---------------------------------------------------------------------

def _prepare_brfss(df: pd.DataFrame) -> tuple:
    y = df[config.DATASETS["brfss"]["target"]].astype(int)
    X = df.drop(columns=[config.DATASETS["brfss"]["target"]])
    numeric = ["BMI", "MentHlth", "PhysHlth", "Age", "GenHlth", "Education", "Income"]
    return X, y, [c for c in numeric if c in X.columns]


def _icd9_group(code) -> str:
    """Collapse an ICD-9 diagnosis code into a clinical chapter."""
    if pd.isna(code):
        return "missing"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        value = float(code)
    except ValueError:
        return "other"
    bands = [(390, 460, "circulatory"), (460, 520, "respiratory"),
             (520, 580, "digestive"), (580, 630, "genitourinary"),
             (710, 740, "musculoskeletal"), (800, 1000, "injury"),
             (140, 240, "neoplasms")]
    if 250 <= value < 251:
        return "diabetes"
    for low, high, label in bands:
        if low <= value < high:
            return label
    return "other"


def _prepare_diabetes130(df: pd.DataFrame) -> tuple:
    # Target: readmission within 30 days (the clinically actionable version).
    y = (df["readmitted"] == "<30").astype(int)

    drop = ["encounter_id", "patient_nbr", "readmitted",
            "weight",            # 97% missing
            "payer_code"]        # billing, not clinical
    X = df.drop(columns=[c for c in drop if c in df.columns])

    for column in ["diag_1", "diag_2", "diag_3"]:
        X[column] = X[column].map(_icd9_group)

    X["medical_specialty"] = X["medical_specialty"].fillna("missing")
    # "?" is the repository's missing marker.
    X = X.replace("?", np.nan)

    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures",
               "num_medications", "number_outpatient", "number_emergency",
               "number_inpatient", "number_diagnoses"]
    categorical = [c for c in X.columns if c not in numeric]

    # Drop constant columns (several drugs are never prescribed here).
    constant = [c for c in categorical if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant)
    categorical = [c for c in categorical if c not in constant]

    # Rare categories -> "other", then one-hot.
    for column in categorical:
        counts = X[column].value_counts(dropna=False)
        rare = counts[counts < 100].index
        X[column] = X[column].where(~X[column].isin(rare), "other").fillna("missing")
    X = pd.get_dummies(X, columns=categorical, drop_first=False, dtype=float)
    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(X[column].median())
    return X, y, numeric


def _prepare_support2(df: pd.DataFrame) -> tuple:
    y = df["hospdead"].astype(int)

    # Remove the outcome, anything computed after it, and the study's own
    # model/physician survival estimates - all of them leak the label.
    leak = ["id", "hospdead", "death", "d.time", "sfdm2", "slos",
            "charges", "totcst", "totmcst", "surv2m", "surv6m",
            "prg2m", "prg6m", "dnr", "dnrday", "hday"]
    X = df.drop(columns=[c for c in leak if c in df.columns])

    # pandas >= 3 gives string columns a dedicated dtype, so test for
    # numeric rather than for object.
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    # Median impute + explicit missingness indicators: in ICU data "not
    # measured" is itself clinical information.
    for column in numeric:
        if X[column].isna().any():
            X[f"{column}_missing"] = X[column].isna().astype(float)
        X[column] = X[column].fillna(X[column].median())
    for column in categorical:
        X[column] = X[column].fillna("missing")
    X = pd.get_dummies(X, columns=categorical, drop_first=False, dtype=float)
    numeric = [c for c in numeric if c in X.columns]
    return X, y, numeric


_PREPARE = {"brfss": _prepare_brfss,
            "diabetes130": _prepare_diabetes130,
            "support2": _prepare_support2}


# ---------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------

def four_way_split(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple:
    """Stratified 60 / 10 / 15 / 15 train / val / calibration / test."""
    f = config.SPLIT_FRACTIONS
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1 - f["train"], stratify=y, random_state=seed)
    # of the remaining 40%: val 25%, calib 37.5%, test 37.5%
    rest = 1 - f["train"]
    X_val, X_tmp, y_val, y_tmp = train_test_split(
        X_rest, y_rest, test_size=1 - f["val"] / rest,
        stratify=y_rest, random_state=seed)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=f["test"] / (f["calib"] + f["test"]),
        stratify=y_tmp, random_state=seed)
    return ({"train": X_train, "val": X_val, "calib": X_cal, "test": X_test},
            {"train": y_train, "val": y_val, "calib": y_cal, "test": y_test})


def scale(X_splits: dict, numeric_features: list) -> tuple:
    """Standardise numeric columns using TRAINING statistics only."""
    scaler = StandardScaler().fit(X_splits["train"][numeric_features])
    scaled = {}
    for split, frame in X_splits.items():
        out = frame.copy()
        out[numeric_features] = scaler.transform(frame[numeric_features])
        scaled[split] = out.astype(np.float32)
    return scaled, scaler


def _expand_immutables(names: list, columns: list) -> list:
    """
    Map configured immutable feature names onto the actual columns.

    One-hot encoding turns `gender` into `gender_Female`, `gender_Male`, so a
    literal name match would silently protect nothing.
    """
    protected = []
    for name in names:
        protected += [c for c in columns if c == name or c.startswith(name + "_")]
    return sorted(set(protected))


def load_dataset(key: str, seed: int = config.RANDOM_STATE) -> Dataset:
    """Full path from raw file to a split, scaled, model-ready Dataset."""
    spec = config.DATASETS[key]
    path = download_if_missing(key)
    df = pd.read_csv(path, low_memory=False)

    X, y, numeric = _PREPARE[key](df)
    X = X.loc[:, X.nunique(dropna=False) > 1]          # drop constants
    numeric = [c for c in numeric if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]

    X_splits, y_splits = four_way_split(X, y, seed)
    X_scaled, scaler = scale(X_splits, numeric)

    dataset = Dataset(
        key=key,
        name=spec["name"],
        X=X_scaled,
        y={s: v.to_numpy(dtype=np.float32) for s, v in y_splits.items()},
        feature_names=list(X.columns),
        categorical_features=categorical,
        numeric_features=numeric,
        immutable_features=_expand_immutables(
            config.IMMUTABLE_FEATURES.get(key, []), list(X.columns)),
        scaler=scaler,
        meta={"character": spec["character"],
              "positive_meaning": spec["positive_meaning"],
              "seed": seed},
    )
    check_dataset(dataset)
    return dataset


def check_dataset(dataset: Dataset) -> None:
    """Assertions that must hold before any model sees the data."""
    for split in ("train", "val", "calib", "test"):
        values = dataset.array(split)
        assert values.shape[1] == dataset.n_features, f"{split}: feature count"
        assert np.isfinite(values).all(), f"{split}: non-finite values"
        assert len(np.unique(dataset.y[split])) == 2, f"{split}: not binary"
    train_index = set(dataset.X["train"].index)
    for split in ("val", "calib", "test"):
        assert not train_index & set(dataset.X[split].index), f"{split} overlaps train"


def dataset_table(keys=None) -> pd.DataFrame:
    """Descriptive table of all datasets - Table 1 of the paper."""
    keys = keys or config.DEFAULT_DATASETS
    return pd.DataFrame([load_dataset(k).summary() for k in keys])


# ============================================================
# CHECKLIST
# - Downloads Diabetes-130 and SUPPORT2 from UCI on first use; BRFSS ships
#   with the repo
# - Cleans each dataset separately: BRFSS as-is; Diabetes-130 gets ICD-9
#   diagnosis grouping, rare-category collapsing and one-hot encoding;
#   SUPPORT2 gets label-leaking columns removed (death, d.time, survival
#   and physician prognosis estimates), median imputation and explicit
#   missingness indicators
# - Recodes every target to a single binary label
# - Splits 60/10/15/15 stratified into train/val/calibration/test; the
#   calibration split is reserved for conformal risk control
# - Standardises numeric columns on TRAINING statistics only
# - Returns one uniform `Dataset` object so no later module is
#   dataset-specific
# - check_dataset() asserts finiteness, binary targets and no split overlap
# ============================================================
