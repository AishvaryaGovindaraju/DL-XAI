"""
config.py
=========
Single source of truth for every path, constant and hyper-parameter used by
the UA-XAI pipeline. No other file is allowed to hard-code a number or a path.

Read this file first: it tells you what the pipeline is made of before you
read how any of it works.
"""

from pathlib import Path
import random

import numpy as np

# ---------------------------------------------------------------------
# 1. Paths
#    project/src/config.py -> parents[0]=src, [1]=project, [2]=repo root
# ---------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = SRC_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
XAI_DIR = RESULTS_DIR / "xai_outputs"
FIGURES_DIR = PROJECT_DIR / "figures"

RAW_CSV = RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"
CHECKPOINT_PATH = RESULTS_DIR / "dnn_checkpoint.pt"
DNN_METRICS_PATH = RESULTS_DIR / "dnn_metrics.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"

STRATIFICATION_PATH = PROCESSED_DIR / "uncertainty_stratification.csv"
STRATIFIED_SAMPLES_PATH = PROCESSED_DIR / "stratified_samples.csv"

SHAP_VALUES_PATH = XAI_DIR / "shap_values.npy"
SHAP_IDS_PATH = XAI_DIR / "shap_instance_ids.npy"
LIME_OUTPUTS_PATH = XAI_DIR / "lime_outputs.pkl"
DICE_OUTPUTS_PATH = XAI_DIR / "dice_outputs.pkl"

# The original notebook ran in a hosted sandbox and downloaded its outputs to
# the repo root, so older result files live there instead of the PRD paths
# above. resolve_input() reads whichever copy actually exists.
LEGACY_OUTPUT_DIR = REPO_ROOT


def resolve_input(path: Path) -> Path:
    """Return `path` if it exists, else the legacy repo-root copy of it."""
    path = Path(path)
    if path.exists():
        return path
    legacy = LEGACY_OUTPUT_DIR / path.name
    return legacy if legacy.exists() else path


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for directory in (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, XAI_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------

TARGET_COLUMN = "Diabetes_binary"
N_FEATURES = 21

# Only these four features are continuous enough to standardise; the other
# 17 are binary or ordinal codes and are fed to the network as-is.
SCALED_FEATURES = ["BMI", "MentHlth", "PhysHlth", "Age"]

# 70 / 15 / 15 stratified split.
TEST_VAL_FRACTION = 0.30
VAL_SHARE_OF_TEMP = 0.50

# ---------------------------------------------------------------------
# 3. Deep network and training
# ---------------------------------------------------------------------

RANDOM_STATE = 42

HIDDEN_UNITS = (128, 64, 32)
DROPOUT_RATES = (0.3, 0.3, 0.2)

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
DECISION_THRESHOLD = 0.5

# PRD gate: below this test ROC-AUC the later phases must not be run.
MIN_ACCEPTABLE_AUC = 0.72

# ---------------------------------------------------------------------
# 4. Monte Carlo Dropout and uncertainty strata
# ---------------------------------------------------------------------

MC_PASSES = 50                 # T stochastic forward passes

# Fixed variance cut-points from the PRD, kept for the diagnostic comparison
# in stratification.py. The pipeline itself uses equal-sized percentile
# strata because the observed variances never reach these values.
PRD_THRESHOLDS = (0.05, 0.15)
PRD_ADJUSTED_THRESHOLDS = (0.03, 0.10)

STRATUM_NAMES = ("LOW", "MEDIUM", "HIGH")
MIN_STRATUM_SIZE = 150         # PRD minimum before a stratum is usable
SAMPLES_PER_STRATUM = 200      # 3 x 200 = 600 explained instances

# ---------------------------------------------------------------------
# 5. XAI methods
# ---------------------------------------------------------------------

SHAP_BACKGROUND_SIZE = 500

LIME_NUM_FEATURES = N_FEATURES
LIME_DISCRETIZE_CONTINUOUS = False

DICE_METHOD = "random"
DICE_TOTAL_CFS = 3
DICE_IMMUTABLE_FEATURES = ["Age", "Sex", "Education"]

# Conditional assignment policy the paper evaluates (PRD Part 7).
XAI_ASSIGNMENT_POLICY = {"LOW": "SHAP", "MEDIUM": "LIME", "HIGH": "DiCE"}


# ---------------------------------------------------------------------
# 6. Reproducibility
# ---------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed python, numpy and torch so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Derives every folder path from the location of this file (no absolute
#     paths, so the repo can be moved or cloned anywhere).
# [x] Names each input and output artefact once: raw CSV, scaler, DNN
#     checkpoint, stratified sample, SHAP / LIME / DiCE outputs.
# [x] resolve_input() falls back to the repo-root copies produced by the
#     original hosted-notebook runs, so old results still load.
# [x] ensure_dirs() creates the output folders before anything writes.
# [x] Holds all dataset constants (target column, the 4 scaled features,
#     70/15/15 split fractions).
# [x] Holds all model and training hyper-parameters (layer sizes, dropout,
#     batch size, learning rate, epochs, patience, the 0.72 AUC gate).
# [x] Holds MC Dropout settings (T = 50), stratum settings (3 strata, 200
#     sampled per stratum, 150 minimum) and the PRD variance thresholds
#     kept only for the diagnostic comparison.
# [x] Holds XAI settings (SHAP background size, LIME options, DiCE method,
#     the 3 immutable features) and the LOW/MEDIUM/HIGH -> XAI policy.
# [x] set_seeds() seeds python, numpy and torch from one place.
# =====================================================================
