"""
config.py
=========
Single source of truth for paths, datasets, hyper-parameters and the
risk-control settings. Nothing else in the project hard-codes a number.

Project: Conformal Selective Explanation (CSE) - deciding per patient which
post-hoc explanation to deliver, or to abstain, with a distribution-free
bound on the quality of what is delivered.
"""

from pathlib import Path
import random

import numpy as np

# ---------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results"
XAI_DIR = RESULTS_DIR / "xai_outputs"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, XAI_DIR, MODELS_DIR,
              FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Datasets
#    Three clinical tabular tasks spanning a deliberate size gradient
#    (254k -> 102k -> 9k). Smaller n means more epistemic uncertainty,
#    which is the axis this study is about.
# ---------------------------------------------------------------------

DATASETS = {
    "brfss": {
        "name": "CDC BRFSS2015 Diabetes Health Indicators",
        "file": RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv",
        "url": None,                      # already in the repo
        "target": "Diabetes_binary",
        "positive_meaning": "diabetes or prediabetes",
        "character": "population survey, self-reported, coarse features",
    },
    "diabetes130": {
        "name": "UCI Diabetes 130-US Hospitals (1999-2008)",
        "file": RAW_DIR / "diabetes130.csv",
        "url": "https://archive.ics.uci.edu/static/public/296/data.csv",
        "target": "readmitted",           # recoded to <30-day readmission
        "positive_meaning": "readmitted within 30 days",
        "character": "hospital administrative records, many categoricals",
    },
    "support2": {
        "name": "UCI SUPPORT2 (seriously ill hospitalised adults)",
        "file": RAW_DIR / "support2.csv",
        "url": "https://archive.ics.uci.edu/static/public/880/data.csv",
        "target": "hospdead",
        "positive_meaning": "in-hospital death",
        "character": "ICU physiology, heavy missingness, small n",
    },
}

DEFAULT_DATASETS = ["brfss", "diabetes130", "support2"]

# train / val / calibration / test. The calibration split is what makes the
# conformal guarantee possible: never used for training or for reporting.
SPLIT_FRACTIONS = {"train": 0.60, "val": 0.10, "calib": 0.15, "test": 0.15}

# ---------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------

RANDOM_STATE = 42
SEEDS = [0, 1, 2]                 # repeated runs; --seeds overrides

MLP_HIDDEN = (256, 128, 64)
MLP_DROPOUT = 0.3

FTT_D_TOKEN = 64                  # FT-Transformer token width
FTT_N_BLOCKS = 3
FTT_N_HEADS = 8
FTT_DROPOUT = 0.1

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8
DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------------------
# 4. Uncertainty quantification
# ---------------------------------------------------------------------

MC_PASSES = 50                    # T stochastic forward passes
ENSEMBLE_SIZE = 5                 # deep-ensemble members
CALIBRATION_BINS = 15             # for ECE / reliability diagrams

# ---------------------------------------------------------------------
# 5. Explanations
# ---------------------------------------------------------------------

N_EXPLAIN = 300                   # instances explained per dataset per seed
SHAP_BACKGROUND_SIZE = 100
LIME_NUM_SAMPLES = 2000
IG_STEPS = 32
DICE_TOTAL_CFS = 3
DICE_METHOD = "random"

# Features a counterfactual may never change (per dataset).
IMMUTABLE_FEATURES = {
    "brfss": ["Age", "Sex", "Education"],
    "diabetes130": ["age", "gender", "race"],
    "support2": ["age", "sex", "race"],
}

ATTRIBUTION_METHODS = ["shap", "lime", "ig"]
COUNTERFACTUAL_METHODS = ["dice"]
ALL_METHODS = ATTRIBUTION_METHODS + COUNTERFACTUAL_METHODS

# Explanation-risk estimation
INFIDELITY_SAMPLES = 20           # perturbations per instance
INFIDELITY_NOISE = 0.20           # std of the Gaussian perturbation
STABILITY_NEIGHBOURS = 5          # re-explanations per instance
STABILITY_RADIUS = 0.10           # epsilon-ball radius (scaled space)

# Weights combining the normalised components into one risk in [0, 1].
RISK_WEIGHTS_ATTRIBUTION = {"infidelity": 0.5, "instability": 0.5}
RISK_WEIGHTS_COUNTERFACTUAL = {"invalidity": 0.6, "proximity": 0.2, "sparsity": 0.2}

# ---------------------------------------------------------------------
# 6. Selective explanation / conformal risk control
# ---------------------------------------------------------------------

RISK_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]   # alpha sweep
DEFAULT_ALPHA = 0.10
QUALITY_HEAD_HIDDEN = (64, 32)
QUALITY_HEAD_EPOCHS = 60
QUALITY_HEAD_LR = 1e-3

POLICIES = ["fixed_shap", "fixed_lime", "fixed_ig", "fixed_dice",
            "random", "uncertainty_gating", "cse", "oracle"]

# ---------------------------------------------------------------------
# 7. Statistics
# ---------------------------------------------------------------------

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
ALPHA_SIGNIFICANCE = 0.05
MULTIPLICITY_CORRECTION = "holm"

# ---------------------------------------------------------------------
# 8. Reproducibility
# ---------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECKLIST
# - Defines every path; ensure_dirs() creates the output tree
# - Registers the three clinical datasets (BRFSS, Diabetes-130, SUPPORT2)
#   with targets, download URLs and immutable features
# - Sets the 60/10/15/15 train/val/calibration/test split; the calibration
#   split exists solely to support the conformal guarantee
# - Holds all model, training, UQ, explanation and risk-control constants
# - set_seeds() seeds python, numpy and torch from one place
# ============================================================
