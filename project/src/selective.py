"""
selective.py
============
The contribution: decide per patient WHICH explanation to deliver, or to
deliver NONE, with a distribution-free bound on what gets delivered.

Two pieces
----------
1. A quality head g(h(x), u_epistemic, u_aleatoric, p) -> predicted risk for
   each explanation method. It reads the classifier's penultimate
   representation, so it can see more than the scalar uncertainty that
   existing uncertainty-gating approaches use. Whether that extra
   information helps is hypothesis H1, and it is measured, not assumed.

2. Conformal risk control (Angelopoulos et al., 2022) on top of it. Deliver
   the lowest-predicted-risk explanation when that prediction clears a
   threshold lambda; otherwise abstain.

What exactly is guaranteed - stated precisely, because it matters
----------------------------------------------------------------
We control the POPULATION risk

    R(lambda) = E[ risk(x, m*(x)) * 1{deliver} ]  <=  alpha

i.e. the expected harm per patient of the explanations the system shows,
where abstaining contributes zero. This is the quantity CRC's finite-sample
guarantee covers exactly, with the standard correction term.

The SELECTIVE risk E[risk | deliver] is a ratio of two random quantities
and is NOT covered by that theorem, so it is reported empirically with
bootstrap intervals and never described as guaranteed. Abstaining from
everything attains R = 0 trivially, which is why the honest object of study
is the coverage-risk curve: at each alpha, what fraction of patients still
receive an explanation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------
# 1. Features for the quality head
# ---------------------------------------------------------------------

def build_quality_features(predictor, X: np.ndarray, uncertainty: dict,
                           use_representation: bool = True) -> np.ndarray:
    """
    Concatenate the model's representation with its uncertainty summary.

    use_representation=False gives the uncertainty-only ablation, which is
    what isolates the contribution of the learned representation.
    """
    scalars = np.column_stack([
        uncertainty["mean"],
        uncertainty["epistemic"],
        uncertainty["aleatoric"],
        uncertainty["variance"],
    ]).astype(np.float32)
    if not use_representation:
        return scalars
    representation = predictor.embed(X).astype(np.float32)
    return np.hstack([representation, scalars])


# ---------------------------------------------------------------------
# 2. The quality head
# ---------------------------------------------------------------------

class QualityHead(nn.Module):
    """Small MLP predicting one risk value per explanation method."""

    def __init__(self, n_inputs: int, n_methods: int,
                 hidden=config.QUALITY_HEAD_HIDDEN):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(0.1)]
            width = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_methods)

    def forward(self, x):
        return torch.sigmoid(self.head(self.body(x)))      # risks live in [0, 1]


def fit_quality_head(features: np.ndarray, risks: np.ndarray,
                     epochs: int = config.QUALITY_HEAD_EPOCHS,
                     lr: float = config.QUALITY_HEAD_LR,
                     seed: int = config.RANDOM_STATE,
                     verbose: bool = False) -> tuple:
    """
    Train the head to regress measured explanation risk.

    risks: (n_instances, n_methods), may contain NaN where a method has no
    measurement for that instance; those entries are masked out of the loss.
    """
    config.set_seeds(seed)
    X = torch.as_tensor(np.asarray(features, dtype=np.float32))
    y = torch.as_tensor(np.nan_to_num(risks, nan=0.0).astype(np.float32))
    mask = torch.as_tensor(np.isfinite(risks).astype(np.float32))

    # Standardise inputs; the same statistics are reused at prediction time.
    mean, std = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mean) / std

    model = QualityHead(X.shape[1], y.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        predicted = model(X)
        loss = (((predicted - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
        loss.backward()
        optimiser.step()
        history.append(float(loss.item()))
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    quality head epoch {epoch + 1}: mse {loss.item():.4f}")
    return model, {"loss": history, "mean": mean, "std": std}


def predict_risk(model, normalisation: dict, features: np.ndarray) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(np.asarray(features, dtype=np.float32))
    X = (X - normalisation["mean"]) / normalisation["std"]
    with torch.no_grad():
        return model(X).cpu().numpy()


# ---------------------------------------------------------------------
# 3. Conformal risk control
# ---------------------------------------------------------------------

def crc_threshold(predicted_risk: np.ndarray, realised_risk: np.ndarray,
                  alpha: float, risk_bound: float = 1.0) -> float:
    """
    Largest lambda whose empirical population risk still clears the
    corrected level:

        lambda_hat = sup { lambda : R_hat(lambda) <= alpha - (B - alpha)/n }

    with R_hat(lambda) = (1/n) sum realised_risk_i * 1{predicted_risk_i <= lambda}.
    R_hat is non-decreasing in lambda, so the search is a scan over the
    candidate thresholds. Returns -inf when even the smallest threshold
    violates the level, i.e. the system must abstain everywhere.
    """
    predicted_risk = np.asarray(predicted_risk, dtype=float)
    realised_risk = np.asarray(realised_risk, dtype=float)
    finite = np.isfinite(predicted_risk) & np.isfinite(realised_risk)
    predicted_risk, realised_risk = predicted_risk[finite], realised_risk[finite]
    n = len(predicted_risk)
    if n == 0:
        return -np.inf

    corrected = alpha - (risk_bound - alpha) / n
    if corrected <= 0:
        return -np.inf

    order = np.argsort(predicted_risk)
    cumulative = np.cumsum(realised_risk[order]) / n
    admissible = np.where(cumulative <= corrected)[0]
    if len(admissible) == 0:
        return -np.inf
    return float(predicted_risk[order][admissible[-1]])


def empirical_risk(realised_risk: np.ndarray, delivered: np.ndarray) -> dict:
    """Coverage, population risk (abstention = 0) and selective risk."""
    realised_risk = np.asarray(realised_risk, dtype=float)
    delivered = np.asarray(delivered, dtype=bool)
    n = len(delivered)
    population = float(np.nansum(np.where(delivered, realised_risk, 0.0)) / n)
    selective = (float(np.nanmean(realised_risk[delivered]))
                 if delivered.any() else np.nan)
    return {"coverage": float(delivered.mean()),
            "population_risk": population,
            "selective_risk": selective,
            "n": n}


# ---------------------------------------------------------------------
# 4. Policies
#    Each returns (chosen method index per instance, delivered mask).
# ---------------------------------------------------------------------

def policy_fixed(method_index: int, n_instances: int) -> tuple:
    return (np.full(n_instances, method_index),
            np.ones(n_instances, dtype=bool))


def policy_random(n_instances: int, n_methods: int,
                  seed: int = config.RANDOM_STATE) -> tuple:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, n_methods, size=n_instances),
            np.ones(n_instances, dtype=bool))


def policy_oracle(realised_risk: np.ndarray, coverage: float = 1.0) -> tuple:
    """
    Upper bound: pick the truly best method per instance, and if coverage is
    limited, keep the instances whose best risk is lowest. Not achievable -
    it reads the answers - but it bounds how much is on the table.
    """
    risks = np.where(np.isfinite(realised_risk), realised_risk, np.inf)
    choice = risks.argmin(axis=1)
    best = risks[np.arange(len(choice)), choice]
    if coverage >= 1.0:
        return choice, np.ones(len(choice), dtype=bool)
    keep = int(round(coverage * len(choice)))
    threshold_index = np.argsort(best)[:keep]
    delivered = np.zeros(len(choice), dtype=bool)
    delivered[threshold_index] = True
    return choice, delivered


def policy_uncertainty_gating(epistemic: np.ndarray, realised_risk_calib: np.ndarray,
                              epistemic_calib: np.ndarray,
                              coverage: float, n_bins: int = 3) -> tuple:
    """
    The literature baseline (cf. uncertainty gating, 2026): bin by epistemic
    uncertainty, assign each bin the method that was best on average in that
    bin on the calibration split, and abstain on the most uncertain
    instances until the target coverage is reached.

    Matching coverage is what makes the comparison with CSE fair.
    """
    edges = np.quantile(epistemic_calib, np.linspace(0, 1, n_bins + 1)[1:-1])
    calib_bin = np.digitize(epistemic_calib, edges)
    best_per_bin = {}
    for b in range(n_bins):
        mask = calib_bin == b
        if mask.any():
            means = np.nanmean(np.where(np.isfinite(realised_risk_calib[mask]),
                                         realised_risk_calib[mask], np.nan), axis=0)
            best_per_bin[b] = int(np.nanargmin(means))
        else:
            best_per_bin[b] = 0

    test_bin = np.digitize(epistemic, edges)
    choice = np.array([best_per_bin[b] for b in test_bin])
    keep = int(round(coverage * len(choice)))
    delivered = np.zeros(len(choice), dtype=bool)
    delivered[np.argsort(epistemic)[:keep]] = True     # least uncertain first
    return choice, delivered


def policy_cse(predicted_risk: np.ndarray, threshold: float) -> tuple:
    """
    Conformal Selective Explanation: choose the method with the lowest
    predicted risk, deliver only if that prediction clears the
    CRC-calibrated threshold.
    """
    choice = predicted_risk.argmin(axis=1)
    best_predicted = predicted_risk[np.arange(len(choice)), choice]
    return choice, best_predicted <= threshold


# ---------------------------------------------------------------------
# 5. Evaluating a set of policies on one test sample
# ---------------------------------------------------------------------

def realised_risk_of_choice(realised_risk: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return realised_risk[np.arange(len(choice)), choice]


def evaluate_policies(realised_risk_test: np.ndarray,
                      predicted_risk_test: np.ndarray,
                      epistemic_test: np.ndarray,
                      realised_risk_calib: np.ndarray,
                      predicted_risk_calib: np.ndarray,
                      epistemic_calib: np.ndarray,
                      method_names: list,
                      alpha: float = config.DEFAULT_ALPHA,
                      seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Run every policy at one risk level and return one row per policy.

    The CRC threshold is fitted on the calibration explanations only; every
    baseline that needs a coverage target is given CSE's coverage, so all
    policies are compared at the same number of delivered explanations.
    """
    n_instances, n_methods = realised_risk_test.shape
    threshold = crc_threshold(
        predicted_risk_calib[np.arange(len(predicted_risk_calib)),
                             predicted_risk_calib.argmin(axis=1)],
        realised_risk_of_choice(realised_risk_calib,
                                predicted_risk_calib.argmin(axis=1)),
        alpha=alpha)

    rows = []

    def record(name, choice, delivered):
        realised = realised_risk_of_choice(realised_risk_test, choice)
        row = {"policy": name, "alpha": alpha,
               **empirical_risk(realised, delivered)}
        row["guarantee_met"] = bool(row["population_risk"] <= alpha)
        chosen = pd.Series(np.asarray(method_names)[choice][delivered]) \
            if delivered.any() else pd.Series(dtype=object)
        for method in method_names:
            row[f"share_{method}"] = float((chosen == method).mean()) if len(chosen) else 0.0
        rows.append(row)

    cse_choice, cse_delivered = policy_cse(predicted_risk_test, threshold)
    record("cse", cse_choice, cse_delivered)
    coverage = float(cse_delivered.mean())

    for index, method in enumerate(method_names):
        record(f"fixed_{method}", *policy_fixed(index, n_instances))

    record("random", *policy_random(n_instances, n_methods, seed))
    record("uncertainty_gating",
           *policy_uncertainty_gating(epistemic_test, realised_risk_calib,
                                      epistemic_calib, coverage))
    record("oracle", *policy_oracle(realised_risk_test, coverage))

    table = pd.DataFrame(rows)
    table["crc_threshold"] = threshold
    return table


def check_guarantee(curve_table: pd.DataFrame) -> pd.DataFrame:
    """
    Check the bound the way the theorem states it.

    Conformal risk control guarantees E[L(lambda_hat)] <= alpha, where the
    expectation runs over the calibration draw AND the test point. A single
    run is one draw from that distribution, and because the correction term
    (B - alpha)/n is small, a tight bound is exceeded in roughly half of
    individual runs by construction. Judging validity from one run is a
    misreading of the guarantee.

    This aggregates realised population risk across runs (seeds, datasets,
    architectures) and compares the MEAN against alpha, while also
    reporting the per-run exceedance rate.
    """
    rows = []
    for (dataset, alpha), group in curve_table.groupby(["dataset", "alpha"]):
        realised = group["population_risk"].to_numpy(dtype=float)
        rows.append({
            "dataset": dataset,
            "alpha": alpha,
            "n_runs": len(realised),
            "mean_population_risk": float(np.mean(realised)),
            "se": float(np.std(realised, ddof=1) / np.sqrt(len(realised)))
                  if len(realised) > 1 else np.nan,
            "mean_coverage": float(group["coverage"].mean()),
            "exceedance_rate": float(np.mean(realised > alpha)),
            "bound_met_in_expectation": bool(np.mean(realised) <= alpha),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "alpha"])


def coverage_risk_curve(predicted_risk_test: np.ndarray,
                        realised_risk_test: np.ndarray,
                        predicted_risk_calib: np.ndarray,
                        realised_risk_calib: np.ndarray,
                        alphas=None) -> pd.DataFrame:
    """
    The headline result: for each target alpha, how many patients still get
    an explanation, and what risk they actually received.
    """
    alphas = alphas or config.RISK_LEVELS
    calib_choice = predicted_risk_calib.argmin(axis=1)
    calib_predicted = predicted_risk_calib[np.arange(len(calib_choice)), calib_choice]
    calib_realised = realised_risk_of_choice(realised_risk_calib, calib_choice)

    rows = []
    for alpha in alphas:
        threshold = crc_threshold(calib_predicted, calib_realised, alpha=alpha)
        choice, delivered = policy_cse(predicted_risk_test, threshold)
        realised = realised_risk_of_choice(realised_risk_test, choice)
        rows.append({"alpha": alpha, "crc_threshold": threshold,
                     **empirical_risk(realised, delivered)})
    table = pd.DataFrame(rows)
    table["guarantee_met"] = table["population_risk"] <= table["alpha"]
    return table


# ============================================================
# CHECKLIST
# - build_quality_features(): model representation + epistemic/aleatoric
#   uncertainty + predicted probability (and the uncertainty-only variant
#   used for the ablation)
# - QualityHead: small MLP predicting each method's explanation risk before
#   that explanation is computed; NaN targets are masked out of the loss
# - crc_threshold(): conformal risk control with the (B - alpha)/n
#   finite-sample correction, fitted on the calibration explanations only
# - Guarantee is on POPULATION risk (abstention counts as zero); selective
#   risk is reported but explicitly not claimed as guaranteed
# - Policies implemented: cse, fixed_<method>, random, uncertainty_gating
#   (the 2026 literature baseline) and oracle, all compared at matched
#   coverage
# - coverage_risk_curve(): the paper's headline figure - fraction of
#   patients explained versus the risk level requested
# ============================================================
