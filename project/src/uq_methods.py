"""
uq_methods.py
=============
Uncertainty quantification, decomposed.

The project's earlier version used a single number - MC-Dropout predictive
variance - and treated it as "epistemic uncertainty". It is not: it mixes
two things that mean different clinical things.

    total       H[E_theta p]          how unsure the system is overall
    aleatoric   E_theta H[p]          irreducible noise in the data
    epistemic   H[E p] - E H[p]       what the model does not know (mutual
                                      information between prediction and
                                      parameters) - reducible with more data

Only the epistemic part supports the claim "the model is out of its depth
here", which is the signal the selective-explanation policy needs.

Two samplers are provided (MC Dropout, deep ensemble) so no conclusion
rests on a single UQ method, plus temperature scaling and calibration
metrics, because uncertainty that is not calibrated licenses nothing.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss

import config


# ---------------------------------------------------------------------
# Samplers: produce a (n_samples, n_instances) matrix of probabilities
# ---------------------------------------------------------------------

def enable_mc_dropout(module: nn.Module) -> None:
    """
    Eval mode, then switch ONLY the dropout layers back on.

    A blanket module.train() would also put BatchNorm into batch-statistics
    mode, making a patient's prediction depend on who else is in the batch -
    that is batch noise, not epistemic uncertainty.
    """
    module.eval()
    for layer in module.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()


def mc_dropout_samples(predictor, X, n_passes: int = config.MC_PASSES) -> np.ndarray:
    """T stochastic forward passes through one dropout network."""
    module = predictor.module
    enable_mc_dropout(module)
    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    samples = []
    with torch.no_grad():
        for _ in range(n_passes):
            batch_out = [module(X_tensor[i:i + 4096]).cpu().numpy()
                         for i in range(0, len(X_tensor), 4096)]
            samples.append(np.concatenate(batch_out))
    module.eval()
    return np.stack(samples)


def ensemble_samples(members, X) -> np.ndarray:
    """One probability row per ensemble member."""
    return np.stack([m.predict_proba(X) for m in members])


def get_samples(method: str, predictor_or_members, X) -> np.ndarray:
    if method == "mc_dropout":
        predictor = (predictor_or_members[0]
                     if isinstance(predictor_or_members, (list, tuple))
                     else predictor_or_members)
        return mc_dropout_samples(predictor, X)
    if method == "ensemble":
        return ensemble_samples(predictor_or_members, X)
    raise ValueError(f"unknown UQ method: {method}")


UQ_METHODS = ["mc_dropout", "ensemble"]


# ---------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------

def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def decompose(samples: np.ndarray) -> dict:
    """
    Split predictive uncertainty into aleatoric and epistemic parts.

    samples: (n_samples, n_instances) of P(positive).
    Returns mean prediction, the three uncertainty terms, and the raw
    predictive variance kept for comparison with the earlier design.
    """
    mean = samples.mean(axis=0)
    total = _binary_entropy(mean)                      # H[E p]
    aleatoric = _binary_entropy(samples).mean(axis=0)  # E H[p]
    epistemic = np.maximum(total - aleatoric, 0.0)     # mutual information
    return {
        "mean": mean,
        "variance": samples.var(axis=0),
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
    }


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def expected_calibration_error(y_true, probabilities,
                               n_bins: int = config.CALIBRATION_BINS) -> float:
    """Standard equal-width ECE."""
    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error, n = 0.0, len(probabilities)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        if not mask.any():
            continue
        error += mask.sum() / n * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def reliability_curve(y_true, probabilities,
                      n_bins: int = config.CALIBRATION_BINS) -> dict:
    """Bin centres, observed frequency and count - for the reliability plot."""
    probabilities, y_true = np.asarray(probabilities), np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, observed, counts = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        centres.append((low + high) / 2)
        observed.append(float(y_true[mask].mean()) if mask.any() else np.nan)
        counts.append(int(mask.sum()))
    return {"bin_centre": np.array(centres), "observed": np.array(observed),
            "count": np.array(counts)}


def calibration_metrics(y_true, probabilities) -> dict:
    return {
        "ece": expected_calibration_error(y_true, probabilities),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "mean_predicted": float(np.mean(probabilities)),
        "observed_rate": float(np.mean(y_true)),
    }


def fit_temperature(logits_or_probs, y_true, is_probability: bool = True) -> float:
    """
    Temperature scaling on the validation split (Guo et al., 2017).

    Returns T; apply with apply_temperature(). T > 1 means the model was
    over-confident.
    """
    p = np.clip(np.asarray(logits_or_probs), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)) if is_probability else np.asarray(logits_or_probs)
    logit_tensor = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(np.asarray(y_true), dtype=torch.float32)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimiser.zero_grad()
        scaled = logit_tensor / torch.exp(log_temperature)
        loss = nn.functional.binary_cross_entropy_with_logits(scaled, target)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(torch.exp(log_temperature).item())


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)) / temperature
    return 1.0 / (1.0 + np.exp(-logits))


# ============================================================
# CHECKLIST
# - Two samplers: MC Dropout (dropout layers only; BatchNorm stays in eval
#   so uncertainty is not contaminated by batch statistics) and deep
#   ensembles
# - decompose(): total / aleatoric / epistemic via the entropy-minus-
#   expected-entropy (mutual information) identity - the earlier design's
#   single predictive variance conflated these
# - Predictive variance is still reported, for comparison with that design
# - Calibration: ECE, Brier, reliability curve, and temperature scaling
#   fitted on validation - uncalibrated uncertainty supports no claim
# - Nothing here trains or explains; it only turns a fitted model into
#   per-instance uncertainty numbers
# ============================================================
