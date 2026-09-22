"""
training.py
===========
Training for the deep models, and fitting for the shallow baselines.

Class-weighted BCE with early stopping on validation ROC-AUC. The same
routine trains a single model or each member of a deep ensemble - an
ensemble here is just K independently seeded runs, which is what makes its
disagreement a usable epistemic-uncertainty signal.
"""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import config
import models as models_module


def make_loader(X, y, batch_size: int = config.BATCH_SIZE, shuffle: bool = False):
    dataset = TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32)),
                            torch.as_tensor(np.asarray(y, dtype=np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def positive_class_weight(y_train: np.ndarray) -> float:
    """Balanced weight for the minority class: n / (2 * n_positive)."""
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
    return float(len(y_train) / (2 * max(counts[1], 1)))


def _weighted_bce(probabilities, targets, pos_weight: float):
    """Per-sample BCE with positives up-weighted, then averaged."""
    losses = nn.functional.binary_cross_entropy(
        probabilities.clamp(1e-7, 1 - 1e-7), targets, reduction="none")
    weights = torch.where(targets == 1.0, pos_weight, 1.0)
    return (losses * weights).mean()


def train_deep_model(module, dataset, seed: int = 0,
                     max_epochs: int = config.MAX_EPOCHS,
                     patience: int = config.EARLY_STOPPING_PATIENCE,
                     verbose: bool = False) -> tuple:
    """
    Train one deep model on dataset.train, select on dataset.val ROC-AUC.

    Returns (module with the best weights restored, history dict).
    """
    config.set_seeds(seed)
    train_loader = make_loader(dataset.array("train"), dataset.y["train"],
                               shuffle=True)
    val_loader = make_loader(dataset.array("val"), dataset.y["val"])

    pos_weight = positive_class_weight(dataset.y["train"])
    optimiser = torch.optim.Adam(module.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_state, stale = -np.inf, None, 0

    for epoch in range(1, max_epochs + 1):
        module.train()
        total, seen = 0.0, 0
        for X_batch, y_batch in train_loader:
            optimiser.zero_grad()
            loss = _weighted_bce(module(X_batch), y_batch, pos_weight)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(X_batch)
            seen += len(X_batch)
        train_loss = total / seen

        module.eval()
        probabilities, labels, total, seen = [], [], 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predicted = module(X_batch)
                total += _weighted_bce(predicted, y_batch, pos_weight).item() * len(X_batch)
                seen += len(X_batch)
                probabilities.append(predicted.cpu().numpy())
                labels.append(y_batch.cpu().numpy())
        val_loss = total / seen
        val_auc = roc_auc_score(np.concatenate(labels), np.concatenate(probabilities))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc, best_state, stale = val_auc, copy.deepcopy(module.state_dict()), 0
        else:
            stale += 1
        if verbose:
            print(f"  epoch {epoch:03d}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  auc {val_auc:.4f}")
        if stale >= patience:
            break

    if best_state is not None:
        module.load_state_dict(best_state)
    history["best_val_auc"] = best_auc
    history["epochs_run"] = epoch
    return module, history


def train_ensemble(dataset, architecture: str, seed: int = 0,
                   n_members: int = config.ENSEMBLE_SIZE,
                   verbose: bool = False) -> tuple:
    """
    Train a deep ensemble: K models, different init seeds, same data.

    Member disagreement is the epistemic-uncertainty signal used in
    uq_methods.py; a single model with MC Dropout cannot express it as well.
    """
    members, histories = [], []
    for k in range(n_members):
        module = models_module.build_deep_model(architecture, dataset.n_features,
                                                seed=seed * 100 + k)
        module, history = train_deep_model(module, dataset, seed=seed * 100 + k,
                                           verbose=False)
        members.append(models_module.TorchPredictor(module))
        histories.append(history)
        if verbose:
            print(f"  member {k + 1}/{n_members}: val AUC {history['best_val_auc']:.4f}")
    return members, histories


def fit_baselines(dataset, seed: int = config.RANDOM_STATE) -> dict:
    """Fit logistic regression and gradient boosting on the training split."""
    fitted = {}
    for name, predictor in models_module.build_baselines(seed).items():
        fitted[name] = predictor.fit(dataset.array("train"), dataset.y["train"])
    return fitted


def save_ensemble(members, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([m.module.state_dict() for m in members], path)


def load_ensemble(architecture: str, n_features: int, path: Path) -> list:
    states = torch.load(path, map_location="cpu", weights_only=False)
    members = []
    for state in states:
        module = models_module.build_deep_model(architecture, n_features)
        module.load_state_dict(state)
        members.append(models_module.TorchPredictor(module))
    return members


# ============================================================
# CHECKLIST
# - Class-weighted BCE (positives up-weighted by n / 2*n_pos) to handle the
#   11-26% positive rates across the three datasets
# - Adam with weight decay, early stopping on validation ROC-AUC, best
#   checkpoint restored
# - train_ensemble(): K independently seeded models = the deep ensemble
#   whose disagreement gives epistemic uncertainty
# - fit_baselines(): logistic regression and gradient boosting on the same
#   training split, for the "is deep learning needed" comparison
# - save_ensemble()/load_ensemble() so explanation phases never retrain
# - The test and calibration splits are never touched here
# ============================================================
