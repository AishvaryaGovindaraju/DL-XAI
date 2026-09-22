"""
training.py
===========
Phase 4b - class-weighted training of the DNN with early stopping on
validation ROC-AUC.

The dataset is roughly 86% negative, so an unweighted loss produces a model
that predicts "no diabetes" for almost everyone. Positive cases are therefore
up-weighted by the balanced class weight computed from the training split.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import config
from preprocessing import Splits


def make_loaders(splits: Splits, batch_size: int = config.BATCH_SIZE) -> dict:
    """Wrap the three splits in DataLoaders; only training data is shuffled."""
    X_train, X_val, X_test = splits.as_arrays()

    def loader(X, y, shuffle):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return {
        "train": loader(X_train, splits.y_train, True),
        "val": loader(X_val, splits.y_val, False),
        "test": loader(X_test, splits.y_test, False),
    }


def positive_class_weight(y_train: np.ndarray) -> float:
    """Balanced weight for the minority (diabetes) class: n / (2 * n_pos)."""
    counts = np.bincount(y_train.astype(int))
    return float(len(y_train) / (len(counts) * counts[1]))


def _weighted_bce(criterion, predictions, targets, pos_weight):
    """Per-sample BCE, positives multiplied by pos_weight, then averaged."""
    losses = criterion(predictions, targets)
    weights = torch.where(
        targets == 1.0,
        torch.tensor(pos_weight, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    return (losses * weights).mean()


def train_model(model, loaders: dict, pos_weight: float,
                max_epochs: int = config.MAX_EPOCHS,
                patience: int = config.EARLY_STOPPING_PATIENCE,
                verbose: bool = True) -> tuple:
    """
    Train until validation ROC-AUC stops improving for `patience` epochs,
    then restore the best checkpoint. Returns (model, history).
    """
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc, best_state, epochs_without_improvement = -np.inf, None, 0

    for epoch in range(1, max_epochs + 1):
        # ---- training ------------------------------------------------
        model.train()
        running_loss, n_seen = 0.0, 0
        for X_batch, y_batch in loaders["train"]:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = _weighted_bce(criterion, predictions, y_batch, pos_weight)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            n_seen += X_batch.size(0)
        train_loss = running_loss / n_seen

        # ---- validation ----------------------------------------------
        model.eval()
        running_loss, n_seen = 0.0, 0
        val_probabilities, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loaders["val"]:
                predictions = model(X_batch).squeeze(1)
                loss = _weighted_bce(criterion, predictions, y_batch, pos_weight)
                running_loss += loss.item() * X_batch.size(0)
                n_seen += X_batch.size(0)
                val_probabilities.extend(predictions.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        val_loss = running_loss / n_seen
        val_auc = roc_auc_score(val_labels, val_probabilities)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # ---- early stopping / checkpointing --------------------------
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            marker = "  <- best"
        else:
            epochs_without_improvement += 1
            marker = f"  patience {epochs_without_improvement}/{patience}"

        if verbose:
            print(f"epoch {epoch:03d}/{max_epochs}  "
                  f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"val_auc {val_auc:.4f}{marker}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_auc"] = best_auc
    return model, history


def save_checkpoint(model, history: dict, path=None) -> None:
    """Persist weights plus the training curve that produced them."""
    path = config.CHECKPOINT_PATH if path is None else path
    config.ensure_dirs()
    torch.save({"state_dict": model.state_dict(), "history": history}, path)


def load_checkpoint(model, path=None) -> tuple:
    """Load weights into `model`; returns (model, history)."""
    path = config.resolve_input(config.CHECKPOINT_PATH if path is None else path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint.get("history", {})


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Builds train/val/test DataLoaders (batch size 256, shuffle on the
#     training loader only).
# [x] Computes the balanced positive-class weight from the training split
#     to counter the ~86/14 class imbalance.
# [x] Applies that weight per sample on top of BCELoss(reduction='none'),
#     which is the correct pairing for the model's sigmoid output.
# [x] Trains with Adam (lr 1e-3) for up to 100 epochs.
# [x] Selects the epoch with the best VALIDATION ROC-AUC, stops after 10
#     epochs without improvement, and restores that checkpoint.
# [x] Records train loss, val loss and val AUC per epoch in `history`.
# [x] save_checkpoint()/load_checkpoint() store weights and history in
#     project/results/dnn_checkpoint.pt so later phases never retrain.
# [x] Never touches the test set.
# =====================================================================
