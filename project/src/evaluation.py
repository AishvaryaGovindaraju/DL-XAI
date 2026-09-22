"""
evaluation.py
=============
Phase 4c - deterministic test-set performance of the trained DNN.

"Deterministic" matters: this file runs the model in eval() mode, so dropout
is OFF and every prediction is a single number. The stochastic version lives
in mc_dropout.py and is a different question (how sure is the model?), not a
different accuracy metric.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config


def predict_probabilities(model, loader) -> tuple:
    """Deterministic forward pass. Returns (probabilities, true_labels)."""
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probabilities.extend(model(X_batch).squeeze(1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
    return np.asarray(probabilities), np.asarray(labels)


def classification_metrics(y_true, probabilities,
                           threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Threshold-free (ROC-AUC, PR-AUC) plus threshold-based metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": threshold,
    }


def check_auc_gate(metrics: dict, minimum: float = config.MIN_ACCEPTABLE_AUC) -> bool:
    """
    PRD gate: the later phases are only meaningful if the model is usable.
    Returns True when test ROC-AUC clears the minimum; warns when it does not.
    """
    passed = metrics["roc_auc"] >= minimum
    if not passed:
        print(f"WARNING: test ROC-AUC {metrics['roc_auc']:.4f} is below the "
              f"minimum acceptable {minimum:.2f}. Retune before Phase 5.")
    return passed


def save_metrics(metrics: dict, path=None) -> None:
    """Write the metric row to results/dnn_metrics.csv."""
    path = config.DNN_METRICS_PATH if path is None else path
    config.ensure_dirs()
    pd.DataFrame([metrics]).to_csv(path, index=False)


def save_test_predictions(probabilities, y_true, path=None) -> None:
    """Per-instance deterministic predictions, keyed by test_instance_id."""
    path = config.TEST_PREDICTIONS_PATH if path is None else path
    config.ensure_dirs()
    pd.DataFrame({
        "test_instance_id": np.arange(len(y_true)),
        "true_label": y_true.astype(int),
        "predicted_probability": probabilities,
    }).to_csv(path, index=False)


def format_metrics(metrics: dict) -> str:
    """Readable block for the console and the paper's results table."""
    return (
        f"ROC-AUC   {metrics['roc_auc']:.4f}\n"
        f"PR-AUC    {metrics['pr_auc']:.4f}\n"
        f"Accuracy  {metrics['accuracy']:.4f}\n"
        f"Precision {metrics['precision']:.4f}\n"
        f"Recall    {metrics['recall']:.4f}\n"
        f"F1        {metrics['f1']:.4f}\n"
        f"Confusion matrix: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Runs the trained model over the test loader in eval() mode, so
#     dropout is disabled and predictions are deterministic.
# [x] Computes threshold-free metrics (ROC-AUC, PR-AUC) and, at the 0.5
#     threshold, accuracy / precision / recall / F1.
# [x] Reports the confusion matrix as explicit TN, FP, FN, TP counts -
#     important because the classes are imbalanced.
# [x] check_auc_gate() enforces the PRD's 0.72 minimum ROC-AUC before the
#     uncertainty and XAI phases are allowed to mean anything.
# [x] Saves one metric row to results/dnn_metrics.csv and per-instance
#     predictions to results/test_predictions.csv (keyed by the same
#     test_instance_id the XAI phases use).
# [x] Stochastic/MC-Dropout evaluation is deliberately NOT here; see
#     mc_dropout.py.
# =====================================================================
