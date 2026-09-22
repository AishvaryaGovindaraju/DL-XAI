"""
evaluation.py
=============
Predictive performance of the classifiers - the layer everything else
rests on. If the deep model is not competitive here, no downstream
explanation result is worth reporting, so this is also where the
"is deep learning justified?" comparison is produced.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import config
import uq_methods


def classification_metrics(y_true, probabilities,
                           threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Threshold-free metrics first, then metrics at the chosen threshold."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_predictor(predictor, dataset, split: str = "test",
                       label: str = None) -> dict:
    """Performance plus calibration for one model on one split."""
    probabilities = predictor.predict_proba(dataset.array(split))
    y_true = dataset.y[split]
    row = {"model": label or predictor.name, "dataset": dataset.key, "split": split}
    row.update(classification_metrics(y_true, probabilities))
    row.update(uq_methods.calibration_metrics(y_true, probabilities))
    return row


def evaluate_ensemble(members, dataset, split: str = "test",
                      label: str = "ensemble") -> dict:
    """Ensemble mean prediction, evaluated like any single model."""
    mean = uq_methods.ensemble_samples(members, dataset.array(split)).mean(axis=0)
    y_true = dataset.y[split]
    row = {"model": label, "dataset": dataset.key, "split": split}
    row.update(classification_metrics(y_true, mean))
    row.update(uq_methods.calibration_metrics(y_true, mean))
    return row


def compare_models(deep_results: list, baseline_results: list) -> pd.DataFrame:
    """
    One table per dataset: every model, sorted by ROC-AUC.

    The gap between the best deep model and the best baseline is the
    project's answer to "why deep learning" - reported whichever way it
    comes out.
    """
    table = pd.DataFrame(deep_results + baseline_results)
    return table.sort_values(["dataset", "roc_auc"], ascending=[True, False])


def deep_learning_verdict(table: pd.DataFrame, deep_models=("mlp", "ft_transformer",
                                                            "ensemble")) -> pd.DataFrame:
    """Best deep vs best shallow ROC-AUC per dataset, and the difference."""
    rows = []
    for key, group in table.groupby("dataset"):
        deep = group[group["model"].str.contains("|".join(deep_models))]
        shallow = group[~group["model"].str.contains("|".join(deep_models))]
        if deep.empty or shallow.empty:
            continue
        best_deep = deep.loc[deep["roc_auc"].idxmax()]
        best_shallow = shallow.loc[shallow["roc_auc"].idxmax()]
        rows.append({
            "dataset": key,
            "best_deep": best_deep["model"],
            "deep_roc_auc": best_deep["roc_auc"],
            "best_shallow": best_shallow["model"],
            "shallow_roc_auc": best_shallow["roc_auc"],
            "difference": best_deep["roc_auc"] - best_shallow["roc_auc"],
        })
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, filename: str) -> None:
    config.ensure_dirs()
    table.to_csv(config.TABLES_DIR / filename, index=False)


# ============================================================
# CHECKLIST
# - classification_metrics(): ROC-AUC, PR-AUC, accuracy, precision,
#   recall, F1 and the explicit confusion-matrix counts
# - Every model is also scored for calibration (ECE, Brier) in the same row
# - evaluate_ensemble(): scores the deep ensemble's mean prediction
# - compare_models() builds the per-dataset leaderboard
# - deep_learning_verdict() reports best deep vs best shallow AUC and the
#   gap, honestly, in whichever direction it falls
# - save_table() writes to results/tables/
# ============================================================
