"""
explain_shap.py
===============
Phase 6A - SHAP attributions for the 600 sampled test instances.

SHAP answers "how much did each feature push this prediction away from the
average prediction?", with the additivity guarantee
base_value + sum(shap_values) = model output. That guarantee is checked here
rather than assumed.
"""

import numpy as np
import shap
import torch

import config


def make_predict_fn(model):
    """Wrap the network as the plain numpy -> P(diabetes) function SHAP wants."""
    def predict(data: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(data, dtype=torch.float32)
            return model(tensor).squeeze(1).cpu().numpy()
    return predict


def build_background(X_train, size: int = config.SHAP_BACKGROUND_SIZE,
                     seed: int = config.RANDOM_STATE) -> np.ndarray:
    """
    Reference distribution SHAP measures deviations from: a random sample of
    training rows. Fixed seed, so attributions are reproducible.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_train), size=size, replace=False)
    return X_train.iloc[indices].to_numpy(dtype=np.float32)


def build_explainer(model, X_train, feature_names):
    """Model-agnostic SHAP explainer over the background sample."""
    return shap.Explainer(
        make_predict_fn(model),
        build_background(X_train),
        feature_names=list(feature_names),
    )


def additivity_error(explanation, predict_fn, X: np.ndarray) -> np.ndarray:
    """|base + sum(shap) - model output| per instance; should be ~0."""
    base = np.asarray(explanation.base_values).reshape(-1)
    reconstructed = base + explanation.values.sum(axis=1)
    return np.abs(reconstructed - predict_fn(X))


def explain_instances(model, X_train, X_sample, verbose: bool = True):
    """
    Compute SHAP values for the sampled instances.

    Returns (shap_values array of shape (n_instances, 21), explanation object).
    """
    predict_fn = make_predict_fn(model)
    explainer = build_explainer(model, X_train, X_sample.columns)

    X_array = X_sample.to_numpy(dtype=np.float32)
    explanation = explainer(X_array)
    values = explanation.values

    assert values.shape == (len(X_sample), config.N_FEATURES), (
        f"unexpected SHAP shape {values.shape}")
    assert not np.isnan(values).any(), "SHAP values contain NaN"

    if verbose:
        errors = additivity_error(explanation, predict_fn, X_array)
        print(f"SHAP values: {values.shape}")
        print(f"additivity error - max {errors.max():.2e}, "
              f"mean {errors.mean():.2e}")
    return values, explanation


def save_outputs(values: np.ndarray, instance_ids: np.ndarray) -> None:
    """Store attributions and the instance ids they line up with."""
    config.ensure_dirs()
    np.save(config.SHAP_VALUES_PATH, values)
    np.save(config.SHAP_IDS_PATH, instance_ids)


def load_outputs() -> tuple:
    """Load (shap_values, instance_ids) saved by a previous run."""
    return (np.load(config.resolve_input(config.SHAP_VALUES_PATH)),
            np.load(config.resolve_input(config.SHAP_IDS_PATH)))


def mean_absolute_importance(values: np.ndarray, feature_names) -> "pd.Series":
    """Global feature importance: mean |SHAP value| per feature."""
    import pandas as pd
    return pd.Series(np.abs(values).mean(axis=0),
                     index=list(feature_names)).sort_values(ascending=False)


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Wraps the PyTorch model as a numpy prediction function (eval mode,
#     no gradients) so SHAP can treat it as a black box.
# [x] Builds a fixed-seed 500-row background sample from TRAINING data -
#     the reference the attributions are measured against.
# [x] Creates a model-agnostic shap.Explainer with real feature names.
# [x] Explains the 600 stratified instances and returns a (600, 21)
#     attribution matrix.
# [x] Asserts the output shape and absence of NaN.
# [x] Checks SHAP's additivity property explicitly (base + sum of
#     attributions should reproduce the model output) instead of assuming
#     the explainer behaved.
# [x] Saves shap_values.npy and shap_instance_ids.npy so the attributions
#     stay aligned to test_instance_id.
# [x] mean_absolute_importance() gives the global ranking for the paper.
# =====================================================================
