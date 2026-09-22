"""
explain_dice.py
===============
Phase 6C - DiCE counterfactuals for the same 600 sampled test instances.

SHAP and LIME say which features mattered. DiCE answers a different question:
what would have to change for this prediction to flip? Age, Sex and Education
are locked, because a recommendation to change them is not actionable.

Failures are data, not bugs
---------------------------
For some instances DiCE cannot find a valid counterfactual. The framework
predicts that this happens more often in the HIGH-uncertainty stratum, so
failures are counted and kept per instance instead of being silently dropped.
"""

import pickle
import time

import numpy as np
import pandas as pd
import torch
from dice_ml import Dice, Data
from dice_ml.constants import ModelTypes
from dice_ml.model_interfaces.pytorch_model import PyTorchModel

import config


class BinaryPyTorchModel(PyTorchModel):
    """
    DiCE expects a two-column class-probability output, but the network has
    a single sigmoid unit. This adapter turns p into [1 - p, p] for both the
    numpy and the tensor code paths DiCE uses.
    """

    def get_output(self, input_instance, model_score=True,
                   transform_data=False, out_tensor=False):
        output = super().get_output(input_instance, model_score=model_score,
                                    transform_data=transform_data,
                                    out_tensor=out_tensor)
        if out_tensor:
            p_positive = output.view(-1, 1)
            return torch.cat([1.0 - p_positive, p_positive], dim=1)
        p_positive = np.asarray(output).reshape(-1, 1)
        return np.hstack([1.0 - p_positive, p_positive])


def build_data_interface(X_train, y_train) -> Data:
    """
    Reference data for DiCE's sampling ranges.

    All 21 columns are declared continuous: they are already numeric codes
    fed straight to the network, and declaring them categorical triggers a
    string-dtype incompatibility in recent pandas.
    """
    frame = X_train.copy()
    frame[config.TARGET_COLUMN] = y_train.astype(int)
    return Data(dataframe=frame,
                continuous_features=X_train.columns.tolist(),
                outcome_name=config.TARGET_COLUMN)


def mutable_features(feature_names) -> list:
    """Every feature except the PRD's immutable ones."""
    return [f for f in feature_names if f not in config.DICE_IMMUTABLE_FEATURES]


def build_explainer(model, X_train, y_train) -> tuple:
    """Returns (dice_explainer, dice_model_wrapper)."""
    model.eval()
    dice_model = BinaryPyTorchModel(model=model, model_path="", backend="PYT")
    dice_model.model_type = ModelTypes.Classifier
    explainer = Dice(build_data_interface(X_train, y_train), dice_model,
                     method=config.DICE_METHOD)
    return explainer, dice_model


def generate_one(explainer, query_instance: pd.DataFrame,
                 features_to_vary: list) -> pd.DataFrame:
    """Counterfactuals for one instance, or None if DiCE cannot find any."""
    result = explainer.generate_counterfactuals(
        query_instance,
        total_CFs=config.DICE_TOTAL_CFS,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        verbose=False,
    )
    return result.cf_examples_list[0].final_cfs_df


def immutables_respected(original: pd.DataFrame, counterfactuals: pd.DataFrame) -> bool:
    """Verify Age, Sex and Education were not altered."""
    if counterfactuals is None:
        return True
    return all(
        np.allclose(counterfactuals[feature].astype(float),
                    float(original.iloc[0][feature]))
        for feature in config.DICE_IMMUTABLE_FEATURES
    )


def generate_for_sample(model, X_train, y_train, X_test, y_test,
                        sample_table: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Generate counterfactuals for every sampled instance.

    Returns {test_instance_id: {...}}; `counterfactuals` is None and `error`
    is filled in when generation failed for that instance.
    """
    explainer, _ = build_explainer(model, X_train, y_train)
    features_to_vary = mutable_features(X_train.columns)

    outputs, failures = {}, 0
    started = time.time()
    for position, row in enumerate(sample_table.itertuples(index=False), start=1):
        instance_id = int(row.test_instance_id)
        query_instance = X_test.iloc[[instance_id]]

        record = {
            "test_instance_id": instance_id,
            "sigma_squared": float(row.sigma_squared),
            "uncertainty_stratum": str(row.uncertainty_stratum),
            "true_label": int(y_test[instance_id]),
            "counterfactuals": None,
            "immutables_respected": None,
            "error": None,
        }
        try:
            counterfactuals = generate_one(explainer, query_instance, features_to_vary)
            record["counterfactuals"] = counterfactuals
            record["immutables_respected"] = immutables_respected(
                query_instance, counterfactuals)
        except Exception as error:                      # noqa: BLE001
            record["error"] = f"{type(error).__name__}: {error}"
            failures += 1

        outputs[instance_id] = record
        if verbose and position % 50 == 0:
            print(f"DiCE {position}/{len(sample_table)}  "
                  f"failures {failures}  ({time.time() - started:.0f}s)")

    if verbose:
        print(f"\nDiCE finished: {len(outputs) - failures} succeeded, "
              f"{failures} failed")
        print(failure_rate_by_stratum(outputs))
    return outputs


def failure_rate_by_stratum(outputs: dict) -> pd.DataFrame:
    """Success / failure counts per stratum - the H3 result."""
    frame = pd.DataFrame([
        {"uncertainty_stratum": record["uncertainty_stratum"],
         "failed": record["counterfactuals"] is None}
        for record in outputs.values()
    ])
    summary = frame.groupby("uncertainty_stratum")["failed"].agg(["size", "sum"])
    summary.columns = ["n", "n_failed"]
    summary["failure_rate"] = (summary["n_failed"] / summary["n"]).round(4)
    return summary.reindex(list(config.STRATUM_NAMES))


def save_outputs(outputs: dict) -> None:
    config.ensure_dirs()
    with open(config.DICE_OUTPUTS_PATH, "wb") as handle:
        pickle.dump(outputs, handle)


def load_outputs() -> dict:
    with open(config.resolve_input(config.DICE_OUTPUTS_PATH), "rb") as handle:
        return pickle.load(handle)


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Adapts the single-sigmoid network to the two-column probability
#     output DiCE expects (BinaryPyTorchModel), for both numpy and tensor
#     paths.
# [x] Builds the DiCE data interface from the TRAINING split, declaring
#     all 21 columns continuous to avoid the categorical/pandas dtype
#     incompatibility.
# [x] Locks Age, Sex and Education so counterfactuals stay actionable;
#     immutables_respected() verifies that per instance instead of
#     trusting the library.
# [x] Generates 3 opposite-class counterfactuals per instance with
#     method='random' for the same 600 stratified instances.
# [x] Catches per-instance failures, records the error text, and keeps the
#     row - failures are a reported finding (expected to concentrate in
#     the HIGH stratum), not something to debug away.
# [x] failure_rate_by_stratum() summarises those failures per stratum.
# [x] Saves dice_outputs.pkl keyed by test_instance_id, with uncertainty
#     metadata attached.
# =====================================================================
