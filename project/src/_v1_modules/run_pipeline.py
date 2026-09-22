"""
run_pipeline.py
===============
The whole study, in order, in one file. Read main() to see the workflow;
read the individual modules to see how each step works.

    raw CSV
      -> preprocessing        (70/15/15 split, scaler fitted on train only)
      -> model + training     (class-weighted DNN, early stopping on val AUC)
      -> evaluation           (deterministic test metrics, 0.72 AUC gate)
      -> mc_dropout           (T=50 stochastic passes -> per-instance variance)
      -> stratification       (LOW/MEDIUM/HIGH + 600-instance sample)
      -> explain_shap / explain_lime / explain_dice  (same 600 instances)

Usage
-----
    python run_pipeline.py                 # everything
    python run_pipeline.py --stages train mc strata
    python run_pipeline.py --stages shap lime dice   # reuses the checkpoint

Stages after `train` reload the saved checkpoint, so the network is trained
once and every explanation phase runs against the same weights.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data_loading
import evaluation
import explain_dice
import explain_lime
import explain_shap
import mc_dropout
import model as model_module
import preprocessing
import stratification

STAGES = ["data", "train", "evaluate", "mc", "strata", "shap", "lime", "dice"]


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main(stages=None) -> dict:
    stages = STAGES if not stages else stages
    config.ensure_dirs()
    config.set_seeds()
    state = {}

    # ---- data ---------------------------------------------------------
    section("PHASE 1-2  data loading and preprocessing")
    df = data_loading.run_eda(verbose="data" in stages)
    splits = preprocessing.build_splits(df)
    loaders = None
    print(f"train {splits.X_train.shape}  val {splits.X_val.shape}  "
          f"test {splits.X_test.shape}")
    state["splits"] = splits

    if stages == ["data"]:
        return state

    from training import (load_checkpoint, make_loaders, positive_class_weight,
                          save_checkpoint, train_model)
    loaders = make_loaders(splits)
    network = model_module.build_model()

    # ---- train --------------------------------------------------------
    if "train" in stages:
        section("PHASE 4  DNN training")
        print(model_module.describe(network))
        pos_weight = positive_class_weight(splits.y_train)
        print(f"positive-class weight: {pos_weight:.4f}\n")
        network, history = train_model(network, loaders, pos_weight)
        save_checkpoint(network, history)
    else:
        network, history = load_checkpoint(network)
        print(f"loaded checkpoint: {config.CHECKPOINT_PATH.name}")
    state["model"], state["history"] = network, history

    # ---- evaluate -----------------------------------------------------
    if "evaluate" in stages:
        section("PHASE 4  test-set evaluation (dropout off)")
        probabilities, labels = evaluation.predict_probabilities(
            network, loaders["test"])
        metrics = evaluation.classification_metrics(labels, probabilities)
        print(evaluation.format_metrics(metrics))
        evaluation.check_auc_gate(metrics)
        evaluation.save_metrics(metrics)
        evaluation.save_test_predictions(probabilities, labels)
        state["metrics"] = metrics

    # ---- MC dropout ---------------------------------------------------
    if "mc" in stages or "strata" in stages:
        section("PHASE 4  MC Dropout uncertainty (T = 50)")
        mc_predictions = mc_dropout.mc_predict(network, loaders["test"],
                                               verbose=True)
        mc_dropout.check_mc_output(mc_predictions, len(splits.y_test))
        uncertainty = mc_dropout.summarise(mc_predictions)
        print("variance percentiles:",
              mc_dropout.variance_percentiles(uncertainty["variance"]))
        state["uncertainty"] = uncertainty

    # ---- stratification ----------------------------------------------
    if "strata" in stages:
        section("PHASE 5  uncertainty stratification")
        table, sample = stratification.run_stratification(
            state["uncertainty"]["variance"], state["uncertainty"]["mean"])
        state["stratification"], state["sample"] = table, sample

    # ---- explanations -------------------------------------------------
    explanation_stages = [s for s in ("shap", "lime", "dice") if s in stages]
    if not explanation_stages:
        return state

    sample = state.get("sample", stratification.load_sample())
    instance_ids = sample["test_instance_id"].to_numpy()
    X_sample = splits.X_test.iloc[instance_ids].copy()
    y_sample = splits.y_test[instance_ids]

    if "shap" in stages:
        section("PHASE 6A  SHAP")
        values, _ = explain_shap.explain_instances(network, splits.X_train, X_sample)
        explain_shap.save_outputs(values, instance_ids)
        print("\ntop features by mean |SHAP|:")
        print(explain_shap.mean_absolute_importance(
            values, splits.feature_names).head(10))
        state["shap_values"] = values

    if "lime" in stages:
        section("PHASE 6B  LIME")
        outputs = explain_lime.explain_instances(
            network, splits.X_train, X_sample, sample, y_sample)
        explain_lime.save_outputs(outputs)
        state["lime_outputs"] = outputs

    if "dice" in stages:
        section("PHASE 6C  DiCE counterfactuals")
        outputs = explain_dice.generate_for_sample(
            network, splits.X_train, splits.y_train,
            splits.X_test, splits.y_test, sample)
        explain_dice.save_outputs(outputs)
        state["dice_outputs"] = outputs

    return state


def parse_args():
    parser = argparse.ArgumentParser(description="Run the UA-XAI pipeline.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                        help=f"subset of: {' '.join(STAGES)} (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args().stages)


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Shows the full workflow in one readable function: data ->
#     preprocessing -> training -> evaluation -> MC Dropout ->
#     stratification -> SHAP / LIME / DiCE.
# [x] Creates output folders and sets all random seeds before anything
#     runs.
# [x] --stages lets any subset be re-run; stages after training reload
#     the saved checkpoint instead of retraining, so every explanation
#     phase uses identical weights.
# [x] Passes the SAME 600 stratified instances to SHAP, LIME and DiCE,
#     selected by test_instance_id from the saved sample.
# [x] Prints one clearly headed block per phase and returns a `state`
#     dict, so the pipeline can also be driven from a notebook.
# [x] Not implemented here, matching the current codebase: Phase 3
#     (logistic-regression and XGBoost baselines), Phase 7 (XAI quality
#     metrics) and Phase 8 (statistical analysis and figures).
# =====================================================================
