"""
experiments.py
==============
Orchestration. Everything else defines a capability; this file defines the
study.

The chain, once per (dataset, architecture, seed):

    data -> deep ensemble + baselines      E1  predictive + calibration
         -> uncertainty decomposition      E2  does uncertainty rank risk?
         -> explanations on calib + test   (SHAP, LIME, IG, DiCE)
         -> measured explanation risk
         -> quality head on calib          E3  H1: representation vs uncertainty
         -> conformal risk control         E4  H2: policies at matched coverage
                                           E5  H3: is the bound respected?
         -> abstention analysis            E6  H4: where does it abstain?
         -> ablations

Why the calibration split is explained twice as densely as the test split:
the quality head is fitted on one half of the calibration explanations and
the conformal threshold on the other. Nothing that touches the test split
is ever used for fitting.
"""

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd

import config
import data as data_module
import evaluation
import models as models_module
import selective
import statistical_tests as stats_module
import training
import uq_methods
import xai
import xai_metrics


@dataclass
class RunContext:
    """Everything one (dataset, architecture, seed) run produces."""

    dataset_key: str
    architecture: str
    seed: int
    tables: dict
    artefacts: dict


# ---------------------------------------------------------------------
# Building the measured-risk table for one split
# ---------------------------------------------------------------------

def measure_split(predictor, dataset, instance_ids, split: str,
                  methods=None, include_dice: bool = True,
                  seed: int = config.RANDOM_STATE,
                  verbose: bool = True) -> pd.DataFrame:
    """Explain the given instances and measure every explanation."""
    X = dataset.array(split)[instance_ids]
    explanations = xai.explain_instances(predictor, dataset, instance_ids,
                                         methods=methods,
                                         include_dice=include_dice,
                                         split=split, verbose=verbose)
    explainers = xai.build_attribution_explainers(predictor, dataset, methods)

    frames = []
    for method, records in explanations.items():
        if method in config.COUNTERFACTUAL_METHODS:
            frames.append(xai_metrics.measure_counterfactuals(
                records, predictor, X, dataset))
        else:
            frames.append(xai_metrics.measure_attributions(
                records, explainers[method], predictor, X, seed=seed))
    table = pd.concat(frames, ignore_index=True)
    table["split"] = split
    return table


def risks_from_measurements(calib_table: pd.DataFrame, test_table: pd.DataFrame,
                            head_fraction: float = 0.5) -> dict:
    """
    Split the calibration measurements in two, normalise risk on the first
    half, and return aligned instance x method risk matrices.
    """
    calib_ids = np.sort(calib_table["instance_id"].unique())
    cut = int(len(calib_ids) * head_fraction)
    head_ids, crc_ids = calib_ids[:cut], calib_ids[cut:]

    normaliser = xai_metrics.RiskNormaliser().fit(
        calib_table[calib_table["instance_id"].isin(head_ids)],
        xai_metrics.RISK_COMPONENTS)

    calib_scored = xai_metrics.attach_risk(calib_table, normaliser)
    test_scored = xai_metrics.attach_risk(test_table, normaliser)

    head_matrix = xai_metrics.risk_matrix(
        calib_scored[calib_scored["instance_id"].isin(head_ids)])
    crc_matrix = xai_metrics.risk_matrix(
        calib_scored[calib_scored["instance_id"].isin(crc_ids)])
    test_matrix = xai_metrics.risk_matrix(test_scored)

    methods = [m for m in config.ALL_METHODS if m in test_matrix.columns]
    return {"normaliser": normaliser,
            "calib_scored": calib_scored, "test_scored": test_scored,
            "head_ids": head_ids, "crc_ids": crc_ids,
            "head_risk": head_matrix[methods], "crc_risk": crc_matrix[methods],
            "test_risk": test_matrix[methods], "methods": methods}


# ---------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------

def run_experiment(dataset_key: str, architecture: str = "mlp", seed: int = 0,
                   n_explain: int = config.N_EXPLAIN,
                   ensemble_size: int = config.ENSEMBLE_SIZE,
                   uq_method: str = "ensemble",
                   include_dice: bool = True,
                   with_baselines: bool = True,
                   verbose: bool = True) -> RunContext:
    started = time.time()
    config.ensure_dirs()
    config.set_seeds(seed)
    tables, artefacts = {}, {}

    def say(message):
        if verbose:
            print(message)

    # ---- data -------------------------------------------------------
    say(f"\n[{dataset_key} | {architecture} | seed {seed}] loading data")
    dataset = data_module.load_dataset(dataset_key, seed=seed)
    tables["dataset_summary"] = pd.DataFrame([dataset.summary()])
    say(f"  {dataset.n_features} features, "
        f"{len(dataset.y['train'])} train / {len(dataset.y['test'])} test")

    # ---- E1: models and predictive performance ----------------------
    say("  training deep ensemble")
    members, histories = training.train_ensemble(dataset, architecture, seed=seed,
                                                 n_members=ensemble_size,
                                                 verbose=verbose)
    predictor = members[0]
    artefacts["members"] = members

    performance = [evaluation.evaluate_predictor(m, dataset,
                                                 label=f"{architecture}_member{k}")
                   for k, m in enumerate(members[:1])]
    performance.append(evaluation.evaluate_ensemble(members, dataset,
                                                    label=f"{architecture}_ensemble"))
    baseline_rows = []
    if with_baselines:
        say("  fitting baselines")
        for name, model in training.fit_baselines(dataset, seed=seed).items():
            baseline_rows.append(evaluation.evaluate_predictor(model, dataset,
                                                               label=name))
    tables["performance"] = evaluation.compare_models(performance, baseline_rows)
    tables["dl_verdict"] = evaluation.deep_learning_verdict(tables["performance"])
    say("  best ROC-AUC: " +
        ", ".join(f"{r['model']} {r['roc_auc']:.4f}"
                  for _, r in tables["performance"].head(3).iterrows()))

    # ---- uncertainty -------------------------------------------------
    say(f"  uncertainty ({uq_method})")
    source = members if uq_method == "ensemble" else predictor
    uncertainty = {}
    for split in ("calib", "test"):
        samples = uq_methods.get_samples(uq_method, source, dataset.array(split))
        uncertainty[split] = uq_methods.decompose(samples)
    tables["calibration"] = pd.DataFrame([{
        "dataset": dataset_key, "architecture": architecture, "seed": seed,
        "uq_method": uq_method,
        **uq_methods.calibration_metrics(dataset.y["test"], uncertainty["test"]["mean"]),
        "temperature": uq_methods.fit_temperature(
            uncertainty["calib"]["mean"], dataset.y["calib"]),
    }])

    # ---- explanations and measured risk ------------------------------
    calib_ids = xai.select_instances(dataset, n=2 * n_explain, seed=seed, split="calib")
    test_ids = xai.select_instances(dataset, n=n_explain, seed=seed + 1, split="test")

    say(f"  explaining {len(calib_ids)} calibration instances")
    calib_measurements = measure_split(predictor, dataset, calib_ids, "calib",
                                       include_dice=include_dice, seed=seed,
                                       verbose=verbose)
    say(f"  explaining {len(test_ids)} test instances")
    test_measurements = measure_split(predictor, dataset, test_ids, "test",
                                      include_dice=include_dice, seed=seed,
                                      verbose=verbose)

    risks = risks_from_measurements(calib_measurements, test_measurements)
    methods = risks["methods"]
    tables["explanation_quality"] = xai_metrics.summarise_by_method(
        risks["test_scored"]).reset_index()
    artefacts["risks"] = risks

    # ---- quality head ------------------------------------------------
    say("  fitting explanation-quality head")
    def features_for(split, ids, use_representation=True):
        index = {value: position for position, value
                 in enumerate(range(len(dataset.y[split])))}
        rows = np.asarray([index[i] for i in ids])
        subset = {key: value[rows] for key, value in uncertainty[split].items()}
        return selective.build_quality_features(
            predictor, dataset.array(split)[rows], subset,
            use_representation=use_representation)

    head_features = features_for("calib", risks["head_risk"].index)
    crc_features = features_for("calib", risks["crc_risk"].index)
    test_features = features_for("test", risks["test_risk"].index)

    head, normalisation = fit_head_and_check(head_features, risks["head_risk"], seed)
    predicted_crc = selective.predict_risk(head, normalisation, crc_features)
    predicted_test = selective.predict_risk(head, normalisation, test_features)
    artefacts["quality_head"] = head

    # ---- E3 / H1: is the representation a better predictor? ----------
    epistemic_test = uncertainty["test"]["epistemic"][
        [i for i in risks["test_risk"].index]]
    epistemic_crc = uncertainty["calib"]["epistemic"][
        [i for i in risks["crc_risk"].index]]

    best_realised = np.nanmin(risks["test_risk"].to_numpy(dtype=float), axis=1)
    predicted_best = predicted_test.min(axis=1)
    tables["h1_risk_prediction"] = pd.DataFrame([
        {"predictor": "epistemic_uncertainty",
         **stats_module.spearman_with_ci(epistemic_test, best_realised)},
        {"predictor": "quality_head",
         **stats_module.spearman_with_ci(predicted_best, best_realised)},
    ])
    tables["h1_risk_prediction"]["dataset"] = dataset_key
    tables["h1_risk_prediction"]["architecture"] = architecture
    tables["h1_risk_prediction"]["seed"] = seed

    # ---- E2: replication of the published correlation ----------------
    rows = []
    for position, method in enumerate(methods):
        realised = risks["test_risk"].to_numpy(dtype=float)[:, position]
        rows.append({"method": method,
                     **stats_module.spearman_with_ci(epistemic_test, realised)})
    tables["e2_uncertainty_vs_risk"] = pd.DataFrame(rows)
    tables["e2_uncertainty_vs_risk"]["dataset"] = dataset_key
    tables["e2_uncertainty_vs_risk"]["seed"] = seed

    # ---- E4 / E5: policies and the guarantee -------------------------
    say("  evaluating policies")
    policy_frames = []
    for alpha in config.RISK_LEVELS:
        frame = selective.evaluate_policies(
            risks["test_risk"].to_numpy(dtype=float), predicted_test, epistemic_test,
            risks["crc_risk"].to_numpy(dtype=float), predicted_crc, epistemic_crc,
            method_names=methods, alpha=alpha, seed=seed)
        policy_frames.append(frame)
    tables["policies"] = pd.concat(policy_frames, ignore_index=True)
    tables["policies"]["dataset"] = dataset_key
    tables["policies"]["architecture"] = architecture
    tables["policies"]["seed"] = seed

    tables["coverage_risk_curve"] = selective.coverage_risk_curve(
        predicted_test, risks["test_risk"].to_numpy(dtype=float),
        predicted_crc, risks["crc_risk"].to_numpy(dtype=float))
    tables["coverage_risk_curve"]["dataset"] = dataset_key
    tables["coverage_risk_curve"]["architecture"] = architecture
    tables["coverage_risk_curve"]["seed"] = seed

    # ---- E6 / H4: where does the system abstain? ---------------------
    threshold = selective.crc_threshold(
        predicted_crc.min(axis=1),
        selective.realised_risk_of_choice(risks["crc_risk"].to_numpy(dtype=float),
                                          predicted_crc.argmin(axis=1)),
        alpha=config.DEFAULT_ALPHA)
    _, delivered = selective.policy_cse(predicted_test, threshold)
    tables["h4_abstention"] = pd.DataFrame([{
        "dataset": dataset_key, "architecture": architecture, "seed": seed,
        "alpha": config.DEFAULT_ALPHA,
        "coverage": float(delivered.mean()),
        "epistemic_delivered": float(np.mean(epistemic_test[delivered]))
                               if delivered.any() else np.nan,
        "epistemic_abstained": float(np.mean(epistemic_test[~delivered]))
                               if (~delivered).any() else np.nan,
        "cliffs_delta": stats_module.cliffs_delta(epistemic_test[~delivered],
                                                  epistemic_test[delivered]),
    }])

    # ---- policy comparison statistics --------------------------------
    realised_matrix = risks["test_risk"].to_numpy(dtype=float)
    risk_by_policy = {}
    cse_choice, cse_delivered = selective.policy_cse(predicted_test, threshold)
    risk_by_policy["cse"] = np.where(
        cse_delivered,
        selective.realised_risk_of_choice(realised_matrix, cse_choice), 0.0)
    for position, method in enumerate(methods):
        risk_by_policy[f"fixed_{method}"] = realised_matrix[:, position]
    tables["policy_tests"] = stats_module.compare_all_policies(risk_by_policy)
    tables["policy_tests"]["dataset"] = dataset_key
    tables["policy_tests"]["seed"] = seed

    # ---- ablations ---------------------------------------------------
    say("  ablations")
    tables["ablations"] = run_ablations(
        predictor, dataset, uncertainty, risks, features_for, seed, methods)
    tables["ablations"]["dataset"] = dataset_key
    tables["ablations"]["architecture"] = architecture
    tables["ablations"]["seed"] = seed

    say(f"  done in {time.time() - started:.0f}s")
    return RunContext(dataset_key, architecture, seed, tables, artefacts)


def fit_head_and_check(features, risk_frame, seed):
    head, normalisation = selective.fit_quality_head(
        features, risk_frame.to_numpy(dtype=float), seed=seed)
    assert np.isfinite(normalisation["loss"][-1]), "quality head diverged"
    return head, normalisation


# ---------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------

def run_ablations(predictor, dataset, uncertainty, risks, features_for,
                  seed: int, methods: list) -> pd.DataFrame:
    """
    Three questions a reviewer will ask:
      1. does the representation matter, or is scalar uncertainty enough?
      2. does the result survive a different risk weighting?
      3. does the selective policy beat the best single method at the same
         coverage?
    """
    realised_test = risks["test_risk"].to_numpy(dtype=float)
    realised_crc = risks["crc_risk"].to_numpy(dtype=float)
    rows = []

    for label, use_representation in (("representation+uncertainty", True),
                                      ("uncertainty_only", False)):
        head_features = features_for("calib", risks["head_risk"].index,
                                     use_representation)
        crc_features = features_for("calib", risks["crc_risk"].index,
                                    use_representation)
        test_features = features_for("test", risks["test_risk"].index,
                                     use_representation)
        head, normalisation = selective.fit_quality_head(
            head_features, risks["head_risk"].to_numpy(dtype=float), seed=seed)
        predicted_test = selective.predict_risk(head, normalisation, test_features)
        predicted_crc = selective.predict_risk(head, normalisation, crc_features)

        threshold = selective.crc_threshold(
            predicted_crc.min(axis=1),
            selective.realised_risk_of_choice(realised_crc,
                                              predicted_crc.argmin(axis=1)),
            alpha=config.DEFAULT_ALPHA)
        choice, delivered = selective.policy_cse(predicted_test, threshold)
        realised = selective.realised_risk_of_choice(realised_test, choice)
        rows.append({"ablation": "features", "variant": label,
                     **selective.empirical_risk(realised, delivered),
                     "spearman_rho": stats_module.spearman_with_ci(
                         predicted_test.min(axis=1),
                         np.nanmin(realised_test, axis=1))["rho"]})

    original = dict(config.RISK_WEIGHTS_ATTRIBUTION)
    for label, weights in (("infidelity_heavy", {"infidelity": 0.8, "instability": 0.2}),
                           ("instability_heavy", {"infidelity": 0.2, "instability": 0.8}),
                           ("equal", {"infidelity": 0.5, "instability": 0.5})):
        config.RISK_WEIGHTS_ATTRIBUTION.update(weights)
        normaliser = risks["normaliser"]
        rescored = xai_metrics.attach_risk(risks["test_scored"], normaliser)
        matrix = xai_metrics.risk_matrix(rescored)[methods].to_numpy(dtype=float)
        rows.append({"ablation": "risk_weights", "variant": label,
                     "coverage": np.nan, "population_risk": np.nan,
                     "selective_risk": float(np.nanmean(np.nanmin(matrix, axis=1))),
                     "n": len(matrix), "spearman_rho": np.nan})
    config.RISK_WEIGHTS_ATTRIBUTION.update(original)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Multi-run driver
# ---------------------------------------------------------------------

def run_all(dataset_keys=None, architectures=("mlp",), seeds=None,
            save: bool = True, **kwargs) -> dict:
    """Run every combination and concatenate the tables across runs."""
    dataset_keys = dataset_keys or config.DEFAULT_DATASETS
    seeds = seeds if seeds is not None else config.SEEDS
    collected = {}
    for dataset_key in dataset_keys:
        for architecture in architectures:
            for seed in seeds:
                context = run_experiment(dataset_key, architecture, seed, **kwargs)
                for name, table in context.tables.items():
                    table = table.copy()
                    table["dataset"] = dataset_key
                    table["architecture"] = architecture
                    table["seed"] = seed
                    collected.setdefault(name, []).append(table)
    merged = {name: pd.concat(frames, ignore_index=True)
              for name, frames in collected.items()}
    if save:
        config.ensure_dirs()
        for name, table in merged.items():
            table.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    return merged


# ============================================================
# CHECKLIST
# - run_experiment() is the whole study for one dataset x architecture x
#   seed, in the order the paper reports it
# - E1 predictive performance and calibration for the deep ensemble and the
#   shallow baselines, plus the explicit deep-vs-shallow verdict
# - Explanations are computed twice: on the calibration split (half to fit
#   the quality head, half to calibrate the conformal threshold) and on the
#   test split (evaluation only)
# - E2 replicates the published epistemic-uncertainty vs explanation-risk
#   correlation, per method
# - E3/H1 compares the learned quality head against epistemic uncertainty
#   as a predictor of explanation risk
# - E4/E5 run every policy at five risk levels and check whether the
#   conformal bound held
# - E6/H4 tests whether abstention concentrates on high-epistemic cases
# - run_ablations(): representation vs uncertainty-only features, and
#   sensitivity to the risk weighting
# - run_all() loops datasets/architectures/seeds and writes every table to
#   results/tables/
# ============================================================
