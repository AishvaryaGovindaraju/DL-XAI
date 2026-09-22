"""
xai_metrics.py
==============
Measuring explanations, and turning those measurements into one number per
(instance, method) that the selective policy can act on.

Per-instance properties measured
--------------------------------
attributions      infidelity   (Yeh et al., 2019) does the attribution
                               predict what happens when features are
                               perturbed?
                  instability  (Alvarez-Melis & Jaakkola, 2018) how much
                               does the explanation move when the input
                               barely moves?
                  complexity   attribution entropy - a dense explanation
                               is harder to read
counterfactuals   invalidity   did the model's class actually flip?
                  proximity    how far the patient must move
                  sparsity     how many features had to change

The comparability problem, stated openly
----------------------------------------
Infidelity (squared prediction error) and invalidity (a 0/1 event) are not
in the same units, so they cannot be averaged as they stand. Each raw
component is therefore mapped through its own empirical CDF, estimated on
the CALIBRATION split, giving a value in [0, 1] that means "this
explanation is worse than this fraction of calibration explanations by the
same method". Combining those normalised components is a modelling choice,
not a fact; experiments.py runs a sensitivity analysis over the weights.
"""

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------
# Attribution properties
# ---------------------------------------------------------------------

def infidelity(predict_proba, x: np.ndarray, attribution: np.ndarray,
               n_samples: int = config.INFIDELITY_SAMPLES,
               noise: float = config.INFIDELITY_NOISE,
               rng=None) -> float:
    """
    E[(I . attribution - (f(x) - f(x - I)))^2] over Gaussian perturbations I.

    Low infidelity means the attribution really does describe how the model
    responds around this patient.
    """
    rng = rng or np.random.default_rng(config.RANDOM_STATE)
    perturbations = rng.normal(0.0, noise, size=(n_samples, len(x))).astype(np.float32)
    baseline = float(predict_proba(x.reshape(1, -1))[0])
    perturbed = predict_proba((x - perturbations).astype(np.float32))
    explained = perturbations @ attribution
    actual = baseline - perturbed
    return float(np.mean((explained - actual) ** 2))


def instability(explain_batch, x: np.ndarray, attribution: np.ndarray,
                radius: float = config.STABILITY_RADIUS,
                n_neighbours: int = config.STABILITY_NEIGHBOURS,
                rng=None) -> float:
    """
    Local Lipschitz estimate: max ||a(x) - a(x')|| / ||x - x'|| over
    neighbours x' sampled inside an epsilon-ball around x.
    """
    rng = rng or np.random.default_rng(config.RANDOM_STATE)
    directions = rng.normal(size=(n_neighbours, len(x)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scales = rng.uniform(0.2, 1.0, size=(n_neighbours, 1)) * radius
    neighbours = (x + directions * scales).astype(np.float32)

    neighbour_attributions = explain_batch(neighbours)
    ratios = []
    for neighbour, neighbour_attribution in zip(neighbours, neighbour_attributions):
        distance = np.linalg.norm(neighbour - x)
        if distance > 1e-9:
            ratios.append(np.linalg.norm(neighbour_attribution - attribution) / distance)
    return float(max(ratios)) if ratios else np.nan


def complexity(attribution: np.ndarray) -> float:
    """
    Entropy of the normalised |attribution| distribution (Bhatt et al.,
    2020), scaled to [0, 1]. 0 = one feature explains everything.
    """
    magnitude = np.abs(attribution)
    total = magnitude.sum()
    if total <= 0:
        return 1.0
    p = magnitude / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(attribution)))


# ---------------------------------------------------------------------
# Counterfactual properties
# ---------------------------------------------------------------------

def counterfactual_metrics(explanation, predictor, x: np.ndarray,
                           feature_names, immutable_features) -> dict:
    """Validity, proximity, sparsity and whether immutables were respected."""
    if explanation.counterfactuals is None or len(explanation.counterfactuals) == 0:
        return {"validity": 0.0, "proximity": np.nan, "sparsity": np.nan,
                "immutables_respected": True, "n_counterfactuals": 0}

    cf = explanation.counterfactuals[list(feature_names)].to_numpy(dtype=np.float32)
    probabilities = predictor.predict_proba(cf)
    original = float(predictor.predict_proba(x.reshape(1, -1))[0])
    original_class = int(original >= config.DECISION_THRESHOLD)
    flipped = (probabilities >= config.DECISION_THRESHOLD).astype(int) != original_class

    differences = np.abs(cf - x.reshape(1, -1))
    changed = differences > 1e-6
    immutable_index = [list(feature_names).index(f) for f in immutable_features
                       if f in list(feature_names)]
    return {
        "validity": float(flipped.mean()),
        "proximity": float(differences.sum(axis=1).mean() / len(x)),
        "sparsity": float(changed.sum(axis=1).mean() / len(x)),
        "immutables_respected": bool(not changed[:, immutable_index].any())
                                 if immutable_index else True,
        "n_counterfactuals": int(len(cf)),
    }


# ---------------------------------------------------------------------
# Raw measurement table
# ---------------------------------------------------------------------

def measure_attributions(explanations, explain_batch, predictor, X,
                         seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Infidelity / instability / complexity for one method, all instances."""
    rng = np.random.default_rng(seed)
    rows = []
    for row, explanation in enumerate(explanations):
        x = X[row]
        rows.append({
            "instance_id": explanation.instance_id,
            "method": explanation.method,
            "prediction": explanation.prediction,
            "infidelity": infidelity(predictor.predict_proba, x,
                                     explanation.attribution, rng=rng),
            "instability": instability(explain_batch, x, explanation.attribution,
                                       rng=rng),
            "complexity": complexity(explanation.attribution),
        })
    return pd.DataFrame(rows)


def measure_counterfactuals(explanations, predictor, X, dataset) -> pd.DataFrame:
    rows = []
    for row, explanation in enumerate(explanations):
        metrics = counterfactual_metrics(explanation, predictor, X[row],
                                          dataset.feature_names,
                                          dataset.immutable_features)
        rows.append({"instance_id": explanation.instance_id,
                     "method": explanation.method,
                     "prediction": explanation.prediction, **metrics})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# From raw measurements to a single risk in [0, 1]
# ---------------------------------------------------------------------

class RiskNormaliser:
    """
    Maps raw quality components onto [0, 1] using the calibration split's
    empirical CDF, per method and per component. This is what puts
    attribution and counterfactual quality on one axis.
    """

    def __init__(self):
        self.references = {}          # (method, component) -> sorted values

    def fit(self, table: pd.DataFrame, components: list) -> "RiskNormaliser":
        for method, group in table.groupby("method"):
            for component in components:
                if component not in group:
                    continue
                values = group[component].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if len(values):
                    self.references[(method, component)] = np.sort(values)
        return self

    def transform_component(self, method: str, component: str, values) -> np.ndarray:
        reference = self.references.get((method, component))
        values = np.asarray(values, dtype=float)
        if reference is None or len(reference) == 0:
            return np.full_like(values, 0.5, dtype=float)
        ranks = np.searchsorted(reference, values, side="right") / len(reference)
        return np.clip(np.nan_to_num(ranks, nan=1.0), 0.0, 1.0)

    def risk(self, table: pd.DataFrame) -> pd.Series:
        """Weighted sum of normalised components; higher = worse explanation."""
        risk = pd.Series(0.0, index=table.index)
        for method, group in table.groupby("method"):
            if method in config.COUNTERFACTUAL_METHODS:
                parts = {
                    "invalidity": (1.0 - group["validity"].to_numpy(dtype=float),
                                   config.RISK_WEIGHTS_COUNTERFACTUAL["invalidity"]),
                    "proximity": (self.transform_component(method, "proximity",
                                                           group["proximity"]),
                                  config.RISK_WEIGHTS_COUNTERFACTUAL["proximity"]),
                    "sparsity": (self.transform_component(method, "sparsity",
                                                          group["sparsity"]),
                                 config.RISK_WEIGHTS_COUNTERFACTUAL["sparsity"]),
                }
            else:
                parts = {
                    "infidelity": (self.transform_component(method, "infidelity",
                                                            group["infidelity"]),
                                   config.RISK_WEIGHTS_ATTRIBUTION["infidelity"]),
                    "instability": (self.transform_component(method, "instability",
                                                             group["instability"]),
                                    config.RISK_WEIGHTS_ATTRIBUTION["instability"]),
                }
            combined = sum(values * weight for values, weight in parts.values())
            total_weight = sum(weight for _, weight in parts.values())
            risk.loc[group.index] = np.clip(combined / total_weight, 0.0, 1.0)
        return risk


RISK_COMPONENTS = ["infidelity", "instability", "complexity",
                   "proximity", "sparsity"]


def attach_risk(measurement_table: pd.DataFrame,
                normaliser: RiskNormaliser) -> pd.DataFrame:
    table = measurement_table.copy()
    table["risk"] = normaliser.risk(table)
    return table


def risk_matrix(table: pd.DataFrame, methods=None) -> pd.DataFrame:
    """Wide form: one row per instance, one risk column per method."""
    methods = methods or config.ALL_METHODS
    wide = table.pivot_table(index="instance_id", columns="method", values="risk")
    return wide.reindex(columns=[m for m in methods if m in wide.columns])


def summarise_by_method(table: pd.DataFrame) -> pd.DataFrame:
    """Mean of every measured property per method - the paper's XAI table."""
    columns = [c for c in RISK_COMPONENTS + ["validity", "risk"] if c in table]
    return table.groupby("method")[columns].agg(["mean", "std", "count"])


# ============================================================
# CHECKLIST
# - Attribution quality: infidelity (perturbation-based faithfulness),
#   instability (local Lipschitz over an epsilon-ball) and complexity
#   (attribution entropy)
# - Counterfactual quality: validity, proximity, sparsity and an explicit
#   check that immutable features were not altered
# - measure_*() build the raw per-instance measurement tables
# - RiskNormaliser maps each raw component through its CALIBRATION-split
#   ECDF, which is what makes attribution and counterfactual quality
#   comparable on one [0, 1] axis
# - The weighting of normalised components is a declared modelling choice,
#   ablated in experiments.py - not presented as a natural fact
# - risk_matrix() gives the instance x method risk table the selective
#   policies in selective.py consume
# ============================================================
