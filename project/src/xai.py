"""
xai.py
======
Explanation generation. Four methods, one interface.

    attribution methods   shap, lime, ig    -> vector of per-feature weights
    counterfactual method dice              -> alternative feature vectors

Everything returns an `Explanation`, so xai_metrics.py can score any method
without knowing which one it is. Each explainer is also exposed as a plain
callable `explain_batch(X) -> attributions`, because measuring stability
means re-explaining perturbed copies of an instance many times.

SHAP, LIME and DiCE are imported lazily: a missing package disables that
one method with a clear message instead of breaking the pipeline.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

import config


@dataclass
class Explanation:
    """One explanation of one instance by one method."""

    instance_id: int
    method: str
    attribution: np.ndarray = None          # attribution methods
    counterfactuals: pd.DataFrame = None    # counterfactual methods
    prediction: float = None
    extra: dict = field(default_factory=dict)

    @property
    def is_attribution(self) -> bool:
        return self.attribution is not None

    @property
    def failed(self) -> bool:
        return self.attribution is None and self.counterfactuals is None


# ---------------------------------------------------------------------
# Integrated Gradients - native to the deep model, cheap, differentiable
# ---------------------------------------------------------------------

def integrated_gradients(module, X: np.ndarray, baseline: np.ndarray = None,
                         steps: int = config.IG_STEPS) -> np.ndarray:
    """
    Attributions along the straight path from a baseline to each instance
    (Sundararajan et al., 2017). The baseline is the training mean, i.e.
    the average patient in the scaled space.
    """
    module.eval()
    X_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32))
    if baseline is None:
        baseline = np.zeros(X_tensor.shape[1], dtype=np.float32)
    base = torch.as_tensor(np.asarray(baseline, dtype=np.float32)).unsqueeze(0)

    total = torch.zeros_like(X_tensor)
    for step in range(1, steps + 1):
        alpha = step / steps
        point = (base + alpha * (X_tensor - base)).clone().requires_grad_(True)
        output = module(point).sum()
        grad, = torch.autograd.grad(output, point)
        total += grad
    return ((X_tensor - base) * total / steps).detach().numpy()


# ---------------------------------------------------------------------
# Explainer factories: each returns explain_batch(X) -> (n, n_features)
# ---------------------------------------------------------------------

def make_ig_explainer(predictor, dataset):
    baseline = dataset.array("train").mean(axis=0)

    def explain_batch(X):
        return integrated_gradients(predictor.module, X, baseline=baseline)
    return explain_batch


def make_shap_explainer(predictor, dataset, background_size=config.SHAP_BACKGROUND_SIZE):
    """
    Gradient-based SHAP for the deep models (fast, uses the network's
    gradients), permutation SHAP as the model-agnostic fallback.
    """
    import shap

    rng = np.random.default_rng(config.RANDOM_STATE)
    train = dataset.array("train")
    background = train[rng.choice(len(train), size=min(background_size, len(train)),
                                  replace=False)]

    if getattr(predictor, "is_deep", False):
        # GradientExplainer indexes the output as (n, n_outputs); the
        # networks return (n,), so re-add the trailing axis.
        class _TwoDimensional(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x).unsqueeze(-1)

        explainer = shap.GradientExplainer(_TwoDimensional(predictor.module),
                                           torch.as_tensor(background))

        def explain_batch(X):
            values = explainer.shap_values(
                torch.as_tensor(np.asarray(X, dtype=np.float32)))
            values = values[0] if isinstance(values, list) else values
            values = np.asarray(values)
            return values[..., 0] if values.ndim == 3 else values
    else:
        explainer = shap.Explainer(predictor.predict_proba, background)

        def explain_batch(X):
            return np.asarray(explainer(np.asarray(X, dtype=np.float32)).values)
    return explain_batch


def make_lime_explainer(predictor, dataset,
                        num_samples: int = config.LIME_NUM_SAMPLES):
    """
    LIME with discretize_continuous=False, so it perturbs in the same
    standardised space the network was trained on and its weights stay
    comparable with SHAP and IG.
    """
    from lime.lime_tabular import LimeTabularExplainer

    def predict_proba_2col(X):
        p = predictor.predict_proba(X)
        return np.column_stack([1.0 - p, p])

    explainer = LimeTabularExplainer(
        training_data=dataset.array("train"),
        feature_names=dataset.feature_names,
        class_names=["negative", "positive"],
        mode="classification",
        discretize_continuous=False,
        random_state=config.RANDOM_STATE,
    )
    name_to_index = {name: i for i, name in enumerate(dataset.feature_names)}

    def explain_batch(X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros_like(X)
        for row, instance in enumerate(X):
            explanation = explainer.explain_instance(
                instance, predict_proba_2col,
                num_features=X.shape[1], num_samples=num_samples)
            for name, weight in explanation.as_list():
                if name in name_to_index:
                    out[row, name_to_index[name]] = weight
        return out
    return explain_batch


ATTRIBUTION_FACTORIES = {"ig": make_ig_explainer,
                         "shap": make_shap_explainer,
                         "lime": make_lime_explainer}


def build_attribution_explainers(predictor, dataset, methods=None) -> dict:
    """Build every requested attribution explainer, skipping unavailable ones."""
    methods = methods or config.ATTRIBUTION_METHODS
    explainers = {}
    for method in methods:
        try:
            explainers[method] = ATTRIBUTION_FACTORIES[method](predictor, dataset)
        except ImportError as error:
            print(f"  [skip] {method}: package unavailable ({error.name})")
    return explainers


# ---------------------------------------------------------------------
# DiCE counterfactuals
# ---------------------------------------------------------------------

def make_dice_explainer(predictor, dataset):
    """
    DiCE on the scaled feature space, with the dataset's immutable
    features locked so recommendations stay actionable.
    """
    import dice_ml
    from dice_ml.model_interfaces.base_model import BaseModel

    frame = dataset.X["train"].copy()
    frame["_target"] = dataset.y["train"].astype(int)

    class Wrapped(BaseModel):
        def __init__(self):
            super().__init__(model=None, backend="sklearn")

        def get_output(self, input_instance, transform_data=False, model_score=True):
            values = (input_instance.to_numpy(dtype=np.float32)
                      if hasattr(input_instance, "to_numpy")
                      else np.asarray(input_instance, dtype=np.float32))
            p = predictor.predict_proba(values).reshape(-1, 1)
            return np.hstack([1.0 - p, p])

    data_interface = dice_ml.Data(
        dataframe=frame,
        continuous_features=list(dataset.feature_names),
        outcome_name="_target")
    explainer = dice_ml.Dice(data_interface, Wrapped(), method=config.DICE_METHOD)
    mutable = [f for f in dataset.feature_names if f not in dataset.immutable_features]

    def explain_one(instance_frame: pd.DataFrame):
        result = explainer.generate_counterfactuals(
            instance_frame, total_CFs=config.DICE_TOTAL_CFS,
            desired_class="opposite", features_to_vary=mutable)
        cf = result.cf_examples_list[0].final_cfs_df
        if cf is None or len(cf) == 0:
            return None
        return cf.drop(columns=["_target"], errors="ignore")
    return explain_one


# ---------------------------------------------------------------------
# Producing the explanation set for a sample of instances
# ---------------------------------------------------------------------

def select_instances(dataset, n: int = config.N_EXPLAIN,
                     seed: int = config.RANDOM_STATE, split: str = "test") -> np.ndarray:
    """
    Random sample of row positions from `split`.

    Random, not uncertainty-stratified: the selective policy has to be
    evaluated on the population it would actually face.
    """
    rng = np.random.default_rng(seed)
    size = len(dataset.y[split])
    return np.sort(rng.choice(size, size=min(n, size), replace=False))


def explain_instances(predictor, dataset, instance_ids, methods=None,
                      include_dice: bool = True, split: str = "test",
                      verbose: bool = True) -> dict:
    """
    Explain the selected instances with every available method.

    `split` is "calib" when building the data the quality head and the
    conformal threshold are fitted on, and "test" when evaluating.
    Returns {method: [Explanation, ...]} in instance_id order.
    """
    X = dataset.array(split)[instance_ids]
    predictions = predictor.predict_proba(X)
    explainers = build_attribution_explainers(predictor, dataset, methods)
    results = {}

    for method, explain_batch in explainers.items():
        attributions = explain_batch(X)
        results[method] = [
            Explanation(instance_id=int(i), method=method,
                        attribution=attributions[row], prediction=float(predictions[row]))
            for row, i in enumerate(instance_ids)]
        if verbose:
            print(f"  {method}: {len(results[method])} explanations")

    if include_dice and "dice" in config.COUNTERFACTUAL_METHODS:
        try:
            explain_one = make_dice_explainer(predictor, dataset)
        except ImportError as error:
            print(f"  [skip] dice: package unavailable ({error.name})")
            return results
        records, failures = [], 0
        frames = dataset.X[split].iloc[instance_ids]
        for row, i in enumerate(instance_ids):
            single = frames.iloc[[row]]
            try:
                cf = explain_one(single)
            except Exception as error:                 # failures are data
                cf, note = None, str(error)[:120]
            else:
                note = None if cf is not None else "no counterfactual found"
            failures += cf is None
            records.append(Explanation(instance_id=int(i), method="dice",
                                       counterfactuals=cf,
                                       prediction=float(predictions[row]),
                                       extra={"error": note}))
        results["dice"] = records
        if verbose:
            print(f"  dice: {len(records) - failures} found, {failures} failed")
    return results


# ============================================================
# CHECKLIST
# - One Explanation object covers both attributions and counterfactuals
# - Integrated Gradients implemented directly on the network (baseline =
#   training mean, 32 steps)
# - SHAP uses GradientExplainer for deep models, permutation SHAP as the
#   model-agnostic fallback
# - LIME runs with discretize_continuous=False so its weights are
#   comparable with SHAP and IG in the same standardised space
# - DiCE generates 3 opposite-class counterfactuals with the dataset's
#   immutable features locked; failures are recorded, not dropped
# - Every explainer is also a plain explain_batch(X) callable, which is
#   what makes the stability measurements in xai_metrics.py possible
# - shap / lime / dice_ml are imported lazily; a missing package disables
#   only that method
# ============================================================
