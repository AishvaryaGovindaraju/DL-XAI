"""
mc_dropout.py
=============
Phase 4d - Monte Carlo Dropout: the epistemic uncertainty estimate the whole
framework is conditioned on.

Idea: keep dropout active at inference time and run the same test set T times.
Each pass samples a different sub-network, so each instance gets T slightly
different probabilities. The MEAN of those is the prediction; the VARIANCE is
how much the model disagrees with itself about that instance.

One decision worth reading before you change it
-----------------------------------------------
A plain `model.train()` would activate dropout, but it would ALSO put the
BatchNorm layers into batch-statistics mode, so an instance's prediction
would start depending on which other patients share its batch. That is batch
noise, not epistemic uncertainty. So the default here is:

    model.eval()                      -> BatchNorm uses fixed running stats
    every nn.Dropout module .train()  -> dropout stays stochastic

This is what produced the stratification in the notebooks
(`mc_variance_final`). `batchnorm_in_train_mode=True` reproduces the earlier
whole-model .train() behaviour if you need it for an ablation.
"""

import numpy as np
import torch
import torch.nn as nn

import config


def enable_mc_dropout(model, batchnorm_in_train_mode: bool = False) -> None:
    """Put the model in the inference state used for MC sampling."""
    if batchnorm_in_train_mode:
        model.train()
        return
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def dropout_state(model) -> dict:
    """Which layers are stochastic right now - call this to verify setup."""
    return {
        name: module.training
        for name, module in model.named_modules()
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d))
    }


def mc_predict(model, loader, n_passes: int = config.MC_PASSES,
               batchnorm_in_train_mode: bool = False,
               verbose: bool = False) -> np.ndarray:
    """
    Run T stochastic forward passes over `loader`.

    Returns an array of shape (n_passes, n_instances); column j holds the T
    sampled probabilities for test instance j.
    """
    enable_mc_dropout(model, batchnorm_in_train_mode)
    passes = []
    with torch.no_grad():
        for t in range(n_passes):
            batch_predictions = [model(X_batch).squeeze(1).cpu()
                                 for X_batch, _ in loader]
            passes.append(torch.cat(batch_predictions))
            if verbose and (t + 1) % 10 == 0:
                print(f"MC pass {t + 1}/{n_passes}")
    return torch.stack(passes).numpy()


def summarise(mc_predictions: np.ndarray) -> dict:
    """Per-instance predictive mean, variance and standard deviation."""
    mean = mc_predictions.mean(axis=0)
    variance = mc_predictions.var(axis=0)
    return {"mean": mean, "variance": variance, "std": np.sqrt(variance)}


def variance_percentiles(variance: np.ndarray,
                         percentiles=(50, 90, 95, 99, 99.9)) -> dict:
    """Where the uncertainty actually sits - drives the stratum design."""
    summary = {f"p{p}": float(np.percentile(variance, p)) for p in percentiles}
    summary["min"] = float(variance.min())
    summary["max"] = float(variance.max())
    return summary


def check_mc_output(mc_predictions: np.ndarray, n_instances: int,
                    n_passes: int = config.MC_PASSES) -> None:
    """Shape and validity checks for the sampled prediction matrix."""
    assert mc_predictions.shape == (n_passes, n_instances), (
        f"expected ({n_passes}, {n_instances}), got {mc_predictions.shape}")
    assert not np.isnan(mc_predictions).any(), "MC predictions contain NaN"
    assert mc_predictions.var(axis=0).max() > 0, (
        "zero variance everywhere - dropout was not active during sampling")


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] enable_mc_dropout() puts the model in eval() and then switches ONLY
#     the nn.Dropout modules back to train(), so dropout is stochastic
#     while BatchNorm keeps its fixed running statistics.
# [x] Documents why: whole-model .train() would make each prediction
#     depend on its batch neighbours, contaminating the uncertainty
#     estimate with batch noise. The old behaviour is still reachable via
#     batchnorm_in_train_mode=True for ablation.
# [x] dropout_state() reports which dropout/BatchNorm layers are live, so
#     the configuration can be verified rather than assumed.
# [x] mc_predict() runs T = 50 stochastic passes and returns a
#     (T, n_instances) matrix of sampled probabilities.
# [x] summarise() reduces that matrix to per-instance predictive mean,
#     variance (the uncertainty score) and standard deviation.
# [x] variance_percentiles() reports the observed spread of the variance,
#     which is the evidence for how the strata are cut.
# [x] check_mc_output() asserts the shape, absence of NaN, and non-zero
#     variance - a zero-variance result means dropout was never active.
# =====================================================================
