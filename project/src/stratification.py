"""
stratification.py
=================
Phase 5 - split the test set into LOW / MEDIUM / HIGH epistemic-uncertainty
strata and draw the 600 instances that all three XAI methods will explain.

Why percentile strata and not the PRD's fixed cut-points
--------------------------------------------------------
The PRD specifies sigma^2 < 0.05 / 0.05-0.15 / >= 0.15. The trained model's
predictive variance never gets anywhere near those values, so those rules put
every instance in LOW and leave MEDIUM and HIGH empty - unusable for a
between-stratum comparison. The pipeline therefore ranks instances by
variance and cuts equal thirds. `threshold_strata()` is kept so the paper can
report the fixed-threshold counts as the justification for that choice.
"""

import numpy as np
import pandas as pd

import config


def threshold_strata(variance: np.ndarray, thresholds=config.PRD_THRESHOLDS) -> np.ndarray:
    """Fixed-cut-point labelling (the PRD rule). Kept for the diagnostic."""
    low, high = thresholds
    return np.where(variance < low, "LOW",
                    np.where(variance < high, "MEDIUM", "HIGH"))


def equal_thirds_strata(variance: np.ndarray) -> np.ndarray:
    """
    Rank by variance and label the lowest third LOW, the middle MEDIUM and
    the highest HIGH. Strata are equal-sized by construction, so stratum
    size cannot confound the between-stratum comparisons.
    """
    n = len(variance)
    order = np.argsort(variance)
    size = n // 3

    labels = np.empty(n, dtype=object)
    labels[order[:size]] = "LOW"
    labels[order[size:2 * size]] = "MEDIUM"
    labels[order[2 * size:]] = "HIGH"
    return labels


def build_stratification_table(variance: np.ndarray,
                               mean: np.ndarray = None) -> pd.DataFrame:
    """One row per test instance: id, uncertainty, stratum."""
    table = pd.DataFrame({
        "test_instance_id": np.arange(len(variance)),
        "sigma_squared": variance,
        "uncertainty_stratum": equal_thirds_strata(variance),
    })
    if mean is not None:
        table["mc_mean_probability"] = mean
    return table


def stratum_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Size and variance range of each stratum - a table for the paper."""
    grouped = table.groupby("uncertainty_stratum")["sigma_squared"]
    summary = pd.DataFrame({
        "n": grouped.size(),
        "min_sigma_squared": grouped.min(),
        "max_sigma_squared": grouped.max(),
        "mean_sigma_squared": grouped.mean(),
    }).reindex(list(config.STRATUM_NAMES))
    summary["meets_minimum"] = summary["n"] >= config.MIN_STRATUM_SIZE
    return summary


def compare_threshold_rules(variance: np.ndarray) -> pd.DataFrame:
    """
    Stratum counts under the PRD thresholds, the adjusted thresholds and the
    percentile rule. This is the evidence for using percentiles.
    """
    rules = {
        f"PRD {config.PRD_THRESHOLDS}": threshold_strata(variance, config.PRD_THRESHOLDS),
        f"adjusted {config.PRD_ADJUSTED_THRESHOLDS}": threshold_strata(
            variance, config.PRD_ADJUSTED_THRESHOLDS),
        "equal thirds (used)": equal_thirds_strata(variance),
    }
    return pd.DataFrame({
        name: pd.Series(labels).value_counts().reindex(
            list(config.STRATUM_NAMES), fill_value=0)
        for name, labels in rules.items()
    })


def sample_instances(table: pd.DataFrame,
                     n_per_stratum: int = config.SAMPLES_PER_STRATUM,
                     seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Draw an equal number of instances from each stratum (3 x 200 = 600)."""
    for stratum in config.STRATUM_NAMES:
        available = (table["uncertainty_stratum"] == stratum).sum()
        if available < n_per_stratum:
            raise ValueError(
                f"stratum {stratum} has {available} instances, "
                f"cannot sample {n_per_stratum}")

    sampled = pd.concat([
        table[table["uncertainty_stratum"] == stratum].sample(
            n=n_per_stratum, random_state=seed)
        for stratum in config.STRATUM_NAMES
    ])
    return (sampled
            .sort_values(["uncertainty_stratum", "test_instance_id"])
            .reset_index(drop=True))


def save_outputs(table: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Persist the full stratification and the 600-instance XAI sample."""
    config.ensure_dirs()
    table.to_csv(config.STRATIFICATION_PATH, index=False)
    sample.to_csv(config.STRATIFIED_SAMPLES_PATH, index=False)


def load_sample() -> pd.DataFrame:
    """Load the 600-instance sample used by all three XAI phases."""
    return pd.read_csv(config.resolve_input(config.STRATIFIED_SAMPLES_PATH))


def run_stratification(variance: np.ndarray, mean: np.ndarray = None,
                       verbose: bool = True) -> tuple:
    """Full Phase 5: label, summarise, sample, save. Returns (table, sample)."""
    table = build_stratification_table(variance, mean)
    sample = sample_instances(table)
    save_outputs(table, sample)

    if verbose:
        print("Stratum counts under each rule:")
        print(compare_threshold_rules(variance))
        print("\nStrata actually used:")
        print(stratum_summary(table))
        print(f"\nSampled {len(sample)} instances for explanation.")
    return table, sample


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Labels every test instance LOW / MEDIUM / HIGH by MC Dropout
#     predictive variance.
# [x] Uses equal-sized percentile thirds, so the three strata cannot
#     differ in size - stratum size can therefore not explain any
#     difference in explanation quality later.
# [x] Keeps the PRD's fixed thresholds (0.05/0.15 and 0.03/0.10) as a
#     reported comparison, since those rules collapse the test set into a
#     single stratum on this model - that is the reason for percentiles.
# [x] stratum_summary() reports per-stratum size and variance range and
#     flags the PRD's 150-instance minimum.
# [x] Draws 200 instances per stratum with the fixed seed (600 total) -
#     the exact same instances are explained by SHAP, LIME and DiCE.
# [x] Saves the full stratification table and the 600-row sample to
#     project/data/processed/.
# [x] load_sample() is the single entry point the three XAI files use.
# =====================================================================
