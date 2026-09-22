"""
statistical_tests.py
====================
The statistics the hypotheses are decided with.

Named statistical_tests.py rather than statistics.py on purpose: a module
called statistics.py on sys.path shadows the Python standard library module
of that name, and several third-party packages import it.

What is here
------------
- bootstrap confidence intervals (no normality assumed anywhere)
- paired Wilcoxon for policy-vs-policy comparisons on the same instances
- Friedman + Nemenyi when several policies are compared across datasets
- Spearman correlation with a CI, for "does uncertainty predict explanation
  risk?" (H1 and the replication check)
- Holm correction, applied to every family of tests reported together
- effect sizes, because a p-value alone answers nothing at n = 15,000
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats

import config


# ---------------------------------------------------------------------
# Intervals and effect sizes
# ---------------------------------------------------------------------

def bootstrap_ci(values, statistic=np.mean, n_boot: int = config.N_BOOTSTRAP,
                 level: float = config.CI_LEVEL,
                 seed: int = config.RANDOM_STATE) -> dict:
    """Percentile bootstrap interval for any statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    samples = statistic(draws, axis=1)
    lower = (1 - level) / 2 * 100
    return {"estimate": float(statistic(values)),
            "lo": float(np.percentile(samples, lower)),
            "hi": float(np.percentile(samples, 100 - lower)),
            "n": int(len(values))}


def cliffs_delta(a, b) -> float:
    """
    Non-parametric effect size in [-1, 1]: P(a > b) - P(a < b).
    Reported alongside Wilcoxon, which only gives a p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    comparisons = np.sign(a[:, None] - b[None, :])
    return float(comparisons.mean())


def rank_biserial(x, y) -> float:
    """Matched-pairs effect size for the paired Wilcoxon test."""
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    return float(2 * positive / ranks.sum() - 1)


# ---------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------

def holm_correction(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, monotone, same order as input."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def paired_comparison(x, y, label_x: str, label_y: str) -> dict:
    """Paired Wilcoxon signed-rank plus effect size and median difference."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    x, y = x[both], y[both]
    if len(x) < 5 or np.allclose(x, y):
        return {"comparison": f"{label_x} vs {label_y}", "n": int(len(x)),
                "statistic": np.nan, "p_value": 1.0,
                "median_difference": float(np.median(x - y)) if len(x) else np.nan,
                "rank_biserial": 0.0}
    statistic, p_value = stats.wilcoxon(x, y)
    return {"comparison": f"{label_x} vs {label_y}", "n": int(len(x)),
            "statistic": float(statistic), "p_value": float(p_value),
            "median_difference": float(np.median(x - y)),
            "rank_biserial": rank_biserial(x, y)}


def compare_all_policies(risk_by_policy: dict) -> pd.DataFrame:
    """
    Every policy pair, paired Wilcoxon on the same instances, Holm-corrected
    across the whole family.
    """
    rows = [paired_comparison(risk_by_policy[a], risk_by_policy[b], a, b)
            for a, b in itertools.combinations(risk_by_policy, 2)]
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm_correction(table["p_value"])
        table["significant"] = table["p_holm"] < config.ALPHA_SIGNIFICANCE
    return table


def friedman_nemenyi(matrix: pd.DataFrame) -> dict:
    """
    Friedman test across blocks (rows = datasets or seeds, columns =
    policies) plus the Nemenyi critical difference for the post-hoc.
    """
    values = matrix.dropna().to_numpy(dtype=float)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan,
                "mean_ranks": {}, "critical_difference": np.nan,
                "note": "too few blocks or groups for Friedman"}
    statistic, p_value = stats.friedmanchisquare(*values.T)
    ranks = np.apply_along_axis(stats.rankdata, 1, values).mean(axis=0)
    n_blocks, k = values.shape
    q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.164)
    critical = q_alpha * np.sqrt(k * (k + 1) / (6 * n_blocks))
    return {"statistic": float(statistic), "p_value": float(p_value),
            "mean_ranks": dict(zip(matrix.columns, ranks.round(3))),
            "critical_difference": float(critical),
            "note": "Nemenyi CD at alpha=0.05"}


def spearman_with_ci(x, y, n_boot: int = config.N_BOOTSTRAP,
                     seed: int = config.RANDOM_STATE) -> dict:
    """
    Spearman rho with a bootstrap CI. Used for "does epistemic uncertainty
    rank explanation risk?" and for the learned predictor's version of the
    same question.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    both = np.isfinite(x) & np.isfinite(y)
    x, y = x[both], y[both]
    if len(x) < 10:
        return {"rho": np.nan, "p_value": np.nan, "lo": np.nan, "hi": np.nan,
                "n": int(len(x))}
    rho, p_value = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(min(n_boot, 1000)):
        index = rng.integers(0, len(x), len(x))
        if len(np.unique(y[index])) > 2:
            samples.append(stats.spearmanr(x[index], y[index]).statistic)
    return {"rho": float(rho), "p_value": float(p_value),
            "lo": float(np.percentile(samples, 2.5)) if samples else np.nan,
            "hi": float(np.percentile(samples, 97.5)) if samples else np.nan,
            "n": int(len(x))}


def summarise_across_seeds(table: pd.DataFrame, group_columns: list,
                           value_column: str) -> pd.DataFrame:
    """Mean with bootstrap CI over seeds, per group - how results are reported."""
    rows = []
    for keys, group in table.groupby(group_columns):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(bootstrap_ci(group[value_column]))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# CHECKLIST
# - Deliberately NOT named statistics.py, which would shadow the standard
#   library module of that name
# - bootstrap_ci(): percentile intervals for every reported number
# - paired_comparison() / compare_all_policies(): paired Wilcoxon over all
#   policy pairs with Holm correction across the family
# - friedman_nemenyi(): omnibus test plus critical difference when policies
#   are compared across datasets or seeds
# - spearman_with_ci(): the correlation tests behind H1 and the replication
#   of the published uncertainty-vs-explanation-quality result
# - Effect sizes (Cliff's delta, rank-biserial) reported next to every
#   p-value
# - summarise_across_seeds(): the mean +/- CI form all tables use
# ============================================================
