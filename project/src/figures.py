"""
figures.py
==========
The paper's figures. One function per figure, each takes a results table
(or a run context) and writes a PNG + PDF into project/figures/.

Figure 1  coverage-risk curve            the headline: how many patients
                                         still get an explanation at each
                                         requested risk level
Figure 2  policy comparison              CSE vs fixed methods vs
                                         uncertainty gating vs oracle
Figure 3  uncertainty vs explanation risk the replication check
Figure 4  reliability diagrams            is the uncertainty calibrated?
Figure 5  abstention profile              where the system declines to
                                          explain
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

PALETTE = {"cse": "#1b4965", "oracle": "#5fa8d3", "uncertainty_gating": "#c1121f",
           "random": "#8d99ae", "fixed_shap": "#2a9d8f", "fixed_lime": "#e9c46a",
           "fixed_ig": "#f4a261", "fixed_dice": "#9b5de5"}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })


def _save(fig, name: str) -> list:
    config.ensure_dirs()
    paths = []
    for extension in ("png", "pdf"):
        path = config.FIGURES_DIR / f"{name}.{extension}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------

def figure_coverage_risk(curve: pd.DataFrame, name: str = "fig1_coverage_risk") -> list:
    """Coverage and realised risk against the requested level alpha."""
    _style()
    aggregated = curve.groupby(["dataset", "alpha"], as_index=False).agg(
        coverage=("coverage", "mean"),
        population_risk=("population_risk", "mean"),
        selective_risk=("selective_risk", "mean"))

    datasets = sorted(aggregated["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for key in datasets:
        part = aggregated[aggregated["dataset"] == key].sort_values("alpha")
        axes[0].plot(part["alpha"], part["coverage"], marker="o", label=key)
        axes[1].plot(part["alpha"], part["population_risk"], marker="o", label=key)

    axes[0].set_xlabel(r"requested risk level $\alpha$")
    axes[0].set_ylabel("coverage (fraction explained)")
    axes[0].set_title("How many patients receive an explanation")
    axes[0].set_ylim(-0.02, 1.02)

    limits = [aggregated["alpha"].min(), aggregated["alpha"].max()]
    axes[1].plot(limits, limits, ls="--", c="0.4", lw=1, label=r"$y=\alpha$ (bound)")
    axes[1].set_xlabel(r"requested risk level $\alpha$")
    axes[1].set_ylabel("realised population risk")
    axes[1].set_title("Is the guarantee respected?")
    axes[1].legend(fontsize=8)
    return _save(fig, name)


def figure_policy_comparison(policies: pd.DataFrame,
                             alpha: float = config.DEFAULT_ALPHA,
                             name: str = "fig2_policy_comparison") -> list:
    """Population risk per policy at one risk level, per dataset."""
    _style()
    subset = policies[np.isclose(policies["alpha"], alpha)]
    pivot = subset.pivot_table(index="policy", columns="dataset",
                               values="population_risk", aggfunc="mean")
    order = [p for p in ["oracle", "cse", "uncertainty_gating", "random",
                         "fixed_shap", "fixed_lime", "fixed_ig", "fixed_dice"]
             if p in pivot.index]
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    width = 0.8 / max(len(pivot.columns), 1)
    positions = np.arange(len(pivot.index))
    for offset, dataset_key in enumerate(pivot.columns):
        ax.bar(positions + offset * width, pivot[dataset_key], width,
               label=dataset_key)
    ax.set_xticks(positions + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.axhline(alpha, ls="--", c="0.4", lw=1)
    ax.annotate(rf"$\alpha={alpha}$", (len(positions) - 0.5, alpha),
                fontsize=8, va="bottom", ha="right", color="0.3")
    ax.set_ylabel("population risk of delivered explanations")
    ax.set_title(f"Policies at matched coverage (lower is better), "
                 rf"$\alpha={alpha}$")
    ax.legend(fontsize=8)
    return _save(fig, name)


def figure_uncertainty_vs_risk(correlations: pd.DataFrame,
                               name: str = "fig3_uncertainty_vs_risk") -> list:
    """Spearman rho between epistemic uncertainty and explanation risk."""
    _style()
    aggregated = correlations.groupby(["dataset", "method"], as_index=False).agg(
        rho=("rho", "mean"), lo=("lo", "mean"), hi=("hi", "mean"))

    methods = sorted(aggregated["method"].unique())
    datasets = sorted(aggregated["dataset"].unique())
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    width = 0.8 / max(len(datasets), 1)
    positions = np.arange(len(methods))
    for offset, dataset_key in enumerate(datasets):
        part = aggregated[aggregated["dataset"] == dataset_key].set_index("method")
        part = part.reindex(methods)
        centres = positions + offset * width
        ax.bar(centres, part["rho"], width, label=dataset_key)
        ax.errorbar(centres, part["rho"],
                    yerr=[part["rho"] - part["lo"], part["hi"] - part["rho"]],
                    fmt="none", ecolor="0.3", capsize=2, lw=0.8)
    ax.axhline(0, c="0.3", lw=0.8)
    ax.set_xticks(positions + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("Epistemic uncertainty vs measured explanation risk")
    ax.legend(fontsize=8)
    return _save(fig, name)


def figure_reliability(y_true_by_model: dict, name: str = "fig4_reliability") -> list:
    """Reliability diagrams for the models' predicted probabilities."""
    import uq_methods
    _style()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", c="0.4", lw=1, label="perfect")
    for label, (y_true, probabilities) in y_true_by_model.items():
        curve = uq_methods.reliability_curve(y_true, probabilities)
        keep = curve["count"] > 0
        ax.plot(curve["bin_centre"][keep], curve["observed"][keep],
                marker="o", ms=4, label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)
    return _save(fig, name)


def figure_abstention(abstention: pd.DataFrame,
                      name: str = "fig5_abstention") -> list:
    """Mean epistemic uncertainty of delivered vs abstained instances."""
    _style()
    aggregated = abstention.groupby("dataset", as_index=False).agg(
        delivered=("epistemic_delivered", "mean"),
        abstained=("epistemic_abstained", "mean"),
        coverage=("coverage", "mean"))

    positions = np.arange(len(aggregated))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(positions - 0.2, aggregated["delivered"], 0.4, label="explained")
    ax.bar(positions + 0.2, aggregated["abstained"], 0.4, label="abstained")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{row.dataset}\n(coverage {row.coverage:.2f})"
                        for row in aggregated.itertuples()])
    ax.set_ylabel("mean epistemic uncertainty (nats)")
    ax.set_title("Where the system declines to explain")
    ax.legend(fontsize=8)
    return _save(fig, name)


def make_all(tables: dict) -> list:
    """Every figure that the available tables support."""
    written = []
    if "coverage_risk_curve" in tables:
        written += figure_coverage_risk(tables["coverage_risk_curve"])
    if "policies" in tables:
        written += figure_policy_comparison(tables["policies"])
    if "e2_uncertainty_vs_risk" in tables:
        written += figure_uncertainty_vs_risk(tables["e2_uncertainty_vs_risk"])
    if "h4_abstention" in tables:
        written += figure_abstention(tables["h4_abstention"])
    return written


# ============================================================
# CHECKLIST
# - Figure 1: coverage and realised risk against the requested alpha, with
#   the y = alpha reference line that shows whether the bound held
# - Figure 2: population risk per policy at matched coverage (CSE, fixed
#   methods, uncertainty gating, random, oracle)
# - Figure 3: Spearman correlation between epistemic uncertainty and
#   explanation risk, with bootstrap intervals
# - Figure 4: reliability diagrams for the classifiers
# - Figure 5: epistemic uncertainty of explained vs abstained instances
# - Every figure uses fig/ax and fig.savefig, writes PNG + PDF at 300 dpi
#   into project/figures/
# - make_all() renders whichever figures the supplied tables support
# ============================================================
