# `project/src` — Conformal Selective Explanation (CSE)

Decide, per patient, **which** post-hoc explanation to deliver — or to
deliver **none** — with a distribution-free bound on the quality of what is
delivered.

Every file ends with a short `# CHECKLIST` block. Read the files in the
order below and the whole study reads top to bottom.

## Reading order

| File | Role |
|------|------|
| `config.py` | every path, dataset, hyper-parameter and risk-control constant. Start here. |
| `data.py` | load/clean/split the three clinical datasets into one uniform `Dataset` |
| `models.py` | MLP, FT-Transformer, and the logistic-regression / gradient-boosting baselines |
| `training.py` | class-weighted training, early stopping, deep ensembles |
| `uq_methods.py` | MC Dropout + deep ensembles, aleatoric/epistemic decomposition, calibration |
| `evaluation.py` | predictive performance, calibration, and the deep-vs-shallow verdict |
| `xai.py` | SHAP / LIME / Integrated Gradients / DiCE behind one interface |
| `xai_metrics.py` | infidelity, instability, complexity, CF validity/proximity → one risk in [0,1] |
| `selective.py` | **the contribution**: quality head + conformal risk control + all policies |
| `statistical_tests.py` | bootstrap CIs, Wilcoxon, Friedman/Nemenyi, Spearman, Holm, effect sizes |
| `experiments.py` | the study: E1–E6 and the ablations, per dataset × architecture × seed |
| `figures.py` | the five paper figures |
| `run_pipeline.py` | CLI entry point |

`_v1_modules/` holds the previous (uncertainty-stratification) version of
the code, kept for reference. Nothing imports it.

## Running it

```bash
cd project/src
python run_pipeline.py --describe     # dataset table, then exit
python run_pipeline.py --smoke        # ~2 min wiring check (support2, tiny)
python run_pipeline.py --datasets support2 --seeds 0 --no-dice
python run_pipeline.py                # the full study
```

Outputs: `project/results/tables/*.csv`, `project/figures/fig*.png|pdf`.

## The pipeline in one paragraph

Three clinical tabular datasets are split 60/10/15/15 into
train/val/**calibration**/test. A deep ensemble is trained; its predictive
distribution is decomposed into aleatoric and epistemic uncertainty. Four
explainers run on a sample of calibration instances and a sample of test
instances, and every explanation is *measured* (faithfulness, stability,
validity, proximity). Those measurements are mapped through the calibration
split's ECDF into a common risk scale in [0,1]. A small quality head, reading
the classifier's penultimate representation plus its uncertainty, learns to
predict that risk before an explanation is computed. Conformal risk control
then converts the head's predictions into a delivery rule with a finite-sample
guarantee, and the policy is compared against fixed explainers, random
routing, the published uncertainty-gating baseline, and an oracle — all at
matched coverage.

## What is guaranteed, precisely

The conformal bound is on the **population** risk

```
E[ risk(x, m*(x)) · 1{deliver} ] ≤ α
```

with abstention contributing zero. That is exactly the quantity conformal
risk control covers, using the standard `α − (B−α)/n` correction. The
**selective** risk `E[risk | deliver]` is a ratio of random quantities and is
*not* covered by that theorem — it is reported empirically with bootstrap
intervals and is never described as guaranteed. Abstaining from everything
attains population risk 0 trivially, which is why the object of study is the
coverage–risk curve rather than the risk alone.

## Verification status

Run end-to-end on SUPPORT2 (MLP, ensemble K=3, 600 calibration + 300 test
instances, SHAP + IG, no DiCE): every stage executes, the quality head
reaches Spearman ρ = 0.694 against measured risk versus 0.281 for raw
epistemic uncertainty, and CSE reaches population risk 0.080 at α = 0.10 and
coverage 0.32 against 0.150 for the uncertainty-gating baseline at the same
coverage. Full numbers and the two design consequences are in §19 of
`CSE_PRD_v2.md`.

**Not executed here:** `lime` would not install in the verification
environment, so the LIME path is unexercised code; DiCE ran only in the
smoke test; and the full grid (3 datasets × 2 architectures × 3 seeds) has
not been run. Nothing in `results/tables/` yet is a publishable result.

## Three declared modelling choices

These are choices, not facts, and each is ablated rather than asserted:

1. **Cross-method comparability.** Infidelity (a squared prediction error)
   and counterfactual invalidity (a 0/1 event) are not in the same units.
   Each component is mapped through its own calibration-split ECDF, so risk
   means "worse than this fraction of calibration explanations of the same
   kind". `experiments.run_ablations()` re-runs the analysis under different
   component weights.
2. **Instance-level context.** The base paper's context is user/task/domain,
   established with human-grounded studies. This study operationalises
   context at the *instance* level with functional proxies and no human
   subjects. That is a narrower claim, stated as such.
3. **MC Dropout keeps BatchNorm in eval mode.** Only `nn.Dropout` modules are
   reactivated. A blanket `model.train()` would make a patient's uncertainty
   depend on who shares their batch.

## Differences from the previous version

The earlier design binned MC-Dropout predictive variance into LOW/MEDIUM/HIGH
and hard-assigned SHAP/LIME/DiCE to the bins. That was dropped for three
reasons, all documented in `UA_XAI_AUDIT_AND_PLANS.md`:

- the claim is preempted (arXiv 2603.29915, 2507.12913);
- every BRFSS test instance had σ² < 0.018, far below the PRD's own LOW
  cut-point of 0.05, so the strata were rank terciles, not uncertainty levels;
- total predictive variance conflates aleatoric with epistemic uncertainty,
  and only the epistemic part supports "the model is out of its depth here".

What survives: the DNN, MC Dropout, the three explainers, and the
healthcare framing — all now components of a mechanism that produces a
checkable guarantee instead of an asserted policy.

## Dependencies

`torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `shap`,
`lime`, `dice-ml` (`xgboost` optional — gradient boosting falls back to
scikit-learn's `HistGradientBoostingClassifier`). `shap`, `lime` and
`dice_ml` are imported lazily: a missing package disables that one explainer
with a printed note instead of breaking the run.
