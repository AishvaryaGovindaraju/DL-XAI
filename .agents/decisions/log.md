# Historical Decisions & Migrations

## Migration Index
<!-- date | migration file | what changed -->
(none — no schema/migration concept in this project)

## Decisions

**2026-09 (PRD v3.0 frozen)** | Dataset: CDC Diabetes Health Indicators
(BRFSS2015, 253,680 rows, Kaggle) chosen over UCI 130-hospital diabetes and
Pima/UCI heart disease datasets | **Why:** large enough to justify a DNN over
simpler models, no patient-level leakage (unlike UCI 130-hospital), not
oversaturated in prior XAI literature (unlike Pima/UCI heart), clean binary
target, public provenance | **Rejected:** Pima Indians Diabetes (too small,
oversaturated in XAI papers), UCI 130-hospital (patient-level leakage risk).
See `UA_XAI_FINAL_PRD.md` Part 3.

**2026-09 (PRD v3.0 frozen)** | Conditional XAI assignment policy: LOW
uncertainty stratum → SHAP, MEDIUM → LIME, HIGH → DiCE | **Why:** SHAP's
Shapley attribution is most reliable when the model's decision boundary is
stable (low uncertainty); LIME's local surrogate approach tolerates moderate
uncertainty; DiCE counterfactuals stay informative ("what would change the
outcome") even when feature attribution is unreliable under high uncertainty.
Grounded in Jahn et al. (2026) findings that counterfactuals outperform
feature importance for understanding, and that feature-importance methods do
best under stable conditions | **Rejected:** uniform/fixed single-method
application across all strata (this is exactly the non-conditional baseline
the paper argues against — RQ1/H1). See `UA_XAI_FINAL_PRD.md` Part 7.

**2026-09 (PRD v3.0 frozen)** | Uncertainty quantification method: MC Dropout
(T=50 passes, `model.train()` kept active at inference) | **Why:** established,
well-cited technique (Gal & Ghahramani 2016) for epistemic uncertainty in
DNNs without needing an ensemble or Bayesian layers — keeps the novelty
claim scoped to *using* uncertainty for XAI selection, not inventing a new UQ
method (explicitly out of scope, PRD Part 1.3) | **Rejected:** deep ensembles,
full Bayesian NN (more expensive, and PRD explicitly disclaims "not a novel UQ
method" so simplest established technique was preferred).

**2026-09 (PRD v3.0 frozen)** | Target journals: *Expert Systems with
Applications* (primary), *Computers in Biology and Medicine* (secondary),
*Engineering Applications of AI* (tertiary), *IEEE Access* (backup, faster
turnaround) | **Why:** all SCI Q1/Q2, accept Indian-institution submissions,
fit for a computational (non-clinical-trial) healthcare XAI paper.
