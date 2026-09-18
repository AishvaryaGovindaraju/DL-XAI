# Invariants, Tech Stack & File Map

## Research Identity
Paper: *Uncertainty-Conditioned Post-Hoc XAI Selection for Deep Learning
Healthcare Decision Support* — UA-XAI framework. Uses MC Dropout epistemic
uncertainty to conditionally select among SHAP / LIME / DiCE per prediction,
evaluated on the CDC Diabetes Health Indicators (BRFSS2015) dataset.
Novelty claim, gap anchors, and full spec: `UA_XAI_FINAL_PRD.md` (frozen v3.0,
Sep 2026 — do not propose scope changes to this doc without the user asking).

## Tech Stack
- **Core:** Python, numpy, pandas, scikit-learn
- **Models:** `torch` (DNN + MC Dropout, primary model), `xgboost` and
  sklearn `LogisticRegression` (baselines)
- **XAI:** `shap` (DeepExplainer, fallback GradientExplainer), `lime`
  (LimeTabularExplainer), `dice-ml` (method='random')
- **Stats:** `scipy.stats` (Kruskal-Wallis, Mann-Whitney U), `scikit_posthocs`
  (Dunn's test), `pingouin` (effect sizes, CIs)
- **Viz:** matplotlib, seaborn (300 DPI minimum for paper figures)
- No web framework, no DB, no deployment target — this is an offline
  compute-and-write-a-paper pipeline.

## Hard Invariants
- **MC Dropout inference must call `model.train()`** (not `.eval()`) during
  the T=50 stochastic forward passes — this is what keeps Dropout active and
  is the entire mechanism the paper's uncertainty estimate depends on.
- **Uncertainty stratum thresholds:** LOW σ²<0.05, MEDIUM 0.05–0.15, HIGH
  σ²≥0.15 (adjustable to 0.03/0.10 only if a stratum drops below 150
  instances — PRD Phase 5).
- **Conditional XAI assignment policy is fixed and must be justified, not
  assumed, in the paper:** LOW→SHAP, MEDIUM→LIME, HIGH→DiCE (PRD Part 7).
- **DiCE immutable features:** `Age`, `Sex`, `Education` — never allowed to
  change in a counterfactual.
- **Minimum acceptable DNN AUC-ROC: 0.72** — below that, tune lr or add
  L2/weight_decay before proceeding to later phases.
- Every statistical result reported must include exact p-value + effect size
  + plain-language interpretation (see `identity.md`).

## File Map
- `DL_XAI.ipynb` (repo root) — **the original monolithic implementation**,
  all phases as sequential cells. Kept as-is; still the canonical source if
  the per-phase notebooks and it ever disagree.
- `project/notebooks/phaseN_*.ipynb` — as of 2026-09-18, the same code split
  into one notebook per PRD phase (cells copied verbatim, byte-for-byte, not
  re-implemented):
  - `phase0_environment_setup.ipynb`
  - `phase1_data_loading_eda.ipynb`
  - `phase2_preprocessing_pipeline.ipynb`
  - (no phase3 file — baseline LR/XGBoost models, PRD Part 6 Models 1–2, were
    never actually implemented in the notebook; only imported in Phase 0)
  - `phase4_dnn_mc_dropout.ipynb`
  - `phase5_uncertainty_stratification.ipynb`
  - `phase6a_shap_explanations.ipynb`
  - `phase6b_lime_explanations.ipynb`
  - `phase6c_dice_counterfactuals.ipynb`
  The split required judgment calls where the notebook's real edit history
  interleaved phases (an early one-instance LIME test sits between two
  rounds of uncertainty-stratification rework) — see `subsystem-notes.md`
  for the exact cell-index mapping and reasoning before assuming a boundary.
  These notebooks are **not standalone-runnable** — no re-import/re-setup
  boilerplate was added, they assume the same kernel state as running
  `DL_XAI.ipynb` cells in original order.
- `UA_XAI_FINAL_PRD.md` (repo root) — frozen spec, phase-by-phase prompts and
  verify-criteria. Treat as source of truth for "what should happen."
- `project/` (other subfolders: `src/{preprocessing,models,xai,evaluation,
  statistics}`, `data/{raw,processed}`, `results/xai_outputs`, `figures/`,
  `paper/`) — still empty `.gitkeep` stubs. **Do not assume code lives here.**
- Root-level `*.npy` / `*.pkl` / `*.csv` (e.g. `shap_values.npy`,
  `lime_outputs.pkl`, `dice_outputs.pkl`, `stratified_samples.csv`) are the
  actual Phase 5–6C outputs, saved at repo root rather than the
  `project/data/processed/` and `project/results/xai_outputs/` paths the PRD
  specifies.
