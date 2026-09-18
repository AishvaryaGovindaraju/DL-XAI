# Subsystem Notes & Load-Bearing Gotchas

## `DL_XAI.ipynb` execution environment
Cell 81 near the end of the notebook lists files under `/mnt/data` — the
notebook has been (at least partly) run in a hosted/sandboxed environment
(cloud code-interpreter style), not directly against the local `project/`
folder tree. This is why Phase 5–6C output artifacts (`stratified_samples.csv`,
`shap_values.npy`, `shap_instance_ids.npy`, `lime_outputs.pkl`,
`dice_outputs.pkl`) landed at the repo root instead of the
`project/data/processed/` and `project/results/xai_outputs/` paths the PRD's
phase prompts specify — they were generated remotely, then downloaded and
committed at the root. When continuing to Phase 7+, either keep reading
inputs from repo root (matches current reality) or do a deliberate one-time
move into `project/...` and update paths — don't silently assume the PRD's
stated paths are where data actually is.

## Mojibake in notebook comments
Comment headers like `STEP 4D � Verify Model Predictions...` (cell 78) show a
broken em-dash — an encoding round-trip issue (likely UTF-8 written/read as
Latin-1 somewhere in the notebook's edit history, possibly from the sandboxed
environment noted above). Cosmetic only, but don't "fix" by guessing the
character — replace with a plain `-` or re-type `—` in UTF-8 if touching those
cells.

## DiCE counterfactual failures are expected, not bugs
Cell 80 diagnoses a specific failed instance (test id 6718) where DiCE could
not generate valid counterfactuals. Per PRD Phase 6C verify-criteria, HIGH-
uncertainty-stratum instances are *expected* to have more `None` counterfactual
entries — this is itself a reportable finding for the paper (H3), not
something to debug away. Don't treat DiCE failures as an implementation bug
without first checking which stratum the failing instance is in.

## Baseline vs. primary model roles
Logistic Regression and XGBoost (PRD Part 6, Models 1–2) exist only as
performance benchmarks/motivation for using a DNN — they are not part of the
XAI evaluation pipeline. Only the DNN (Model 3, with MC Dropout) feeds into
SHAP/LIME/DiCE and the uncertainty stratification. **They were never actually
implemented** — cell 2/4 of `DL_XAI.ipynb` only `!pip install`/`import
xgboost` and `sklearn`; no `LogisticRegression` or `XGBClassifier` training
code exists anywhere in the notebook. Don't assume Phase 3 is done just
because the libraries are imported.

## Cell-index → PRD-phase mapping (`DL_XAI.ipynb`, 83 cells, as of 2026-09-18)
Used to produce `project/notebooks/phaseN_*.ipynb` (see `stack-and-rules.md`).
The notebook's real edit order does not match phase order cleanly — record
kept here so the mapping doesn't have to be re-derived by re-reading all 83
cells:

| Phase | Cell indices | Notes |
|-------|--------------|-------|
| 0 — Environment Setup | 0–4 | |
| 1 — Data Loading & EDA | 5–14 | |
| 2 — Preprocessing Pipeline | 15–26 | |
| 3 — Baseline Models | *(none — not implemented, see above)* | |
| 4 — DNN + MC Dropout | 27–52 | ends at first `mc_mean`/`mc_variance` calc |
| 5 — Uncertainty Stratification | 53, 54, 58–65 | 53–54 is a first-pass categorization later reworked; 58–65 (the researcher's own "STEP 1A"–"STEP 1F"/"STEP 1") redoes it correctly (fixes a BatchNorm-vs-Dropout train/eval bug) and produces the final `stratified_samples.csv` |
| 6A — SHAP | 66–70 (researcher's "STEP 2A"–"STEP 2D") | |
| 6B — LIME | 55, 56, 57, 71–74 | 55–57 is an early single-instance LIME test (explicitly commented `PHASE 6 — LIME EXPLAINER SETUP` in-notebook) that chronologically sits *before* the Phase-5 rework (58–65) but is LIME content, not stratification — grouped with the rest of LIME ("STEP 3A"–"STEP 3D", cells 71–74) rather than left orphaned |
| 6C — DiCE | 75–82 (researcher's "STEP 4A"–"STEP 4F") | includes trailing cell 81 (`/mnt/data` file listing) and empty cell 82 |

If asked to re-split, re-verify, or extend this mapping, treat cells 53–65
as the trickiest region — that's where MC-dropout-inference correctness and
uncertainty-stratification work are tangled together across two attempts.
