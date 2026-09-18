# Active Roadmap & Technical Debt

## Backlog
Phases per `UA_XAI_FINAL_PRD.md` Part 10. Status as of 2026-09-18, verified
directly against `DL_XAI.ipynb` cell contents (not just commit messages —
an earlier version of this table wrongly marked Phase 3 done based on
commit messages alone; always check actual cell source for "done").

| Phase | Task | Status |
|-------|------|--------|
| 0 | Environment + folder setup | Done |
| 1 | Data loading + EDA | Done |
| 2 | Preprocessing pipeline (scale, split, class weights) | Done |
| 3 | Baseline models (Logistic Regression, XGBoost) | **Not implemented** — libraries imported in Phase 0 only, no training code anywhere in the notebook |
| 4 | DNN + MC Dropout training, uncertainty extraction | Done |
| 5 | Uncertainty stratification + sampling | Done (`stratified_samples.csv`) |
| 6A | SHAP explanations | Done (`shap_values.npy`, `shap_instance_ids.npy`) |
| 6B | LIME explanations | Done (`lime_outputs.pkl`) |
| 6C | DiCE counterfactuals | Done (`dice_outputs.pkl`); one diagnosed failure (instance 6718, see `subsystem-notes.md`) |
| 7 | XAI quality metrics (fidelity, stability, sparsity, validity, proximity, actionability) → `results/xai_metrics.csv` | **Not started — next up** |
| 8 | Statistical analysis (Kruskal-Wallis, Dunn's, Mann-Whitney U, effect sizes) + 6 paper figures | Not started |
| 9 | Paper writing (structure in PRD Part 11, ~6,500–7,500 words for ESWA) | Not started |

**Priority:** Phase 7 is the critical path — it's the only thing blocking
Phase 8 (stats) and therefore the whole paper-writing phase. Phase 3 (never
implemented) doesn't block anything downstream — it only feeds the model-
comparison table/figure in Phase 8, so it can be done any time before then.

## Notebook organization
As of 2026-09-18, `DL_XAI.ipynb` (monolithic, all phases) was split into
per-phase notebooks at `project/notebooks/phaseN_*.ipynb`, cells copied
verbatim with no code changes. See `stack-and-rules.md` File Map and
`subsystem-notes.md` cell-index mapping table. When Phase 7 work starts, add
a new `phase7_xai_quality_metrics.ipynb` there to keep the convention going.

## Known Tech Debt
- Output artifacts from Phases 5–6C live at repo root, not at the
  `project/data/processed/` / `project/results/xai_outputs/` paths the PRD's
  own phase prompts specify (see `subsystem-notes.md`). Needs a deliberate
  decision before Phase 7: keep reading from root, or do one clean move +
  path update.
- `project/src/**` is still all empty `.gitkeep` stubs — the PRD's modular
  file layout (PART 13) was never actually populated; everything is in the
  monolithic `DL_XAI.ipynb`. Not necessarily a problem (identity.md notes this
  is the expected working style) but worth flagging if the user later wants
  to extract reusable `.py` modules for the paper's code-availability
  statement.
- Mojibake (`�`) in a handful of notebook comment headers — cosmetic, low
  priority, see `subsystem-notes.md`.
