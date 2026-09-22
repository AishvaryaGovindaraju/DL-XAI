# `project/src` — the UA-XAI pipeline

The notebooks were the lab notebook: they contain the real work plus the
dead ends (two model definitions, two loss formulations, three attempts at
the uncertainty strata). These files are the same work with the final path
kept and the dead ends removed, one file per step of the workflow.

Every file ends with a `# Checklist` block summarising what it does.

## Read in this order

| File | Phase | What happens |
|------|-------|--------------|
| `config.py` | — | Every path, constant and hyper-parameter. Start here. |
| `data_loading.py` | 1 | Load the BRFSS2015 CSV, check integrity, describe it. |
| `preprocessing.py` | 2 | 70/15/15 stratified split, scaler fitted on train only, `Splits` object. |
| `model.py` | 4a | The `UADNN` architecture (21→128→64→32→1, sigmoid). |
| `training.py` | 4b | Class-weighted BCE, Adam, early stopping on validation AUC. |
| `evaluation.py` | 4c | Deterministic test metrics + the 0.72 AUC gate. |
| `mc_dropout.py` | 4d | T=50 stochastic passes → per-instance predictive variance. |
| `stratification.py` | 5 | LOW/MEDIUM/HIGH strata + the 600-instance sample. |
| `explain_shap.py` | 6A | SHAP attributions (+ additivity check). |
| `explain_lime.py` | 6B | LIME surrogates (+ local R² per instance). |
| `explain_dice.py` | 6C | DiCE counterfactuals (+ failure accounting). |
| `run_pipeline.py` | all | Runs the above in order; `--stages` re-runs a subset. |

## Running it

```bash
cd project/src
python run_pipeline.py                        # full pipeline
python run_pipeline.py --stages data          # EDA only
python run_pipeline.py --stages train evaluate mc strata
python run_pipeline.py --stages shap lime dice   # reuses the checkpoint
```

Modules import each other by plain name (`import config`), so run them from
this folder or add it to `sys.path` — `run_pipeline.py` does that itself.

## Verification status

Run end-to-end on a 6,000-row subsample with reduced settings (2 epochs,
T=5, 5 instances per stratum) purely to check the wiring:
`data → preprocessing → training → evaluation → MC Dropout →
stratification → SHAP → DiCE` all execute, the `--stages` checkpoint reload
works, and SHAP's additivity error was ~3e-8. **`explain_lime.py` was not
executed** — the `lime` package could not be installed in the test
environment; its code follows the notebook version unchanged. The full-scale
run (all 253,680 rows, 100 epochs, T=50, 600 instances) has not been rerun
here.

## Where the notebooks and these files differ

Three places, all deliberate:

1. **One model, not two.** The notebooks define `DiabetesDNN` (plain
   sequential, logit output) and then `UA_DNN` (BatchNorm, sigmoid output).
   Only the second one produced the reported results, so only it is here.
2. **MC Dropout keeps BatchNorm in eval mode.** `mc_dropout.py` puts the
   model in `eval()` and reactivates only the `nn.Dropout` modules. A plain
   `model.train()` would also make BatchNorm use batch statistics, so a
   patient's uncertainty would depend on who else is in their batch. Note
   this contradicts the hard invariant written in
   `.agents/context/stack-and-rules.md` ("MC Dropout inference must call
   `model.train()`"); the notebooks' final run — the one that produced
   `stratified_samples.csv` — used the eval-mode version. Pass
   `batchnorm_in_train_mode=True` to reproduce the other behaviour.
3. **Percentile strata, not the PRD's fixed thresholds.** The PRD's
   σ² cut-points (0.05 / 0.15) leave MEDIUM and HIGH empty on this model.
   `stratification.py` cuts equal thirds and keeps
   `compare_threshold_rules()` so the paper can show why.

## Not implemented (matching the current codebase)

- **Phase 3** — logistic-regression and XGBoost baselines.
- **Phase 7** — XAI quality metrics (fidelity, stability, sparsity, DiCE
  validity/proximity).
- **Phase 8** — statistical analysis (Kruskal-Wallis, Dunn, effect sizes)
  and paper figures.

These were never written in the notebooks either; the libraries are imported
in Phase 0 but no code uses them.

## Outputs

Written under the PRD paths:

```
project/data/processed/   uncertainty_stratification.csv, stratified_samples.csv
project/src/models/       scaler.pkl
project/results/          dnn_checkpoint.pt, dnn_metrics.csv, test_predictions.csv
project/results/xai_outputs/  shap_values.npy, shap_instance_ids.npy,
                              lime_outputs.pkl, dice_outputs.pkl
```

The original hosted-notebook runs wrote these to the repo root instead.
`config.resolve_input()` falls back to those root copies when a file is not
in the new location, so existing results still load.
