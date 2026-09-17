# FINAL PROJECT REQUIREMENTS DOCUMENT
## Uncertainty-Conditioned Post-Hoc XAI Selection for Deep Learning Healthcare Decision Support: A Functionally-Grounded Computational Evaluation

**Version:** 3.0 — FROZEN POST LITERATURE AUDIT  
**Status:** Implementation Ready  
**Base Paper:** Jahn, Hühn & Reichert (2026). *A Review on Empirical Studies in Explainable Artificial Intelligence.* Artificial Intelligence Review, 59(175). https://doi.org/10.1007/s10462-026-11595-6  
**Date Frozen:** September 2026

---

## PART 1 — RESEARCH IDENTITY

### 1.1 Precise Novelty Claim

> *This paper presents the first computationally evaluated framework that uses Monte Carlo Dropout epistemic uncertainty to conditionally select among post-hoc XAI methods (SHAP, LIME, DiCE) for a deep learning healthcare classifier on tabular data, evaluating explanation quality across uncertainty strata using functionally-grounded metrics and statistical analysis — operationalizing the context-sensitive XAI selection called for by Jahn et al. (2026) and the XAI-UQ integration proposed as future work in recent position papers.*

This claim is the only novelty claim in this paper. Every section of the paper points back to it.

### 1.2 What This Project Is

A purely computational study that:
1. Trains a DNN with MC Dropout on a healthcare tabular dataset
2. Quantifies epistemic uncertainty per prediction (MC Dropout, T=50 passes)
3. Stratifies test instances by uncertainty level
4. Applies SHAP, LIME, and DiCE to each stratum
5. Implements a conditional policy (one XAI method assigned per stratum)
6. Evaluates explanation quality computationally using 6 metrics
7. Statistically compares conditional (UA-XAI) vs. fixed (non-conditional) explanation quality

### 1.3 What This Project Is NOT

- Not a new XAI method
- Not a new uncertainty quantification method
- Not a clinical deployment system
- Not a human study
- Not adaptive real-time inference
- Not a claim that MC Dropout is novel

### 1.4 Gap Anchors (cite these in the paper)

| Gap | Citing Paper |
|-----|-------------|
| Context-sensitive XAI selection lacks empirical computational validation | Jahn et al. (2026), Section 2.3 |
| UQ + XAI integration proposed but not computationally evaluated on tabular healthcare DNN | Klause et al. (2025), ScienceDirect position paper |
| Healthcare XAI studies use single methods; multi-method quality evaluation is rare | Noor et al. (2025, WIREs); Caterson et al. (2024) |
| DiCE counterfactuals substantially underused in healthcare tabular evaluation | Caterson et al. (2024): LIME in 8/76 papers; DiCE near zero |
| Functionally-grounded evaluation called for as complement to human-grounded XAI review | Jahn et al. (2026), Section 2.2 (Doshi-Velez & Kim 2017 cited within) |

---

## PART 2 — TARGET JOURNALS

**Primary:** *Expert Systems with Applications* (Elsevier, SCI Q1, IF ~8.5)  
**Secondary:** *Computers in Biology and Medicine* (Elsevier, SCI Q1, IF ~7.0)  
**Tertiary:** *Engineering Applications of Artificial Intelligence* (Elsevier, SCI Q1, IF ~8.0)  
**Backup:** *IEEE Access* (IEEE, SCI Q2, IF ~3.9) — faster turnaround

All four accept papers from Indian institutions with no geographic restriction.

---

## PART 3 — DATASET

**Dataset:** CDC Diabetes Health Indicators Dataset  
**Source:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset  
**Instances:** 253,680  
**Features:** 21 (all structured, no free text)  
**Target:** `Diabetes_binary` (0 = no diabetes, 1 = diabetes)  
**License:** Public domain (BRFSS survey data)  
**Missing values:** None (pre-cleaned by CDC)

**Why this dataset over alternatives:**
- Large enough to justify DNN over simpler models
- No patient-level leakage issue (unlike UCI 130-hospital diabetes)
- Not oversaturated in XAI papers (unlike Pima, UCI heart disease)
- Binary classification = clean experimental setup
- Publicly citable with clear provenance

**Class balance check:** Run in Phase 1 EDA. If severely imbalanced (>4:1), apply class weighting to DNN loss (not oversampling on test set).

---

## PART 4 — RESEARCH QUESTIONS AND HYPOTHESES

### Research Questions

**RQ1:** Does a conditional XAI selection policy driven by MC Dropout uncertainty produce higher functionally-grounded explanation quality (fidelity, stability, sparsity) than applying each XAI method uniformly across all uncertainty levels?

**RQ2:** How do SHAP, LIME, and DiCE explanation quality metrics individually vary across low, medium, and high uncertainty strata?

**RQ3:** Which post-hoc XAI method demonstrates the most robust quality performance across all uncertainty strata when applied non-conditionally?

### Hypotheses

**H1:** The conditional UA-XAI framework will produce significantly higher mean fidelity and stability scores than any fixed single-method baseline (Mann-Whitney U, p < 0.05).

**H2:** SHAP will demonstrate higher stability than LIME across all uncertainty strata, consistent with its theoretical Shapley value grounding and prior computational benchmarks (Systematic Benchmarking paper, xAI 2025).

**H3:** DiCE counterfactual validity will decrease significantly in the high-uncertainty stratum compared to low-uncertainty, reflecting the model's reduced ability to produce reliable decision boundaries in uncertain regions.

**H4:** XAI quality metrics (fidelity, stability, sparsity) will show statistically significant differences across uncertainty strata (Kruskal-Wallis H-test, p < 0.05), confirming epistemic uncertainty as a meaningful stratification variable for XAI method selection.

---

## PART 5 — SYSTEM ARCHITECTURE

```
CDC Diabetes Dataset
(253,680 instances × 21 features)
          │
          ▼
┌─────────────────────────┐
│   Preprocessing         │
│   • StandardScaler      │
│   • Stratified 70/15/15 │
│   • Class weight calc.  │
└─────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────┐
│              Model Training              │
│                                          │
│  Logistic Regression  │  XGBoost  │ DNN │
│  (baseline)           │ (baseline)│(MC) │
└──────────────────────────────────────────┘
          │
          ▼ (DNN only from here)
┌─────────────────────────────────────────────┐
│        MC Dropout Inference (T=50)          │
│   For each test instance:                   │
│   • 50 stochastic forward passes            │
│   • μ = mean prediction                     │
│   • σ² = variance (epistemic uncertainty)  │
└─────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│          Uncertainty Stratification              │
│                                                  │
│  LOW     σ² < 0.05   → model is confident       │
│  MEDIUM  0.05–0.15   → moderate uncertainty      │
│  HIGH    σ² > 0.15   → model is uncertain        │
│                                                  │
│  Sample 200 instances per stratum for XAI        │
└──────────────────────────────────────────────────┘
          │
          ├─────────────────────────────────────────┐
          │                                         │
          ▼                                         ▼
┌────────────────────────┐            ┌─────────────────────────┐
│   UA-XAI (Conditional) │            │  Fixed Baselines         │
│                        │            │                          │
│  LOW    → SHAP         │            │  Fixed-SHAP (all strata) │
│  MEDIUM → LIME         │            │  Fixed-LIME (all strata) │
│  HIGH   → DiCE         │            │  Fixed-DiCE (all strata) │
└────────────────────────┘            └─────────────────────────┘
          │                                         │
          └──────────────┬──────────────────────────┘
                         ▼
          ┌──────────────────────────────┐
          │   XAI Quality Evaluation     │
          │                              │
          │  SHAP/LIME: Fidelity         │
          │             Stability        │
          │             Sparsity         │
          │                              │
          │  DiCE:      Validity         │
          │             Proximity        │
          │             Sparsity         │
          │             Actionability    │
          └──────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │     Statistical Analysis     │
          │                              │
          │  Kruskal-Wallis across strata│
          │  Dunn post-hoc (Bonferroni)  │
          │  Mann-Whitney U: UA vs fixed │
          │  Effect sizes: Cohen's d, η² │
          └──────────────────────────────┘
                         │
                         ▼
                       Paper
```

---

## PART 6 — MODELS

### Model 1: Logistic Regression (Baseline)
- Library: `scikit-learn`
- Purpose: Interpretable reference point
- No hyperparameter tuning needed beyond `max_iter=1000`

### Model 2: XGBoost (Strong Baseline)
- Library: `xgboost`
- Purpose: Best tabular ML benchmark; shows DNN adds complexity that motivates XAI
- Default hyperparameters acceptable for baseline comparison

### Model 3: DNN with Monte Carlo Dropout (Primary Model)
- Library: `torch`
- Architecture:

```python
class UA_DNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.drop3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.fc3(x)))
        return torch.sigmoid(self.fc4(x))
```

- Loss: `BCELoss` with class weights
- Optimizer: Adam, lr=1e-3
- Early stopping: patience=10 on validation AUC-ROC
- **Minimum acceptable AUC-ROC:** 0.72 — if below, tune lr or add L2 regularization before proceeding

### MC Dropout Inference

```python
def mc_predict(model, x, T=50):
    model.train()          # keeps dropout ACTIVE
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(T)], dim=0)
    mu = preds.mean(dim=0)       # mean prediction
    sigma2 = preds.var(dim=0)    # epistemic uncertainty
    return mu.cpu().numpy(), sigma2.cpu().numpy()
```

**Critical:** `model.train()` must be called during inference to keep Dropout active. This is the MC Dropout technique (Gal & Ghahramani, 2016 — cite this).

---

## PART 7 — XAI METHODS AND CONDITIONAL POLICY

### Assignment Policy

| Uncertainty Stratum | Assigned XAI Method | Justification |
|---------------------|--------------------|-----------------------------|
| LOW (σ² < 0.05) | SHAP | Model is confident; SHAP's Shapley value attribution is most reliable when the model's decision boundary is stable. Consistent with Jahn et al.'s finding that feature importance methods perform best when conditions are stable. |
| MEDIUM (0.05–0.15) | LIME | Local surrogate approximation is appropriate at moderate uncertainty; LIME's perturbation approach is less sensitive to exact model internals than SHAP. |
| HIGH (σ² > 0.15) | DiCE | When model is uncertain, feature attribution is unreliable; counterfactuals answer "what would change the outcome" regardless of model confidence, offering actionable information even in uncertain regions. Consistent with Jahn et al.'s finding that counterfactuals outperform feature importance in direct comparisons for understanding. |

**This policy is hypothesis-driven. It must be stated as a principled design choice in the paper, not as an assumption.**

### SHAP Implementation
- Library: `shap`
- Explainer: `shap.DeepExplainer` (designed for PyTorch DNNs)
- Background: 100-instance random sample from training set
- Output: SHAP values matrix (instances × features)
- Applied to: ALL strata for individual evaluation; LOW stratum for UA-XAI

### LIME Implementation
- Library: `lime`
- Explainer: `lime.lime_tabular.LimeTabularExplainer`
- Mode: `classification`, `kernel_width='auto'`
- Output: Feature weight dictionary per instance
- Applied to: ALL strata for individual evaluation; MEDIUM stratum for UA-XAI
- Note: LIME is slow. Limit to 200 instances per stratum maximum.

### DiCE Implementation
- Library: `dice-ml`
- Method: `random` (faster than genetic; sufficient for this study)
- Counterfactuals per instance: 3
- Immutable features: `Age`, `Sex`, `Education` (cannot be changed by intervention)
- Output: Counterfactual dataframes per instance
- Applied to: ALL strata for individual evaluation; HIGH stratum for UA-XAI

---

## PART 8 — XAI QUALITY METRICS

All metrics are computed computationally. No human judgment required.

### Metrics for SHAP and LIME

| Metric | Definition | Implementation |
|--------|-----------|----------------|
| **Fidelity** | Pearson correlation between model output on top-k masked input and full model output | Mask top-k features (by abs SHAP/LIME weight), re-predict, correlate |
| **Stability** | Mean cosine similarity of explanation vectors across 10 nearest neighbors | KNN on test set; cosine similarity of attribution vectors |
| **Sparsity** | 1 − (features with nonzero weight / total features) | Count nonzero attributions per instance |

### Metrics for DiCE

| Metric | Definition | Implementation |
|--------|-----------|----------------|
| **Validity** | Fraction of generated counterfactuals that flip the prediction | Re-run model on each counterfactual; check prediction class |
| **Proximity** | Normalized L1 distance between original instance and counterfactual | `sum(|x - x'|) / n_features` |
| **Sparsity** | 1 − (features changed / total features) | Count changed features per counterfactual |
| **Actionability** | Fraction of changed features that are NOT in immutable list | Count changes to mutable features only |

### Output Schema

All metrics are saved to `results/xai_metrics.csv` with columns:

```
instance_id | stratum | xai_method | condition | fidelity | stability | 
sparsity | validity | proximity | actionability
```

`condition` = `UA` (from conditional framework) or `fixed-SHAP` / `fixed-LIME` / `fixed-DiCE`

---

## PART 9 — STATISTICAL ANALYSIS PLAN

### Test 1: Stratum effect on XAI quality (addresses H4)
- **Test:** Kruskal-Wallis H-test
- **Applied to:** Each metric × each XAI method
- **Groups:** LOW vs MEDIUM vs HIGH stratum
- **Post-hoc:** Dunn's test with Bonferroni correction (if H-test significant)
- **Effect size:** η² (eta-squared)

### Test 2: UA-XAI vs fixed baselines (addresses H1)
- **Test:** Mann-Whitney U
- **Applied to:** UA-XAI vs Fixed-SHAP, UA-XAI vs Fixed-LIME, UA-XAI vs Fixed-DiCE
- **Metrics:** All applicable quality metrics
- **Effect size:** Cohen's d

### Test 3: SHAP vs LIME stability comparison (addresses H2)
- **Test:** Mann-Whitney U per stratum
- **Metrics:** Stability, Fidelity

### Test 4: DiCE validity across strata (addresses H3)
- **Test:** Kruskal-Wallis + Dunn's post-hoc
- **Metric:** Validity

### Reporting standard (non-negotiable for Q1 submission)
Every statistical result must report:
- Test statistic (H or U value)
- p-value (exact, not just "< 0.05")
- Effect size with interpretation (small / medium / large)
- 95% confidence interval where applicable
- Plain-language sentence interpreting the result

### Libraries
```
scipy.stats         # Kruskal-Wallis, Mann-Whitney U
scikit_posthocs     # Dunn's test
pingouin            # Effect sizes, CI
```

---

## PART 10 — IMPLEMENTATION PHASES

Each phase produces a specific deliverable. Do not proceed to the next phase without verifying the deliverable.

**How to use these phases:** For each phase, copy the phase description to your AI assistant (ChatGPT or Claude) and say: *"Implement this exactly. Give me complete, runnable Python code for this phase only."*

---

### PHASE 0 — Environment Setup

**Goal:** Working Python environment with all required libraries.

**Prompt to AI assistant:**
> "Create a requirements.txt for a Python 3.10 project with these libraries: torch, shap, lime, dice-ml, xgboost, scikit-learn, pandas, numpy, scipy, scikit_posthocs, pingouin, matplotlib, seaborn, kaggle. Also create this folder structure: project/data/raw, project/data/processed, project/src/preprocessing, project/src/models, project/src/xai, project/src/evaluation, project/src/statistics, project/notebooks, project/results/xai_outputs, project/figures, project/paper."

**Deliverable:** `requirements.txt` + folder structure created  
**Verify:** Run `pip install -r requirements.txt` with no errors

---

### PHASE 1 — Data Loading and EDA

**Goal:** Understand the dataset before touching it.

**Prompt:**
> "Load the CDC Diabetes Health Indicators dataset (CSV). Produce: (1) shape and column list, (2) class balance of Diabetes_binary, (3) missing value count per column, (4) descriptive statistics per feature, (5) correlation heatmap, (6) distribution plots for continuous features. Save all plots to figures/. Print all stats to console."

**Deliverable:** `notebooks/01_eda.ipynb`  
**Verify:** 253,680 rows, 21 columns, Diabetes_binary is 0/1, no missing values confirmed

---

### PHASE 2 — Preprocessing Pipeline

**Goal:** Clean data and create reproducible train/val/test splits.

**Prompt:**
> "Build a preprocessing pipeline for the CDC diabetes dataset. Steps: (1) Verify no missing values. (2) Apply StandardScaler to all continuous features (BMI, MentHlth, PhysHlth, Age). (3) Stratified split: 70% train, 15% val, 15% test — use stratify=Diabetes_binary. (4) Compute class weights for DNN: weight = n_samples / (n_classes * n_samples_per_class). (5) Save train.csv, val.csv, test.csv to data/processed/. (6) Save scaler as scaler.pkl. Print split sizes and class balance per split."

**Deliverable:** `src/preprocessing/pipeline.py` + saved CSVs + `scaler.pkl`  
**Verify:** Three CSVs exist, train has ~177k rows, splits are stratified (same class ratio as original)

---

### PHASE 3 — Baseline Models

**Goal:** Establish performance benchmarks before training the DNN.

**Prompt:**
> "Train Logistic Regression and XGBoost on the preprocessed diabetes training data (train.csv). Evaluate both on test.csv. Report: AUC-ROC, F1 (macro), Precision, Recall, Accuracy for each model. Save results to results/baseline_metrics.csv. Generate a bar chart comparing both models on all metrics and save to figures/baseline_comparison.png."

**Deliverable:** `src/models/baselines.py` + `results/baseline_metrics.csv` + figure  
**Verify:** Both models train without error; AUC-ROC for LR should be ~0.70–0.74, XGBoost ~0.74–0.80

---

### PHASE 4 — DNN with Monte Carlo Dropout

**Goal:** Train the primary model and extract per-instance uncertainty estimates.

**Prompt:**
> "Build and train a PyTorch DNN with Monte Carlo Dropout for binary classification on the CDC diabetes dataset. Architecture: Input → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(32) → ReLU → Dropout(0.2) → Linear(1) → Sigmoid. Training: BCELoss with class weights, Adam optimizer lr=0.001, early stopping patience=10 on validation AUC-ROC, max 100 epochs. For MC Dropout inference: call model.train() during inference, run T=50 forward passes per instance, compute mean (mu) and variance (sigma2) across passes. Run MC inference on ALL test instances. Save to results/test_predictions.csv with columns: instance_id, true_label, mu, sigma2. Evaluate on test set: AUC-ROC, F1, Precision, Recall. Plot train/val loss curves. Save model checkpoint to results/dnn_checkpoint.pt."

**Deliverable:** `src/models/dnn_mc.py` + checkpoint + `results/test_predictions.csv` + `results/dnn_metrics.csv`  
**Verify:** AUC-ROC ≥ 0.72. If below, tune learning rate or add weight_decay=1e-4 to Adam. test_predictions.csv has mu and sigma2 for all test instances.

---

### PHASE 5 — Uncertainty Stratification

**Goal:** Assign test instances to uncertainty strata and sample for XAI.

**Prompt:**
> "Load results/test_predictions.csv. Assign each instance to a stratum based on sigma2: LOW if sigma2 < 0.05, MEDIUM if 0.05 <= sigma2 < 0.15, HIGH if sigma2 >= 0.15. Print stratum sizes. If any stratum has fewer than 150 instances, adjust thresholds (try 0.03 and 0.10) and report new sizes. From each stratum, randomly sample 200 instances (or all if fewer than 200). Save to data/processed/stratified_samples.csv with columns: instance_id, stratum, mu, sigma2, true_label, and all original feature columns. Print final stratum sizes."

**Deliverable:** `data/processed/stratified_samples.csv`  
**Verify:** File has 3 strata, each with 150–200 instances, no overlap between strata

---

### PHASE 6A — SHAP Explanations

**Goal:** Compute SHAP values for all stratified instances.

**Prompt:**
> "Compute SHAP values using shap.DeepExplainer for the PyTorch DNN on all instances in data/processed/stratified_samples.csv. Use 100 random training instances as background. Output: a numpy array of shape (n_instances, n_features) with SHAP values. Save as results/xai_outputs/shap_values.npy and save corresponding instance IDs as results/xai_outputs/shap_instance_ids.npy. If DeepExplainer causes errors with the DNN architecture, use shap.GradientExplainer as fallback."

**Deliverable:** `src/xai/shap_explainer.py` + saved .npy files  
**Verify:** shap_values.npy has shape (600, 21) — 200 per stratum × 3 strata, 21 features

---

### PHASE 6B — LIME Explanations

**Goal:** Compute LIME explanations for all stratified instances.

**Prompt:**
> "Compute LIME explanations using lime.lime_tabular.LimeTabularExplainer for the PyTorch DNN on all instances in data/processed/stratified_samples.csv. Use training data as background reference. For each instance, get the explanation for class 1 (diabetic). Output: a dictionary mapping instance_id to a list of (feature_index, weight) tuples. Save as results/xai_outputs/lime_outputs.pkl. If LIME is slow (>5 min per 100 instances), reduce num_samples to 500 and state this in a comment."

**Deliverable:** `src/xai/lime_explainer.py` + `results/xai_outputs/lime_outputs.pkl`  
**Verify:** pkl file loads; each instance_id has a list of (feature_idx, weight) tuples

---

### PHASE 6C — DiCE Counterfactuals

**Goal:** Generate DiCE counterfactuals for all stratified instances.

**Prompt:**
> "Generate DiCE counterfactuals using dice-ml for the PyTorch DNN on all instances in data/processed/stratified_samples.csv. Settings: method='random', total_CFs=3, desired_class='opposite'. Define immutable_features=['Age', 'Sex', 'Education']. For each instance, save the 3 counterfactual rows plus original instance as a DataFrame. Save all results as results/xai_outputs/dice_outputs.pkl, a dictionary mapping instance_id to the counterfactual DataFrame. If an instance fails to generate valid counterfactuals, record None for that instance_id and continue."

**Deliverable:** `src/xai/dice_explainer.py` + `results/xai_outputs/dice_outputs.pkl`  
**Verify:** pkl loads; instances in HIGH stratum may have more None entries (expected and is itself a finding)

---

### PHASE 7 — XAI Quality Metrics

**Goal:** Compute all quality metrics for all instances in all conditions.

**Prompt:**
> "Implement these XAI quality metrics as Python functions and compute them for all instances in stratified_samples.csv using outputs from Phases 6A, 6B, 6C.
>
> For SHAP and LIME:
> - Fidelity: For each instance, mask all but top-5 features (by absolute attribution weight), run DNN prediction on masked input, compute Pearson correlation between masked-input predictions and full-input predictions across the 200 instances in each stratum.
> - Stability: For each instance, find its 10 nearest neighbors in the test set (Euclidean distance on scaled features). Compute SHAP/LIME explanations for those neighbors. Compute mean cosine similarity between the instance's explanation vector and each neighbor's explanation vector. Average across all instances in stratum.
> - Sparsity: 1 - (count of features with |weight| > 0.001) / total_features. Per instance, then average per stratum.
>
> For DiCE:
> - Validity: fraction of the 3 counterfactuals that produce a different predicted class when run through the DNN.
> - Proximity: mean normalized L1 distance = mean(sum(|x - x_cf|) / n_features) across 3 counterfactuals.
> - Sparsity: mean(1 - changed_features / total_features) across 3 counterfactuals.
> - Actionability: fraction of changed features NOT in ['Age','Sex','Education'] across 3 counterfactuals.
>
> Compute for ALL strata for ALL methods individually AND for the UA-XAI condition (LOW→SHAP, MEDIUM→LIME, HIGH→DiCE).
>
> Save ALL results to results/xai_metrics.csv with columns: instance_id, stratum, xai_method, condition, fidelity, stability, sparsity, validity, proximity, actionability.
> condition values: 'UA', 'fixed-SHAP', 'fixed-LIME', 'fixed-DiCE'"

**Deliverable:** `src/evaluation/xai_metrics.py` + `results/xai_metrics.csv`  
**Verify:** CSV has rows for every instance × method × condition combination; no column is entirely NaN

---

### PHASE 8 — Statistical Analysis and Figures

**Goal:** Run all statistical tests and produce all paper figures.

**Prompt:**
> "Load results/xai_metrics.csv. Run the following statistical analyses:
>
> ANALYSIS 1 — Stratum effect (H4):
> For each metric × XAI method: Kruskal-Wallis H-test across the 3 strata.
> For significant results (p < 0.05): Dunn's post-hoc with Bonferroni correction.
> Compute η² effect size = H / (n-1).
>
> ANALYSIS 2 — UA-XAI vs fixed baselines (H1):
> Mann-Whitney U: UA vs fixed-SHAP, UA vs fixed-LIME, UA vs fixed-DiCE.
> Apply to fidelity, stability, sparsity (for SHAP/LIME strata) and validity, proximity, actionability (for DiCE strata).
> Compute Cohen's d effect size.
>
> ANALYSIS 3 — SHAP vs LIME stability per stratum (H2):
> Mann-Whitney U per stratum for stability metric.
>
> ANALYSIS 4 — DiCE validity across strata (H3):
> Kruskal-Wallis + Dunn's post-hoc on validity metric.
>
> Save all results to results/statistical_results.csv with columns: test, metric, xai_method, stratum_comparison, statistic, p_value, effect_size, effect_interpretation.
>
> FIGURES — generate and save to figures/:
> 1. fig1_architecture.png — skip (will be drawn manually for paper)
> 2. fig2_model_comparison.png — bar chart: LR vs XGBoost vs DNN, 4 metrics
> 3. fig3_uncertainty_distribution.png — histogram of sigma2 across test set with stratum boundaries marked
> 4. fig4_fidelity_boxplots.png — box plots: SHAP and LIME fidelity across 3 strata
> 5. fig5_stability_boxplots.png — box plots: SHAP and LIME stability across 3 strata
> 6. fig6_dice_metrics.png — grouped bar chart: DiCE validity + actionability per stratum
> 7. fig7_ua_vs_fixed_heatmap.png — heatmap: UA-XAI vs all fixed baselines × all metrics (mean values)
>
> All figures: seaborn style, minimum 300 DPI, clear axis labels, no chart junk."

**Deliverable:** `src/statistics/analysis.py` + `results/statistical_results.csv` + all 6 figures  
**Verify:** statistical_results.csv has all 4 analysis types; all 6 figures render correctly

---

### PHASE 9 — Paper Writing

Write only after all results are confirmed and saved. Use the structure in Part 11.

---

## PART 11 — PAPER STRUCTURE

| Section | Content | Target Words |
|---------|---------|-------------|
| Abstract | Problem, framework, dataset, key finding, implication | 250 |
| 1. Introduction | Black-box DL → XAI need → context-sensitivity gap (Jahn et al.) → uncertainty as context variable → this paper's contribution → paper structure | 800 |
| 2. Related Work | 2.1 XAI methods (SHAP, LIME, DiCE), 2.2 MC Dropout UQ, 2.3 XAI + UQ prior work (position paper + Wickstrøm 2020), 2.4 Healthcare tabular XAI (Caterson 2024, Noor 2025), 2.5 Jahn et al. 2026 as anchor and gap | 1,200 |
| 3. Research Gap | Operationalize: Jahn et al. show context matters empirically, call for functionally-grounded evaluation, UQ never used as context variable computationally → our contribution | 400 |
| 4. Proposed UA-XAI Framework | Full architecture, MC Dropout, uncertainty strata, conditional selection policy with justification, quality metrics formal definitions | 1,000 |
| 5. Experimental Setup | Dataset, preprocessing, model architectures, training details, stratum thresholds, XAI implementation details, statistical plan | 700 |
| 6. Results | 6.1 Model performance table, 6.2 Uncertainty distribution, 6.3 XAI quality per stratum, 6.4 UA-XAI vs fixed comparison, 6.5 Statistical results | 1,200 |
| 7. Discussion | H1–H4 confirmed/rejected with evidence, practical implications, connection to Jahn et al. findings, limitations | 800 |
| 8. Conclusion | Summary + future work (human study extension, multi-dataset, real-time deployment) | 300 |
| References | 35–50 citations | — |

**Total target: 6,500–7,500 words** (standard for ESWA)

---

## PART 12 — REQUIRED FIGURES AND TABLES

| ID | Type | Content |
|----|------|---------|
| Fig 1 | Architecture diagram | Full UA-XAI pipeline (draw manually in draw.io or PowerPoint) |
| Fig 2 | Bar chart | LR vs XGBoost vs DNN: AUC-ROC, F1, Precision, Recall |
| Fig 3 | Histogram | σ² distribution across test set with stratum boundaries |
| Fig 4 | Box plots | Fidelity: SHAP and LIME across LOW/MED/HIGH strata |
| Fig 5 | Box plots | Stability: SHAP and LIME across LOW/MED/HIGH strata |
| Fig 6 | Bar chart | DiCE: Validity and Actionability per stratum |
| Fig 7 | Heatmap | UA-XAI vs fixed baselines × all metrics (mean values) |
| Table 1 | Dataset summary | N, features, class balance, data source |
| Table 2 | Model comparison | AUC-ROC, F1, Precision, Recall, ECE per model |
| Table 3 | Stratum sizes | N per stratum, σ² range, sampled N |
| Table 4 | XAI quality: mean ± SD | All metrics × method × stratum |
| Table 5 | Kruskal-Wallis results | H, df, p, η² per metric × method |
| Table 6 | UA vs fixed comparison | Mann-Whitney U, p, Cohen's d per comparison |

---

## PART 13 — FOLDER STRUCTURE

```
project/
├── data/
│   ├── raw/
│   │   └── diabetes_binary_health_indicators_BRFSS2015.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── stratified_samples.csv
│   │   └── scaler.pkl
│   └── README.md
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py
│   ├── models/
│   │   ├── baselines.py
│   │   └── dnn_mc.py
│   ├── xai/
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   └── dice_explainer.py
│   ├── evaluation/
│   │   └── xai_metrics.py
│   └── statistics/
│       └── analysis.py
├── notebooks/
│   └── 01_eda.ipynb
├── results/
│   ├── baseline_metrics.csv
│   ├── dnn_metrics.csv
│   ├── dnn_checkpoint.pt
│   ├── test_predictions.csv
│   ├── xai_metrics.csv
│   ├── statistical_results.csv
│   └── xai_outputs/
│       ├── shap_values.npy
│       ├── shap_instance_ids.npy
│       ├── lime_outputs.pkl
│       └── dice_outputs.pkl
├── figures/
│   ├── fig2_model_comparison.png
│   ├── fig3_uncertainty_distribution.png
│   ├── fig4_fidelity_boxplots.png
│   ├── fig5_stability_boxplots.png
│   ├── fig6_dice_metrics.png
│   └── fig7_ua_vs_fixed_heatmap.png
├── paper/
│   └── manuscript.docx
├── requirements.txt
└── README.md
```

---

## PART 14 — TECHNOLOGY STACK

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| pandas | latest | Data manipulation |
| numpy | latest | Arrays |
| scikit-learn | latest | LR, preprocessing, KNN, metrics |
| xgboost | latest | XGBoost baseline |
| torch | 2.x | DNN + MC Dropout |
| shap | latest | SHAP DeepExplainer / GradientExplainer |
| lime | latest | LIME tabular |
| dice-ml | latest | DiCE counterfactuals |
| scipy | latest | Kruskal-Wallis, Mann-Whitney U |
| scikit_posthocs | latest | Dunn's test |
| pingouin | latest | Effect sizes, CI |
| matplotlib | latest | Figures |
| seaborn | latest | Figure styling |

**Colab note:** All above run on free Google Colab GPU. DNN training on this dataset takes ~15–25 minutes on Colab T4. Save checkpoint to Google Drive after Phase 4 to avoid losing it on session timeout.

---

## PART 15 — EXECUTION ORDER

```
PHASE 0  → Environment + folder structure
    ↓
PHASE 1  → EDA: understand the data
    ↓
PHASE 2  → Preprocessing: clean + split
    ↓
PHASE 3  → Baselines: LR + XGBoost
    ↓
PHASE 4  → DNN + MC Dropout training
         CHECKPOINT: save model to Drive
    ↓
PHASE 5  → Uncertainty stratification
    ↓
PHASE 6A → SHAP explanations
    ↓
PHASE 6B → LIME explanations
    ↓
PHASE 6C → DiCE counterfactuals
    ↓
PHASE 7  → Quality metrics computation
    ↓
PHASE 8  → Statistical analysis + figures
    ↓
PHASE 9  → Paper writing
```

**Do not skip any phase. Each phase output is required input for the next phase.**

---

## PART 16 — REVIEWER DEFENSE TABLE

| Reviewer Attack | Your Defense |
|----------------|-------------|
| "SHAP, LIME, DiCE are not novel" | Correct. Our contribution is the uncertainty-conditioned selection framework and its computational quality evaluation — not the individual methods. |
| "MC Dropout is not novel" | Correct. The novelty is using it as an uncertainty signal to drive XAI method selection — not proposed as a new UQ technique. |
| "A position paper already proposed UQ + XAI integration" | Correct. We are the first to implement and computationally evaluate this integration with a conditional selection policy on a healthcare tabular DNN. Proposal ≠ implementation + evaluation. |
| "No human study" | This paper contributes functionally-grounded evaluation, explicitly identified as a complementary methodology by Doshi-Velez & Kim (2017) and cited by Jahn et al. (2026) Section 2.2. Human study is future work. |
| "Dataset is from 2015" | The BRFSS CDC dataset is a methodological benchmark used in peer-reviewed XAI papers. This study is about the framework, not about current epidemiology. |
| "Uncertainty strata thresholds are arbitrary" | Thresholds were set based on the empirical distribution of σ² in our test set and adjusted to ensure ≥150 instances per stratum. We report all thresholds explicitly and provide the full distribution in Fig 3. |
| "Why these three XAI methods" | SHAP is the dominant method in EHR tabular XAI (Caterson 2024: 63/76 papers). LIME is the primary comparison method. DiCE represents counterfactual explanations shown by Jahn et al. (2026) to outperform feature importance in direct comparisons for understanding. Together they cover the three main result types in Speith's (2022) taxonomy: feature importance, local surrogate, and counterfactual. |

---

## PART 17 — WHAT NOT TO CLAIM

| Never claim | Correct framing |
|------------|----------------|
| "We invented SHAP / LIME / DiCE / MC Dropout" | "We integrate established methods into a novel framework" |
| "Uncertainty-aware XAI has never been done" | "Uncertainty-conditional XAI selection with functionally-grounded quality evaluation on tabular healthcare DNN data has not been implemented or evaluated" |
| "This is clinically deployable" | "This is a methodological benchmark study on a public health dataset" |
| "Our framework always outperforms fixed XAI" | Report results honestly including any null findings |
| "Healthcare XAI is unexplored" | "Healthcare is the most studied XAI domain (Jahn et al. 2026); we contribute computational depth on a specific open gap" |

---

## PART 18 — RISK REGISTER

| Risk | Probability | Mitigation |
|------|------------|------------|
| DNN AUC < 0.72 | Low | Add L2 regularization (weight_decay=1e-4); try lr=5e-4; report actual AUC transparently |
| Strata severely unbalanced | Medium | Adjust σ² thresholds after Phase 5; minimum 150 per stratum; report actual thresholds used |
| DiCE fails to generate counterfactuals in HIGH stratum | Medium | **This confirms H3** — report None rate as a quantitative finding; do not treat as failure |
| SHAP DeepExplainer errors with DNN architecture | Low | Fall back to shap.GradientExplainer; note in methods section |
| LIME too slow | Medium | Reduce num_samples to 500 (from default 5000); state explicitly in methods |
| No significant stratum effect found (H4 rejected) | Low-Medium | Null result is publishable if properly reported; discuss what it means for the conditional policy assumption |
| Colab session timeout during Phase 4 | High | Mount Google Drive before training; save checkpoint every 10 epochs |

---

*PRD Frozen. Proceed to Phase 0.*
