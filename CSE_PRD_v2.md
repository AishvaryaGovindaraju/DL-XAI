# PRD v2 — Conformal Selective Explanation (CSE)

**Risk-controlled post-hoc explanation delivery for deep tabular clinical
models.**

Supersedes `UA_XAI_FINAL_PRD.md`. The reasons for superseding it are in
`UA_XAI_AUDIT_AND_PLANS.md`; the short version is that the previous central
claim is preempted by two 2025–2026 papers and its uncertainty strata were
rank terciles of a variance range in which no instance exceeded the PRD's own
LOW cut-point.

Status: implemented in `project/src/`, verified end-to-end. Numbers in this
document are targets and protocol, not results — with one exception, §19,
which records what a single verification run actually produced and the two
places where it changes the design.

---

## 1. Identity and scope

Deep learning + healthcare decision support + explainability + uncertainty.
All four are structural, none decorative:

- **Deep learning** supplies the predictive model, the ensemble whose
  disagreement is the epistemic signal, and — critically — the *learned
  representation* that the explanation-quality head conditions on. A
  logistic regression cannot host this mechanism; it has no representation.
- **Healthcare** supplies three real clinical tabular tasks and the
  motivation for abstention: showing a clinician an unreliable explanation is
  not a neutral act.
- **XAI** is the object being decided about (SHAP, LIME, IG, DiCE).
- **Uncertainty** is decomposed, calibrated, and used as one input among
  others rather than as the whole mechanism.

**Out of scope, explicitly:** human-subject evaluation. This study measures
functional explanation properties, not clinician outcomes. That boundary is
stated in the paper, not blurred.

---

## 2. Research question and hypotheses

> For each patient, which explanation should be delivered — and should one be
> delivered at all — such that the quality of what is delivered satisfies a
> bound chosen in advance?

| ID | Hypothesis | Decided by |
|----|-----------|-----------|
| **H1** | A quality head reading the classifier's penultimate representation predicts per-instance explanation risk better than epistemic uncertainty alone. | Spearman ρ against measured risk, bootstrap CI, paired across seeds/datasets (E3) |
| **H2** | At matched coverage, CSE delivers lower population explanation risk than any fixed explainer, random routing, and uncertainty gating. | Paired Wilcoxon across instances, Holm-corrected; Friedman + Nemenyi across datasets (E4) |
| **H3** | The realised population risk of delivered explanations respects the requested level α. | Empirical risk vs α across α ∈ {0.05…0.30}, all datasets × seeds (E5) |
| **H4** | Abstention concentrates on instances with high epistemic uncertainty. | Mean epistemic uncertainty delivered vs abstained, Cliff's δ (E6) |
| **H5** | Epistemic uncertainty correlates negatively with explanation quality (i.e. positively with risk) — the published result, retested here. | Spearman ρ per method per dataset (E2) |

H5 is a **replication**, labelled as such. If it fails on our datasets that is
a reportable finding, not a problem to hide.

---

## 3. Positioning against the literature

**Already established — cited, not claimed:**

- Epistemic uncertainty as a proxy for explanation reliability, and routing
  between explainers by it — *Uncertainty Gating for Cost-Aware XAI*,
  arXiv:2603.29915. **This is our primary baseline.**
- Aleatoric/epistemic decomposition to choose between attribution and
  counterfactual explanations — arXiv:2507.12913.
- Uncertainty attached to explanations — *Calibrated Explanations*
  (arXiv:2305.02305), *Fast Calibrated Explanations*, *Ensured*
  (arXiv:2410.05479).
- Conformal guarantees on counterfactuals — *CONFEX*, arXiv:2510.19754.
- Explanation-quality metrics and reliability metrics — *Sanity Checks for
  Explanation Uncertainty* (arXiv:2403.17212), ERI (arXiv:2602.05082).

**What this project adds:**

1. A **distribution-free guarantee on the delivered explanation**, with
   abstention, via conformal risk control. Existing work treats uncertainty
   as a heuristic; none controls the risk of what is shown.
2. A **learned per-instance explanation-quality predictor** built on the
   classifier's representation, tested directly against the uncertainty-only
   proxy that the 2026 baseline uses.
3. The **coverage–risk curve for explanations**: how many patients can be
   given an explanation at each guaranteed quality level.
4. Evaluation across three clinical datasets spanning a deliberate
   size/uncertainty gradient, two deep architectures, two UQ methods, and
   multiple seeds.

**Phrasing discipline.** Searches for "conformal risk control + explanation"
and "explanation quality prediction" returned no matching arXiv records under
several phrasings, which is weak evidence of absence. The paper will say
"to our knowledge" and report the search protocol. The words "first ever"
will not appear.

**Link to the base paper.** Jahn, Hühn & Reichert (*Artificial Intelligence
Review* 59:175, 2026) conclude that XAI method selection must be
context-aware and that existing selection frameworks stay conceptual because
the field lacks empirical evidence on contextual effects. This study
operationalises context at the instance level and makes selection
*verifiable* rather than advisory. It does **not** claim to validate their
user/task/domain findings — no users are studied.

---

## 4. Datasets

| Key | Source | n | Features | Target | Positive rate | Character |
|-----|--------|---|----------|--------|---------------|-----------|
| `brfss` | CDC BRFSS2015 Diabetes Health Indicators | 253,680 | 21 | diabetes/prediabetes | 13.9% | population survey, self-reported, coarse |
| `diabetes130` | UCI Diabetes 130-US Hospitals 1999–2008 | 101,766 | 194 after encoding | readmission < 30 days | 11.2% | hospital administrative records, categorical-heavy |
| `support2` | UCI SUPPORT2 | 9,105 | 75 after encoding | in-hospital death | 25.9% | ICU physiology, heavy missingness, small n |

**Why three, and why these three.** A single dataset cannot support any claim
about *when* a mechanism works, which is the whole point. The n gradient
254k → 102k → 9k is a designed epistemic-uncertainty gradient: the same
method faces genuinely different amounts of "the model does not know". The
feature types differ too (survey codes, administrative categoricals, ICU
measurements with informative missingness), which is what makes the
FT-Transformer worth including.

**Cleaning decisions that must appear in the paper.** Diabetes-130: ICD-9
codes collapsed into clinical chapters; categories with < 100 occurrences
merged; `weight` (97% missing) and `payer_code` dropped. SUPPORT2: the
outcome and everything computed after it removed — `death`, `d.time`,
`sfdm2`, `slos`, cost columns, and the study's own `surv2m`/`surv6m` and
physician `prg2m`/`prg6m` estimates, all of which leak the label; median
imputation with explicit missingness indicators.

**Splits.** 60% train / 10% validation / 15% **calibration** / 15% test,
stratified, per seed. The calibration split exists solely so the conformal
guarantee is valid; it is never trained on and never reported as a result.

---

## 5. Models

| Model | Role |
|-------|------|
| MLP 256-128-64, BatchNorm, dropout 0.3 | primary deep model, MC-Dropout capable |
| FT-Transformer (per-feature tokens, CLS, 3 blocks, 8 heads) | modern tabular deep architecture, matters most on the categorical-heavy dataset |
| Deep ensemble of K = 5 of either | the epistemic-uncertainty source |
| Logistic regression (balanced) | baseline |
| XGBoost / HistGradientBoosting | baseline |

The baselines answer the question the previous PRD left unanswered: **is the
deep model necessary?** The answer is reported honestly per dataset, in
whichever direction it falls. Note in advance: on SUPPORT2 the smoke run had
logistic regression at ROC-AUC 0.901 against the MLP ensemble at 0.897. If
that holds up, the paper says so — and the argument for deep learning rests
where it actually lies: on the ensemble's calibrated epistemic uncertainty
and on the representation the quality head needs, neither of which a linear
model provides.

---

## 6. Uncertainty quantification

Two samplers, so no result depends on one: **MC Dropout** (T = 50; dropout
layers reactivated, BatchNorm left in eval mode) and **deep ensembles**
(K = 5). Both produce a sample of probabilities per instance, decomposed as

```
total      H[E_θ p]                     overall uncertainty
aleatoric  E_θ H[p]                     irreducible data noise
epistemic  H[E p] − E H[p]  (mutual information)   what the model does not know
```

Predictive variance is still reported, to connect with the previous design.
**Calibration is mandatory**, not optional: ECE, Brier, reliability diagrams,
and temperature scaling fitted on validation. Uncertainty that is not
calibrated licenses no downstream claim.

---

## 7. Explanations

| Method | Kind | Notes |
|--------|------|-------|
| SHAP | attribution | GradientExplainer for deep models; permutation SHAP as model-agnostic fallback; 100-row training background |
| LIME | attribution | `discretize_continuous=False` so perturbation happens in the same standardised space the network sees |
| Integrated Gradients | attribution | baseline = training mean, 32 steps; native to the network and cheap enough for stability measurement |
| DiCE | counterfactual | 3 opposite-class counterfactuals, `method="random"`, immutable features locked (age, sex, education/race per dataset) |

300 test instances and 600 calibration instances per dataset per seed. The
calibration explanations are split in half: one half fits the quality head,
the other calibrates the conformal threshold. Failures (DiCE finding no
counterfactual) are recorded per instance and reported, never dropped.

---

## 8. Explanation risk — definition and declared choices

Per instance and method, measured properties:

- attributions: **infidelity** (Yeh et al., 2019), **instability** (local
  Lipschitz over an ε-ball, Alvarez-Melis & Jaakkola, 2018), **complexity**
  (attribution entropy)
- counterfactuals: **validity**, **proximity**, **sparsity**, plus an
  explicit check that immutable features were not altered

**The comparability problem, stated openly.** Infidelity is a squared
prediction error; invalidity is a 0/1 event. They cannot be averaged as they
stand. Each raw component is therefore mapped through its own empirical CDF,
estimated on the calibration split, giving a value in [0,1] meaning "worse
than this fraction of calibration explanations of the same kind". The
weighted combination of those normalised components is a **modelling choice**,
ablated in E-ablations, and the paper says so rather than presenting the risk
scale as a natural quantity.

---

## 9. The mechanism and the guarantee

```
        ┌── deep ensemble ──► p̂, u_epistemic, u_aleatoric
   x ───┤
        └── penultimate representation h(x)
                    │
                    ▼
        quality head  g(h(x), u_epi, u_alea, p̂) ──► r̂(x, m) for every method m
                    │
                    ▼
        m*(x) = argmin_m r̂(x, m)
        deliver m*(x)  if  r̂(x, m*) ≤ λ̂      else  ABSTAIN
                              ▲
                    conformal risk control on the calibration split
```

λ̂ is chosen as

```
λ̂ = sup { λ : R̂(λ) ≤ α − (B − α)/n },
R̂(λ) = (1/n) Σ_i  risk(x_i, m*(x_i)) · 1{ r̂(x_i, m*) ≤ λ },   B = 1
```

**What is guaranteed:** the population risk `E[risk · 1{deliver}] ≤ α` —
expected harm per patient, with abstention contributing zero. This is exactly
the quantity conformal risk control (Angelopoulos et al., 2022) covers.

**What is not guaranteed:** the selective risk `E[risk | deliver]`. It is a
ratio of random quantities, outside that theorem. It is reported empirically
with bootstrap intervals and never called guaranteed. Since abstaining from
everything attains population risk 0 trivially, the object of study is the
**coverage–risk curve**, not the risk alone.

---

## 10. Experiments

| ID | Question | Output |
|----|----------|--------|
| **E1** | Are the models good enough, and is deep learning needed? | `performance.csv`, `dl_verdict.csv`, `calibration.csv`, Fig 4 |
| **E2** | Does epistemic uncertainty rank explanation risk? (H5 replication) | `e2_uncertainty_vs_risk.csv`, Fig 3 |
| **E3** | Does the representation beat uncertainty as a risk predictor? (H1) | `h1_risk_prediction.csv` |
| **E4** | Does CSE beat fixed / random / uncertainty-gating at matched coverage? (H2) | `policies.csv`, `policy_tests.csv`, Fig 2 |
| **E5** | Is the α bound respected? (H3) | `coverage_risk_curve.csv`, Fig 1 |
| **E6** | Where does the system abstain? (H4) | `h4_abstention.csv`, Fig 5 |
| **Ablations** | representation vs uncertainty-only features; risk-weight sensitivity; MC Dropout vs ensemble; MLP vs FT-Transformer | `ablations.csv` |

Grid: 3 datasets × 2 architectures × 3 seeds, with both UQ methods on at
least one architecture. Every table carries `dataset`, `architecture`, `seed`.

---

## 11. Statistical plan

- Every reported number carries a **bootstrap 95% CI** (2,000 resamples).
- Policy comparisons: **paired Wilcoxon** on the same instances, with
  **rank-biserial** effect size; **Holm** correction across the whole family
  of comparisons reported together.
- Across datasets/seeds: **Friedman** omnibus plus **Nemenyi** critical
  difference.
- Correlations: **Spearman** with bootstrap CI (never Pearson — these
  quantities are not linear).
- Effect sizes (**Cliff's δ**) reported beside every p-value. At n = 15,000 a
  p-value alone means nothing.
- No assumption diagnostics are run; the tests chosen are distribution-free,
  which is the reason they were chosen.

---

## 12. Success criteria

The study is a success if it answers its questions, not if the answers are
favourable. Concretely:

- **Must hold** for the paper to make its main claim: H3 (the bound is
  respected) and H2 against at least the uncertainty-gating baseline on a
  majority of dataset × architecture cells.
- **Informative either way:** H1, H4, H5. If the quality head only matches
  scalar uncertainty, the paper reports that and the guarantee remains the
  contribution.
- **Fatal only if:** the risk scale turns out to be an artefact of the
  weighting (ablation shows rank reversal under every re-weighting), or the
  guarantee fails systematically, which would indicate an exchangeability
  violation worth its own analysis.

---

## 13. Code map and execution order

```
project/src/
  config.py              paths, datasets, all constants
  data.py                load → clean → 60/10/15/15 split → scale
  models.py              MLP, FT-Transformer, baselines, embed()
  training.py            class-weighted training, early stopping, ensembles
  uq_methods.py          MC Dropout, ensembles, decomposition, calibration
  evaluation.py          predictive metrics + deep-vs-shallow verdict
  xai.py                 SHAP / LIME / IG / DiCE behind one interface
  xai_metrics.py         measured properties → risk in [0,1]
  selective.py           quality head + conformal risk control + policies
  statistical_tests.py   bootstrap, Wilcoxon, Friedman/Nemenyi, Holm
  experiments.py         E1–E6 + ablations, per dataset × arch × seed
  figures.py             the five figures
  run_pipeline.py        CLI
  _v1_modules/           previous version, kept for reference, unused
```

```bash
python run_pipeline.py --describe                  # dataset table
python run_pipeline.py --smoke                     # ~2 min wiring check
python run_pipeline.py                             # full study
```

---

## 14. Reproducibility

Fixed seeds through `config.set_seeds()`; every split derived from the seed;
the scaler fitted on training rows only; ensembles seeded per member;
explanation instance samples drawn from a seeded generator. All result tables
written to `project/results/tables/` with the dataset/architecture/seed
columns needed to reproduce any single cell. Environment pinned by an
`environment.yml`/`requirements.txt` to be generated at submission.

---

## 15. Threats to validity

1. **Risk-scale construction.** The central quantity is partly defined by us.
   Mitigation: ECDF normalisation on held-out data, weight ablation, and
   reporting per-component results alongside the combined risk.
2. **No human evaluation.** Functional risk is not clinician-perceived
   usefulness. Mitigation: state it; do not claim user-level benefit.
3. **Exchangeability.** Conformal validity assumes calibration and test data
   are exchangeable. Mitigation: they come from the same random split, and
   the bound is checked empirically at every α.
4. **Deep model may not win on accuracy.** Mitigation: report it, and rest
   the deep-learning argument on uncertainty and representation, where it
   actually holds.
5. **Explainer implementation sensitivity.** SHAP variant, LIME sample count
   and DiCE method all affect measured quality. Mitigation: fixed and
   documented settings, with the sample count in the ablation grid.
6. **Multiplicity.** Many cells × many tests. Mitigation: Holm across every
   reported family; hypotheses fixed in advance in this document.

---

## 16. Claims that will not be made

- No "first ever" / "novel framework" language.
- No claim that the selective policy improves clinician decisions.
- No claim that the base paper's context-sensitivity findings are validated.
- No claim of a guarantee on selective risk.
- No claim that deep learning beats gradient boosting on accuracy unless the
  measured results say so.

---

## 17. Target venues

Primary: **Information Fusion**; **IEEE TNNLS** (methodological framing);
**IEEE JBHI** (healthcare framing). Fallback: **Knowledge-Based Systems**,
**Journal of Biomedical Informatics**. Quartile status to be verified against
JCR/Scopus directly before submission — the metrics collected during the
audit are OpenAlex citation proxies, not impact factors.

## 18. Sequence of work

1. Full grid run (3 datasets × 2 architectures × 3 seeds, DiCE included).
2. Inspect E5 first: if the bound fails, diagnose before writing anything.
3. Tables and figures; then the ablation grid.
4. Write methods from `project/src` (the code is the methods section).
5. Related work built on the full texts of the eight papers in §3, not their
   abstracts.
6. Internal red-team against §15 before submission.

---

## 19. What the verification run actually showed

One run: SUPPORT2, MLP, deep ensemble K = 3, seed 0, 600 calibration and 300
test instances, SHAP + Integrated Gradients only (LIME could not be
installed in the verification sandbox; DiCE disabled for speed). Not a
result to publish — a check that the machinery measures what it claims.

| Quantity | Value |
|---|---|
| Ensemble test ROC-AUC | 0.8974 |
| **H1** quality head, Spearman ρ vs measured risk | **0.694** [0.625, 0.752] |
| **H1** epistemic uncertainty, same | 0.281 [0.176, 0.379] |
| **H5** ρ(epistemic, risk), SHAP / IG | 0.252 / 0.284, both p < 1e-5 |
| **H2** population risk at α=0.10, coverage 0.323: CSE / uncertainty gating / oracle / best fixed | **0.080** / 0.150 / 0.053 / 0.500 |
| **H4** mean epistemic, abstained vs delivered | 0.00298 vs 0.00142, Cliff's δ = 0.24 |
| **H3** realised population risk at α = 0.05 / 0.10 / 0.15 / 0.20 / 0.30 | 0.031 / 0.080 / 0.150 / 0.225 / 0.306 |

Two things in that table change the design, and both are written into the
code rather than noted and forgotten.

**19.1 The bound must be checked across runs, not within one.** Conformal
risk control guarantees `E[L(λ̂)] ≤ α`, with the expectation over the
calibration draw *and* the test point. The correction term `(B−α)/n` is
about 0.002 at n = 300, so when the bound is tight a single run exceeds α
roughly half the time by construction — which is exactly the pattern above
(met at α = 0.05 and 0.10, exceeded by 0.00 / 0.03 / 0.01 at the larger
levels). Judging validity from one run misreads the theorem.
`selective.check_guarantee()` now aggregates realised risk across seeds,
datasets and architectures, compares the **mean** against α, and reports the
per-run exceedance rate alongside it. E5 is evaluated that way, and the
paper reports both numbers.

**19.2 The representation may contribute less than the framing assumed.**
The feature ablation gave Spearman ρ = 0.694 with the representation and
0.665 with the four uncertainty scalars alone, and delivered population risk
was marginally *better* without it (0.0758 vs 0.0797). On this dataset the
quality head's advantage over raw epistemic uncertainty (0.694 vs 0.281)
comes mostly from learning a non-linear map over the uncertainty summary,
not from reading the network's representation.

Consequences, applied now rather than at writing time:

- H1 is restated as **"a learned quality head predicts explanation risk
  better than raw epistemic uncertainty"**, which the data supports
  decisively. The representation's marginal value becomes a *sub-question*
  (H1b), answered by the ablation in whichever direction it falls.
- The deep-learning justification does not rest on H1b. It rests on the
  ensemble's decomposed, calibrated uncertainty and on the deep model's
  predictive performance — with the caveat, already visible on SUPPORT2,
  that logistic regression is competitive there (0.901 vs 0.897) and the
  paper must say so.
- SUPPORT2 is the smallest and lowest-dimensional dataset in the set. Whether
  the representation helps on Diabetes-130 (194 features) and BRFSS (254k
  rows) is an open empirical question the full grid answers; no claim either
  way until it is run.
