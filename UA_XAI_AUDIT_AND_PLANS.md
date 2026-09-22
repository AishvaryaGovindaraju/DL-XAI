# UA-XAI: Novelty Audit and Three Routes to a Q1 Submission

Prepared 2026-09-22. Evidence base: the project folder (`UA_XAI_FINAL_PRD.md`,
`DL_XAI.ipynb`, the seven phase notebooks and their stored outputs, the new
`project/src` modules), the base paper, and a literature search over
OpenAlex and arXiv restricted to 2024-2026 (plus three earlier anchors that
turned out to matter).

Read Section 3 first. It is the part that decides whether the paper can be
submitted as currently designed, and the answer is no.

---

## 1. What the project currently is

**Research identity (from the PRD).** *Uncertainty-Conditioned Post-Hoc XAI
Selection for Deep Learning Healthcare Decision Support*. A DNN with Monte
Carlo Dropout produces per-patient epistemic uncertainty; patients are
stratified LOW / MEDIUM / HIGH; each stratum is assigned one explainer
(LOW→SHAP, MEDIUM→LIME, HIGH→DiCE); the claim is that this conditional
assignment beats using any single method everywhere.

**Base paper.** Jahn, Hühn & Reichert, *A review on empirical studies in
explainable artificial intelligence*, Artificial Intelligence Review (2026)
59:175, doi:10.1007/s10462-026-11595-6. A systematic review that screened 838
records and analysed **54** human-grounded empirical XAI studies. Its
conclusions: XAI effectiveness varies significantly across use cases, so
**context-aware method selection** is necessary; the empirical landscape is
imbalanced (some methods and user groups overrepresented, others overlooked);
and its stated research gap is that prior selection frameworks — Liao et al.
(2020), Hashemi et al. (2025) — "remain either largely conceptual or operate
at a high level of abstraction", so "the research field lacks empirical
insights on the contextual effects of XAI properties", which prevents those
frameworks from being validated.

**Measured state of the implementation** (read from the notebooks' stored
outputs, not re-run):

| Quantity | Value |
|---|---|
| DNN test ROC-AUC | 0.8301 |
| DNN test PR-AUC | 0.4264 |
| Recall / precision @ 0.5 | 0.664 / 0.358 |
| ROC-AUC from MC-Dropout mean | 0.8303 |
| MC-Dropout predictive variance, median | 0.00140 |
| σ², 99.9th percentile | 0.00861 |
| σ², maximum | 0.01751 |
| Strata (equal terciles) | 12,684 each |
| σ² range of the "HIGH" stratum | 0.00179 – 0.01751 |
| Instances explained by all three methods | 600 (200/stratum) |
| Implemented phases | 0,1,2,4,5,6A,6B,6C |
| Not implemented | 3 (baselines), 7 (XAI metrics), 8 (statistics) |

---

## 2. Audit findings

**F1 — The uncertainty strata are ranks, not levels. (Critical)**
Every patient in the test set has σ² < 0.018. The PRD's LOW threshold is
σ² < 0.05, so the PRD's own rule classifies 100% of the cohort as LOW; the
notebooks correctly detected this and fell back to equal terciles. The
consequence is that the "HIGH uncertainty" group is the top third of a range
that is two orders of magnitude narrower than the PRD anticipated. Any
sentence of the form "for high-uncertainty patients, counterfactuals are
preferable" is, in the current data, a statement about patients with
σ² ≈ 0.002–0.018, which no clinician would call uncertain. The paper cannot
make level-based claims on rank-based strata, and a reviewer will find this
in the first pass.

**F2 — The core novelty claim is already in the literature. (Critical)**
See Section 3. Two papers (2025, 2026) propose uncertainty-conditioned
explanation selection, with stronger motivation and broader experiments.

**F3 — The assignment policy is asserted, not derived. (Critical)**
There is no argument in the PRD for why *variance magnitude* should select
among explainer families, and there cannot easily be one, because SHAP, LIME
and DiCE do not answer the same question. SHAP gives an axiomatic
attribution, LIME gives a local surrogate, DiCE gives recourse. "DiCE is
better for this patient" is not commensurable with "SHAP is better for that
patient" unless you define a single risk scale across methods — which the
project does not do. The literature's better-posed version of this idea uses
the *aleatoric/epistemic decomposition* to choose between attribution and
recourse (arXiv 2507.12913), not variance bins.

**F4 — There is no answer to "why deep learning". (Critical)**
BRFSS2015 is 21 coarse self-reported survey features. The DNN scores
ROC-AUC 0.8301; published BRFSS results for logistic regression and gradient
boosting sit in the same 0.82–0.84 band. Phase 3 (LR, XGBoost) was never
implemented, so the project currently contains no evidence that the deep
model is needed at all. For a paper whose title says "Deep Learning", this
is the second reviewer kill-shot. Note that MC-Dropout averaging also gained
nothing here: 0.8303 vs 0.8301.

**F5 — The measurement layer does not exist yet. (Major)**
Phase 7 (fidelity, stability, sparsity, DiCE validity/proximity) and Phase 8
(Kruskal-Wallis, Dunn, effect sizes) are unimplemented. Every hypothesis in
the PRD (H1–H4) is therefore currently untestable. The project has produced
explanations but has not measured a single property of them.

**F6 — Rigour floor not met for any Q1 venue. (Major)**
One seed, one split, one dataset, one architecture, one UQ method, no
calibration analysis (no ECE, no Brier, no reliability diagram), no
aleatoric/epistemic decomposition, no multiplicity control, no distribution
shift, no ablation, 600 explained instances out of 38,052, and no
sensitivity analysis on the stratum boundaries that the whole design rests
on.

**F7 — The base paper's standard is not met. (Major)**
The base paper's finding is that context — *user, task, domain* — determines
what works, and its evidence standard is human-grounded study. The project
substitutes model-internal context (uncertainty) and functional proxies. That
substitution is defensible and can be stated as a deliberate scope choice,
but it must be argued explicitly, and the paper must not claim to have
validated the base paper's context-sensitivity thesis, because no user or
task was varied.

**F8 — Code/PRD consistency. (Minor, now documented)**
The PRD's fixed σ² thresholds, the `model.train()` MC-Dropout invariant in
`.agents/context/stack-and-rules.md`, and the code that actually produced
`stratified_samples.csv` disagree. The modular refactor implements what was
actually run (BatchNorm in eval, percentile strata) and documents both
divergences in `project/src/README.md`. Whichever plan you pick, the PRD has
to be re-frozen against the code, not the other way round.

---

## 3. Novelty audit, 2023-2026

### 3.1 What is already established

Searching arXiv and OpenAlex for 2024+ work on uncertainty and explanation
returns a populated, fast-moving area. The items that bear directly on this
project:

| Work | Date | What it establishes |
|---|---|---|
| **Uncertainty Gating for Cost-Aware XAI** (arXiv 2603.29915) | 2026-03 | Epistemic uncertainty as a low-cost proxy for explanation reliability; **routes samples to cheap or expensive XAI methods** based on expected explanation reliability; defers explanation for uncertain samples under budget. Four tabular datasets, five architectures, four XAI methods. Reports a strong negative correlation between epistemic uncertainty and explanation stability, and that uncertainty separates faithful from unfaithful explanations. |
| **Robust Explanations Through Uncertainty Decomposition** (arXiv 2507.12913) | 2025-07 | Uses the aleatoric/epistemic split to **choose the explanation type**: epistemic uncertainty as a rejection criterion for unreliable explanations, aleatoric uncertainty to decide between feature-importance and counterfactual explanations. |
| **Calibrated Explanations** (arXiv 2305.02305) and **Fast Calibrated Explanations** | 2023-05 / 2024-10 | Venn-Abers-based feature-importance explanations that carry uncertainty on the feature weights *and* the probability, model-agnostic, including counterfactuals with embedded uncertainty. |
| **Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions** (arXiv 2410.05479) | 2024-10 | Explanation types targeted at epistemic uncertainty, plus an "ensured ranking" metric trading off uncertainty, probability and alternatives. |
| **CONFEX** (arXiv 2510.19754) | 2025-10 | Uncertainty-aware counterfactuals via conformal prediction + MILP, with local coverage guarantees; explicitly avoids high-uncertainty regions. |
| **Sanity Checks for Explanation Uncertainty** (arXiv 2403.17212) | 2024-03 | Weight- and data-randomisation sanity tests for explanation-uncertainty method combinations. |
| **Reliable Explanations or Random Noise? (ERI)** (arXiv 2602.05082) | 2026-02 | A reliability metric family for explanations under input perturbation, feature redundancy and model updates. |
| **Selective Explanations** (arXiv 2405.19562) | 2024-05 | Detects when a cheap explainer produces low-quality explanations and repairs them. |

### 3.2 Consequence for the current claim

The project's contribution as written — *use MC-Dropout epistemic uncertainty
to select which post-hoc explainer to apply per instance, in healthcare* — is
**preempted**. arXiv 2603.29915 does the routing, with more datasets, more
architectures, and the correlation result the project's H4 was going to
report. arXiv 2507.12913 does the type-selection with a better-motivated
signal. Submitting the current design to a Q1 venue in late 2026 invites a
rejection that cites both.

This does not mean the project is dead. It means the contribution has to move
to a part of the problem that those papers left open.

### 3.3 What is genuinely open

Searches that returned **no arXiv matches** under several phrasings
(`conformal risk control` + `explanation`; `explanation quality prediction`;
`explanation stability` + `training objective`; uncertainty-weighted
attribution-stability regularisation):

1. **No distribution-free guarantee on the delivered explanation.** Existing
   work uses uncertainty as a *heuristic* proxy for explanation reliability.
   Nobody controls the risk: "with probability ≥ 1-α, the explanation this
   system shows a clinician has fidelity above τ, otherwise it shows
   nothing." Conformal risk control is standard machinery and has not been
   applied to explanation delivery.
2. **No learned per-instance explanation-quality predictor.** Uncertainty is
   used as the proxy; nobody has asked whether the network's own
   representation predicts explanation quality *better* than its uncertainty
   does. This is a directly testable improvement over a strong 2026 baseline.
3. **No training-time coupling** between uncertainty and explanation
   stability. Attribution-stability regularisation exists (RoSHAP 2026,
   SHAP-guided regularisation 2025, robust attribution regularisation 2019);
   an uncertainty-weighted version, and the claim that fixing this in
   training removes the need for routing, appears unoccupied.
4. **No evaluation under real temporal shift.** All of the above is evaluated
   i.i.d. BRFSS has annual waves (2011-2023) — genuine covariate shift, free,
   and exactly the regime where epistemic uncertainty is supposed to matter.

Caveat on all four: absence of an arXiv/OpenAlex match under my phrasings is
weak evidence of absence. None of the plans below should use the words "first
ever"; the defensible phrasing is "to our knowledge, no prior work has ...",
with the search protocol reported in the paper.

### 3.4 Is the uncertainty-conditioned selection mechanism scientifically justified?

**No, not in its current form.** Three specific defects, each fixable:

1. *Wrong quantity.* Total predictive variance from MC Dropout conflates
   aleatoric and epistemic uncertainty. Only the epistemic part carries the
   "the model does not know this region" meaning the framework needs. Fix:
   decompose (ensemble/MC-Dropout mutual-information decomposition), and use
   the epistemic component.
2. *Wrong decision structure.* Binning a continuous quantity into three
   ordinal labels and hard-mapping each to a method throws away the
   continuum and cannot express abstention. Fix: define a per-method
   explanation-risk on a common scale and select the argmin, with an
   abstention option — a decision rule, not a lookup table.
3. *No commensurability.* Attribution quality and recourse quality are not
   the same units. Fix: define the risk as *task-relative* (e.g. normalised
   infidelity + instability for attributions, 1-validity + normalised
   proximity for counterfactuals), calibrate each to a common quantile scale
   on held-out data, and state plainly in the paper that this normalisation
   is a modelling choice with its own sensitivity analysis.

Fixing 1-3 turns "we asserted a policy" into "we derived a decision rule",
which is the difference between the current draft and Plan 2 below.

---

## 4. The common floor (required by all three plans)

Independent of which plan you choose, these must be done, because they are
what the reviewers check first:

- **Baselines implemented and reported**: logistic regression, XGBoost, and
  at least one modern tabular DL model (FT-Transformer, or TabPFN-v2
  inference) beside the MLP. If the DNN does not win, say so and re-motivate
  the deep model on what it *does* provide (calibrated epistemic uncertainty
  and a representation to condition on) rather than on accuracy.
- **Seeds and splits**: ≥ 5 seeds, report mean ± 95% CI everywhere; no
  single-split numbers.
- **Calibration**: ECE, Brier, reliability diagrams for every UQ method.
  Uncertainty that is not calibrated cannot license any downstream claim.
- **Uncertainty decomposition**: aleatoric vs epistemic, reported separately.
- **≥ 2 UQ methods**: MC Dropout plus deep ensemble (and ideally
  Laplace/SWAG), so the conclusions are not an artefact of dropout.
- **XAI quality metrics** (Phase 7, currently missing): infidelity,
  local-Lipschitz stability, sparsity/complexity for SHAP/LIME; validity,
  proximity, sparsity, plausibility for DiCE; plus ERI-style reliability.
- **Statistics** (Phase 8, currently missing): mixed-effects or
  Friedman/Nemenyi across datasets and seeds, Holm-corrected, effect sizes
  with CIs, exact p-values.
- **Explain more than 600 instances** — with the metrics implemented, 2,000+
  per stratum per dataset is cheap for SHAP/LIME; keep the subsample only
  for DiCE if runtime forces it, and report the sampling rule.
- **Reproducibility package**: seeds, environment lock, the `project/src`
  pipeline, and a scripted one-command reproduction.
- **Re-freeze the PRD** against the code (F8).

Rough cost of the floor alone: 4-6 weeks. None of it is novel; all of it is
non-negotiable.

---

## 5. Plan 1 — "Evidence": the validation study the field is missing

**Positioning.** The base paper says selection frameworks are theoretical and
unvalidated. The 2025-2026 papers propose uncertainty-based selection but each
validates on its own narrow setup. You deliver the **first systematic,
pre-registered, multi-dataset stress test of the uncertainty→explanation-
reliability hypothesis**, reporting where it holds, where it breaks, and by
how much — including negative results.

**Claim.** Not "we invented a mechanism" but "we established the conditions
under which the mechanism proposed in the literature holds, and produced the
benchmark and guidance table for it."

**Design.**
- Datasets: BRFSS (2015 + 2 further waves for shift), UCI Diabetes 130-US
  hospital readmission, and one richer clinical tabular set (eICU- or
  MIMIC-derived cohort if you can get credentialed; otherwise Heart Failure
  clinical records + Framingham).
- Models: MLP + FT-Transformer + LR/XGBoost baselines.
- UQ: MC Dropout, deep ensemble, Laplace approximation, conformal.
- XAI: SHAP (Deep/Gradient + Kernel), LIME, DiCE, Integrated Gradients.
- Outcome: the full CO-12-style property panel per instance.
- Analysis: does epistemic uncertainty predict explanation
  fidelity/stability, per dataset × model × UQ × XAI cell? Mixed-effects
  model with random effects for dataset and seed. Report the correlation the
  2026 paper reported and test whether it replicates.
- Deliverable artefacts: a public benchmark, a decision guidance table, and
  a replication verdict.

**New modules**: `xai_metrics.py`, `uq_methods.py`, `benchmark_runner.py`,
`statistics.py`, `figures.py`.

**Target venues**: Artificial Intelligence Review, Information Fusion,
Journal of Biomedical Informatics.

**Timeline**: 10-12 weeks after the floor.

**Risks**: "no new method" is the standard rejection reason; mitigated by
framing as benchmark + replication, which AI Review and Information Fusion
do publish, but it must be *large* to carry a Q1 slot.

**Acceptance value: 6 / 10.**

---

## 6. Plan 2 — "Guarantee": Conformal Selective Explanation (recommended)

**Positioning.** Keep the project identity (uncertainty + DL + XAI +
healthcare), move the contribution from *heuristic routing* to *risk-controlled
explanation delivery with a formal guarantee*. This is the gap in §3.3(1-2),
it has a strong, citable 2026 baseline to beat, and the machinery is standard
enough to be feasible.

**The mechanism (what the paper actually contributes).**

1. **Uncertainty**: train the DNN, obtain epistemic and aleatoric components
   by decomposition over a deep ensemble of MC-Dropout networks.
2. **Explanation risk**: for each candidate explainer m and instance x define
   a scalar risk r(x,m) ∈ [0,1] — attributions: normalised infidelity +
   local-Lipschitz instability; counterfactuals: 1-validity + normalised
   proximity + sparsity penalty. Calibrate each method's raw score to a
   common quantile scale on a held-out calibration split so the risks are
   comparable. State the normalisation as a modelling choice and run a
   sensitivity analysis over it.
3. **Learned quality predictor (the deep-learning contribution)**: train a
   small head g(h(x), u_epi(x), u_alea(x)) → r̂(x,m) on the classifier's
   penultimate representation to predict each method's explanation risk
   *before* paying to compute the explanation. Test the central question:
   does the representation predict explanation quality better than
   uncertainty alone? (Uncertainty gating, arXiv 2603.29915, is exactly the
   uncertainty-only baseline.)
4. **Risk control**: select the argmin-risk method, but only deliver it if
   the predicted risk clears a threshold λ chosen by **conformal risk
   control** so that the expected risk of delivered explanations is ≤ α;
   otherwise **abstain** and tell the user no reliable explanation is
   available. The guarantee is finite-sample and distribution-free.
5. **The headline figure**: an explanation coverage-reliability curve —
   fraction of patients who receive an explanation vs the guaranteed
   fidelity bound, per method and for the selective policy. That curve is a
   new object in this literature and is directly meaningful clinically.
6. **Shift**: repeat under BRFSS temporal shift (train on one wave, test on
   later waves) and report guarantee violation, which is where conformal
   assumptions genuinely bite — reporting that honestly is itself a
   contribution.

**Baselines to beat**: fixed-SHAP, fixed-LIME, fixed-DiCE, random routing,
uncertainty-gating (2026), oracle routing (upper bound).

**Hypotheses** (pre-register these):
- H1: the learned risk predictor outranks epistemic uncertainty as a
  predictor of explanation risk (Spearman, per dataset, Holm-corrected).
- H2: at matched coverage, selective explanation attains lower mean delivered
  risk than any fixed method and than uncertainty gating.
- H3: empirical risk of delivered explanations respects the α bound i.i.d.,
  and the violation under temporal shift is bounded/measurable.
- H4: abstention concentrates in the high-epistemic-uncertainty region
  (this is the salvageable version of the current project's H3/H4).

**New modules**: `uq_methods.py` (ensembles + decomposition),
`xai_metrics.py`, `explanation_risk.py`, `quality_model.py`,
`risk_control.py`, `shift_eval.py`, `statistics.py`, `figures.py`. The
existing `model.py`, `training.py`, `mc_dropout.py`, `explain_*.py` are
reused as-is.

**Target venues**: Information Fusion; IEEE TNNLS; IEEE JBHI (the healthcare
framing); Knowledge-Based Systems as fallback.

**Timeline**: 14-18 weeks after the floor. Compute is modest (tabular, CPU +
your RTX 3050 is ample).

**Risks**: (a) g may only match uncertainty rather than beat it — the paper
survives, because the guarantee and the coverage-reliability curve stand, but
the story weakens; (b) the risk-normalisation choice is attackable, hence the
mandatory sensitivity analysis; (c) conformal exchangeability is violated
under shift — report, do not hide.

**Why this is the recommendation.** It is the only one of the three where the
central claim is *provable* rather than empirical, it keeps deep learning
structurally necessary (representation-conditioned prediction + uncertainty
decomposition), it keeps the healthcare identity, it has a concrete recent
baseline to beat, and it converts the project's existing assets (trained DNN,
MC Dropout, three explainers, 600-instance harness) into components rather
than discarding them.

**Acceptance value: 7.5 / 10.**

---

## 7. Plan 3 — "Mechanism-in-training": uncertainty-weighted explanation-stable learning

**Positioning.** Argue that post-hoc routing treats a symptom. If explanations
are unstable where the model is epistemically uncertain, the principled fix is
in the training objective, not in the selection layer.

**The mechanism.** Add to the classification loss a regulariser that
penalises attribution variability across MC-Dropout samples, weighted by
per-instance epistemic uncertainty; derive its relation to the local
Lipschitz constant of the attribution map; show that the resulting network
has measurably more stable and more faithful explanations at equal ROC-AUC,
and that the uncertainty→instability correlation the 2026 paper relies on is
*flattened* — i.e. routing becomes unnecessary.

**Design.** Same datasets/baselines as Plan 2, plus ablations over the
regularisation weight, the attribution estimator used inside the loss
(Integrated Gradients is differentiable and cheap; SHAP is not), and the
accuracy-stability Pareto front. The headline figure is that Pareto front
against unregularised training and against RoSHAP-style robust attribution
regularisation.

**Target venues**: IEEE TNNLS, Neural Networks, Information Fusion.

**Timeline**: 18-24 weeks after the floor; heaviest compute of the three
(attribution inside the training loop).

**Risks**: highest. The neighbourhood is populated (robust attribution
regularisation 2019, SHAP-guided regularisation 2025, RoSHAP 2026), so the
novelty rests on the uncertainty weighting plus the "routing becomes
unnecessary" claim; the regulariser may cost AUC, which turns the paper into
a negative result; and the theory has to be correct, not decorative. Highest
ceiling (a clean result here is an 8+ paper) and the highest chance of
producing 5 months of work with no publishable claim.

**Acceptance value: 6.5 / 10** (expected value; ≈8/10 if the Pareto
improvement is clean, ≈4/10 if it is not).

---

## 8. Decision table

| | Plan 1 Evidence | **Plan 2 Guarantee** | Plan 3 Training |
|---|---|---|---|
| Novelty type | benchmark + replication | mechanism + formal guarantee | training-time method |
| Preemption risk | low (nobody has done it at scale) | low-moderate | moderate-high |
| Is the claim provable? | no, empirical only | **yes (risk control)** | partly (theory + empirics) |
| DL structurally necessary? | weak | **yes** | **yes** |
| Base-paper continuity | **strongest** | strong | weakest |
| Keeps healthcare identity | yes | **yes** | yes |
| Reuses existing code | most | most | some |
| Effort after the floor | 10-12 wk | 14-18 wk | 18-24 wk |
| Compute | low | low-moderate | moderate-high |
| Failure mode | "no new method" rejection | predictor only matches baseline | regulariser costs accuracy |
| **Acceptance value** | **6/10** | **7.5/10** | **6.5/10** |

What the scores mean: my judgement of the probability that the completed work,
executed competently and written well, is accepted at one of the named Q1
venues within two submission attempts. They are calibrated against the audit
findings above, not against enthusiasm. The current design, unchanged, scores
**2/10** — not because the work done so far is bad, but because F2 (the claim
exists in the literature) and F4 (no case for deep learning) are each
independently sufficient for rejection.

**No plan guarantees acceptance.** Any promise of guaranteed publication in a
Q1 venue would be dishonest; acceptance depends on reviewer draw, venue load
and the quality of the writing as much as on the design. What these plans can
do is remove the reasons a reviewer would *have* to reject.

---

## 9. Venue notes

Proxy metrics pulled from OpenAlex (2-year mean citedness; **not** the JCR
Impact Factor, and **not** a quartile). I cannot verify current JCR/Scopus
quartiles with the tools available here — check the JCR or Scopus source list
directly before committing, and check each journal's recent issues for scope
fit.

| Journal | ISSN | 2-yr mean citedness | h-index | APC (USD) |
|---|---|---|---|---|
| Artificial Intelligence Review | 0269-2821 | 31.05 | 200 | 3190 |
| npj Digital Medicine | 2398-6352 | 13.48 | 168 | 4090 |
| Information Fusion | 1566-2535 | 12.53 | 219 | 4740 |
| Medical Image Analysis | 1361-8415 | 11.09 | 219 | 4170 |
| IEEE TNNLS | 2162-237X | 9.28 | 244 | 2645 |
| Computers in Biology and Medicine | 0010-4825 | 8.47 | 187 | 3080 |
| Expert Systems with Applications | 0957-4174 | 7.58 | 349 | 3500 |
| Journal of Biomedical Informatics | 1532-0464 | 6.78 | 171 | 3410 |
| Knowledge-Based Systems | 0950-7051 | 6.53 | 222 | 3430 |
| IEEE JBHI | 2168-2194 | 5.76 | 166 | 2645 |
| Neural Networks | 0893-6080 | 5.74 | 248 | 3110 |
| IEEE Trans. on Artificial Intelligence | 2691-4581 | 4.94 | 61 | 2645 |

For an IEEE target specifically: **IEEE JBHI** for the healthcare framing,
**IEEE TNNLS** for the methodological framing. Both are plausible for Plan 2.

---

## 10. What to do next

1. Pick a plan.
2. Whatever the pick, start the common floor (§4) immediately — it is shared
   by all three and it is where the current project is weakest.
3. In parallel, retrieve and read in full: arXiv 2603.29915, 2507.12913,
   2305.02305, 2510.19754, 2403.17212, 2602.05082. The related-work section
   must engage with these, not cite them in passing; the audit above is based
   on abstracts, and the paper needs the full texts.
4. Re-freeze the PRD against the chosen plan and against the code, and drop
   the claims that F1-F4 invalidate.

---

## References retrieved for this audit

- Jahn, T., Hühn, P. & Reichert, M. A review on empirical studies in
  explainable artificial intelligence. *Artificial Intelligence Review*
  (2026) 59:175. doi:10.1007/s10462-026-11595-6 (base paper; 54 studies
  analysed, 838 screened)
- Uncertainty Gating for Cost-Aware Explainable Artificial Intelligence.
  arXiv:2603.29915, 2026-03-31.
- Robust Explanations Through Uncertainty Decomposition: A Path to
  Trustworthier AI. arXiv:2507.12913, 2025-07-17.
- CONFEX: Uncertainty-Aware Counterfactual Explanations with Conformal
  Guarantees. arXiv:2510.19754, 2025-10-22.
- Ensured: Explanations for Decreasing the Epistemic Uncertainty in
  Predictions. arXiv:2410.05479, 2024-10-07.
- Sanity Checks for Explanation Uncertainty. arXiv:2403.17212, 2024-03-25.
- Calibrated Explanations: with Uncertainty Information and Counterfactuals.
  arXiv:2305.02305, 2023-05-03. (plus Fast Calibrated Explanations, 2024-10)
- Reliable Explanations or Random Noise? A Reliability Metric for XAI.
  arXiv:2602.05082, 2026-02-04.
- Selective Explanations. arXiv:2405.19562, 2024-05-29.
- Related, found but not central: "Why Should You Trust My Explanation?
  Understanding Uncertainty in LIME Explanations" (2019); "Investigating the
  Impact of Model Instability on Explanations and Uncertainty" (2024-02);
  "Machine learning with a reject option: a survey" (2024); RoSHAP (2026-05);
  SHAP-Guided Regularization (2025-07).
