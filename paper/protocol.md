# Scientific analysis protocol

Version: 0.4.0-draft  
Date: 2026-07-20  
Machine-readable source: [`configs/protocol.json`](../configs/protocol.json)  
Status: complete scientific draft; **not frozen and not authorized for the
locked analysis run**.

## 1. Protocol decision in one paragraph

This is a retrospective **repeated nested internal cross-validation** study of
incremental value. It asks whether a pinned, frozen, mean-pooled Neuro-JEPA T1 MRI
representation improves over the same prespecified conventional variables across
the complete diagnostic-state and prognostic endpoint matrix. No historical seed
is reused. Three deterministically generated seeds define three locked repeats;
each repeat uses four participant-level outer folds and five inner folds. Thus
every participant is tested exactly once per repeat, while hyperparameters are
selected without its outer-test data. Results are paired on identical
participants and split variability is reported directly. The repeats are
correlated resamples of ADNI, not independent cohorts, and cannot establish
external performance or clinical utility.

## 2. Why this design is methodologically defensible

The online evidence review was performed on 2026-07-20. It was a targeted review
of primary methodological papers, consensus guidance, and major prospective ADNI
benchmarks, not a systematic review suitable for a PRISMA claim.

| Evidence | Relevant requirement | Protocol consequence |
|---|---|---|
| [CLAIM 2024](https://doi.org/10.1148/ryai.240300) | State partitions, patient-level disjointness, intended test size, preprocessing, model selection, uncertainty, robustness, and code availability; distinguish internal from external testing. | Participant-disjoint nested folds, explicit preprocessing and test-access rules, internal-testing terminology. |
| [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) | Report participants, predictors, outcomes, missing data, full model, discrimination, calibration, clinical utility, and uncertainty. | Full data dictionary, missingness rules, calibration for every probabilistic task, no clinical-utility claim. |
| [PROBAST+AI](https://doi.org/10.1136/bmj-2024-082505) | Risk of bias is assessed across participants/data sources, predictors, outcome, and analysis. | Prespecified eligibility, temporal predictor rules, identical paired cohorts, locked outer folds, and complete endpoint reporting. |
| [FUTURE-AI](https://doi.org/10.1136/bmj-2024-081554) | Evaluate robustness against patient, equipment, data, and centre variation and preserve traceability. | Field-strength, ADNI-phase, site, and subgroup audits plus hashes and immutable provenance; these acquisition variables are not predictors. |
| [Varma and Simon](https://doi.org/10.1186/1471-2105-7-91) and [Krstajic et al.](https://doi.org/10.1186/1756-0381-7-11) | Reusing tuning folds for performance estimation is optimistic, and performance can vary materially across splits. | Nested model selection and three locked repeats quantify rather than hide split instability. |
| [Baseline MRI predictors in ADNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC2764863/) | Hippocampal and amygdala volume and temporal cortical measures distinguish clinical groups and predict progression. | Amygdala/ICV is added beside hippocampus/ICV; unavailable entorhinal and temporal thickness measures are documented, not substituted post hoc. |
| [Longitudinal cerebral atrophy in ADNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC3590059/) | Whole-brain, hippocampal and ventricular change differ across CN, MCI and AD. | Cerebral cortex and ventricular ratios are retained in the prespecified morphometric baseline. |
| [AAPM Task Group 273](https://doi.org/10.1002/mp.16188) | Train, tuning, and test sets should be disjoint at patient level; the use case and population must be explicit. | No visit from one participant crosses a partition; target population is explicitly restricted to eligible ADNI participants. |
| [Cross-validation in medical imaging AI](https://doi.org/10.1148/ryai.220232) | Do not tune on the holdout; repeated examinations from one patient remain together. | Hyperparameters and thresholds use training/tuning data only; the test is opened once. |
| [BMJ prediction-model evaluation series](https://doi.org/10.1136/bmj-2023-074819) and [step-by-step guide](https://doi.org/10.1136/bmj-2023-078276) | Discrimination alone is insufficient; evaluate calibration and overall performance. Survival metrics must account for censoring. | MAE and calibration for regression; AUROC/AUPRC, Brier and calibration for binary outcomes; Uno C, IPCW Brier and fixed-time calibration for survival. |
| [Riley et al. on evaluation sample size](https://doi.org/10.1136/bmj-2023-074821) | Test size should be justified by precision, not only a generic rule. | The finite ADNI cohort is treated as a census; test counts and interval precision are disclosed, with an explicit calibration warning for conversion. |
| [Added-value guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC6460632/) and [critique of NRI/IDI](https://pubmed.ncbi.nlm.nih.gov/23553436/) | Compare a new marker added to a strong baseline and assess both calibration and discrimination; NRI/IDI can mislead. | Primary contrast is conventional versus conventional plus Neuro-JEPA; NRI and IDI are not used. |
| [TADPOLE](https://www.melba-journal.org/papers/2021%3A019.html) | Future diagnosis, ADAS13, and ventricles are difficult, endpoint-dependent tasks; simple baselines matter. | Carry-forward/persistence is mandatory and conclusions remain endpoint-specific. |
| [Neuro-JEPA](https://arxiv.org/abs/2606.14957) | ADNI AD-versus-control and MCI-to-AD were already downstream benchmarks; public data were split by patient. | The work is described as a post-publication incremental-value audit, never as model-naive external confirmation. |

## 3. Research question, estimand, and claim

The research question is:

> Among eligible longitudinal ADNI T1 MRI sessions, does adding the frozen
> Neuro-JEPA representation to prespecified conventional variables improve
> participant-level out-of-fold prediction across a locked diagnostic-state and
> prognostic endpoint matrix?

There is no privileged single primary endpoint. Each core task has one
prespecified comparison and one core performance metric. Every estimand has the
form:

```text
metric(conventional + Neuro-JEPA) - metric(conventional)
```

Negative values favor Neuro-JEPA for error metrics; positive values favor it for
AUROC, AUPRC, and Uno C. The eleven core endpoints jointly cover AD-vs-CN,
concurrent cognition, 12-month cognition and anatomy, 36-month MCI conversion,
and first-MCI survival. All belong to one Benjamini-Hochberg family. A claim for
one endpoint requires a favorable effect, a paired 95% interval excluding zero,
and q<0.05. There is no binary overall-study success rule and no endpoint may be
omitted because it is unfavorable.

The maximum permissible claim is endpoint-specific incremental value in internal
ADNI testing for the pinned representation. The study cannot establish external
generalization, causal benefit, fairness, deployment readiness, or clinical
utility.

## 4. Data source and eligible cohort

The study uses the versioned local ADNI-derived manifest, embedding index,
progression pairs, and horizon-matched pairs. ADNI source data are not modified or
redistributed.

A task row requires:

1. a participant and eligible anchor session in the canonical manifest;
2. an eligible T1 MRI;
3. a finite 768-dimensional pooled representation from the pinned checkpoint;
4. a valid task reference standard and temporal ordering; and
5. assignment to exactly one participant-level partition.

Technical failures, invalid time order, and missing outcomes are excluded with
reasons. No outcome is imputed. No participant, extreme target, residual, scan,
or prediction is removed because it harms performance.

All paired models for an endpoint use the exact same rows. The intersection is
formed before fitting, so a conventional-only model cannot silently benefit from
a larger cohort than an embedding model.

## 5. Neuro-JEPA representation

The encoder is `NYUMedML/Neuro-JEPA` at Hugging Face revision
`abdfc94aa3d2654a81a14afeb834dc04653b8976`. The checkpoint SHA-256 is
`decdf95a0d0d11e4b98ae07c5b180421c9ea076eeeeb818af0a75d88b0a113b0`.
The encoder is never updated with ADNI labels or outcomes.

The existing extracted pooled embeddings are reused without re-extraction. The
extraction runs in evaluation mode, contains no random augmentation, and is
deterministic for the pinned input and checkpoint. Participant folds therefore
change only downstream probe fitting, not the representation. Before the locked
analysis, the embedding index and every referenced pooled file are hashed; every
eligible image must map uniquely to one finite vector of exactly 768 dimensions.
The same immutable files are used in every inner and outer fold.

The deterministic input pipeline is:

1. load the image and ensure channel-first layout;
2. remove NaNs, orient to RAS, and resample to 1 mm isotropic spacing;
3. foreground crop at values greater than zero with margin 4;
4. edge pad/crop to `180×216×180`, resize to `100×120×100`, and center crop
   to `96×108×96`;
5. map the 0.5th–99.5th intensity percentiles to `[0,1]` with clipping;
6. cast to float32 and encode into 576 tokens of dimension 768; and
7. take the arithmetic token mean as the 768-dimensional visit feature.

ADNI was not reported as part of Neuro-JEPA's institutional pretraining corpus,
but it was an explicit downstream benchmark. The encoder may therefore embody
model-development choices influenced by ADNI benchmark performance even without
pretraining participant overlap.

## 6. Partition and test-access policy

No historical seed is used. The three locked repeat seeds are `1658731809`,
`1872921030`, and `621129424`. They were derived before model evaluation as the
first eight hexadecimal SHA-256 digits of the registered base label
`neurojepa-adni-audit|protocol-0.4.0|2026-07-20|outer-repeat-N`, reduced modulo
`2^31-1`.

Each repeat has four outer folds. In a given fold, 25% of participants are outer
test and 75% are outer development. Five-fold inner cross-validation inside the
outer-development data produces the equivalent of 60% fitting and 15% tuning per
inner fold. Hyperparameters are selected by mean inner-fold performance, after
which preprocessing and the model are refitted on the complete 75% outer
development and evaluated once on its 25% test fold.

All visits and anchors from a participant remain together. Each participant is
outer-test exactly once per repeat. The common outer folds are stratified by
baseline diagnosis and any diagnosis change; task outcome balancing is permitted
only within outer-development inner folds. All seeds, folds, endpoint definitions,
and reporting code are locked before any outer-test prediction is generated.

Every fold effect and pooled out-of-fold effect is retained. The paper reports
mean, standard deviation, and range across the three repeat estimates. No repeat
may be discarded or replaced for unfavorable performance.

## 7. Predictor groups

The common conventional inputs are:

- demographic: age at session, baseline sex, education, and APOE4 status;
- morphometric: bilateral hippocampus/ICV, amygdala/ICV, lateral
  ventricles/ICV, inferior lateral ventricles/ICV, and total cerebral
  cortex/ICV.

Task-specific current clinical variables are:

- concurrent cognition: current diagnosis, but never the concurrent target;
- future cognition: current diagnosis and current value of the predicted score;
- future morphometry: current diagnosis, MMSE, ADAS11, and ADAS13;
- first-MCI conversion and survival: current MMSE, ADAS11, and ADAS13.

The six fixed groups are morphometry only, clinical-demographic, conventional
full, Neuro-JEPA only, Neuro-JEPA plus morphometry, and Neuro-JEPA plus the full
conventional set. The core comparison for every endpoint is full conventional
versus full conventional plus Neuro-JEPA. Other groups are ablations.

ICV is used only to construct ratios and is not itself a predictor. Field
strength, site, and ADNI phase are excluded from every predictive feature group;
they are retained only to audit performance heterogeneity. Entorhinal cortex and
temporal cortical thickness are well-supported AD markers but are unavailable in
the canonical manifest, so they are not introduced through a different pipeline.
Amyloid, tau, p-tau, and other sparsely measured biomarkers may only be evaluated
in a clearly labeled exploratory available-case subset.

## 8. Missing predictors and preprocessing

All transformations are fitted without test data. Numeric predictors use the
training median and a missingness indicator when missingness occurs in training.
Categorical predictors use an explicit `__MISSING__` category and one-hot
encoding; unseen test levels are ignored. Numeric conventional predictors and
each embedding dimension are standardized using fitting-partition statistics.
Missing embeddings and outcomes are never imputed.

Missing counts and percentages are reported before imputation for every task and
partition. A structurally absent prespecified column causes an integrity failure;
it is not silently dropped.

## 9. Endpoints and anchors

### 9.1 Concurrent cognitive state

Every target-available eligible session is an observation. Training rows are
weighted by the inverse number of eligible rows for that participant within the
fitting partition. Test absolute errors are averaged within participant and then
across participants.

ADAS13, ADAS11, and MMSE are three core diagnostic-state endpoints under the same
rules. A locked sensitivity removes current diagnosis because ADNI diagnosis
partly reflects cognitive assessment.

### 9.2 AD-versus-CN sanity check

Use the first chronologically eligible MRI overall per participant and retain the
participant only when diagnosis at that anchor is AD or CN. Later AD/CN visits
after an MCI or other ineligible baseline are not substituted. This endpoint
verifies decodability and pipeline behavior. It is
the diagnostic-state domain, not evidence of prognosis or clinical utility.

### 9.3 Future cognition

For each eligible anchor, choose the nearest strictly later target-available
visit within `horizon ± 3` months. Fit change from current score and reconstruct
the future value. Evaluate future-value error and compare every model with
unchanged carry-forward. Twelve months is core; 24 and 36 months are
sensitivity analyses. Multiple anchors are allowed, with participant-equal
fitting and evaluation.

### 9.4 Future morphometry

At each visit, normalize bilateral hippocampal and lateral-ventricular volume by
the contemporaneous ICV. Fit change and evaluate the future ratio. Twelve months
is core; 24 and 36 months are sensitivity analyses. Historical six-month pairs
are explicitly forbidden from locked training, tuning, and testing.
Persistence is mandatory.

### 9.5 MCI conversion by 36 months

The anchor is the first eligible MCI MRI. A positive outcome is the first later AD
diagnosis within 36 months. A negative requires no AD anywhere in observed
follow-up and at least 36 months of follow-up. A first AD diagnosis after 36
months is not treated as stable and is excluded from the binary endpoint. Short
follow-up without AD is also excluded.

### 9.6 First-MCI time to AD

The anchor is the first eligible MCI MRI. The event is first later AD diagnosis.
Participants without later AD are right-censored at the last diagnosis-bearing
visit; zero or negative duration is invalid. Death is not available as a complete
competing endpoint in the canonical manifest and is stated as a limitation.

## 10. Models, tuning, and final refit

Continuous outcomes use Ridge regression with alpha
`[0.01, 0.1, 1, 10, 100, 1000]`. Binary outcomes use L2 logistic regression with
`C=[0.001, 0.01, 0.1, 1, 10, 100]`. Survival uses elastic-net Cox regression with
penalizer `[0.01, 0.1, 1, 10]` and l1 ratio `[0, 0.5, 1]`.

Selection metrics are mean inner-fold participant-equal MAE for regression,
AUROC for AD-versus-CN, AUPRC for conversion, and Uno C truncated at 36 months
for survival. Identical grids and inner folds are used across feature groups.
Ties choose stronger regularization. After selection, preprocessing and the model
are refitted on the complete outer-development fold and evaluated once on its
outer-test fold.

Every failed grid point, warning, and convergence status is retained. A group is
aborted if all prespecified settings fail. The grid cannot be expanded after test
inspection.

## 11. Performance measures

Regression reports participant-equal MAE and RMSE, R², Spearman correlation,
mean error, calibration intercept/slope, observed-versus-predicted calibration,
and residual plots.

Binary classification reports AUROC, AUPRC, Brier score, log loss, calibration
intercept/slope, a smooth calibration curve, sensitivity, specificity, balanced
accuracy, and MCC. Probability 0.5 is the fixed primary threshold. The
balanced-accuracy optimum selected on tuning is secondary and carried unchanged
to test. Expected calibration error is bin-dependent and descriptive only.

Survival reports Uno C at 36 months, cumulative/dynamic AUC and IPCW Brier at 12,
24, and 36 months, integrated Brier score from 12 to 36 months, and smooth
fixed-horizon calibration. Harrell C is descriptive. Proportional hazards are
checked in development data using scaled Schoenfeld residuals and global/feature
tests; violations are disclosed and fixed-horizon metrics receive greater
emphasis.

No decision-curve analysis is planned because no clinical action, deployment
population, or defensible risk threshold is prespecified.

## 12. Paired inference and multiplicity

Incremental effects are computed on matched outer-test predictions. Within each
repeat, all four outer-test folds form one complete out-of-fold prediction set.
Ten thousand paired bootstrap samples resample participants, retain all their
rows, recompute the pooled out-of-fold effect for each repeat, and average the
three correlated repeat effects. Binary resampling preserves outcome class and
survival resampling preserves event status. The fixed bootstrap seed is
`20260721`; 95% percentile intervals are reported.

Fold-specific effects plus the mean, standard deviation, and range across repeats
show split instability. The three repeats are not treated as independent sample
units and no t-test over twelve folds is used.

Benjamini-Hochberg control at q=0.05 is applied once across the eleven core
comparisons:

1. AD-vs-CN AUROC;
2. concurrent ADAS13, ADAS11, and MMSE MAE;
3. 12-month ADAS13, ADAS11, and MMSE MAE;
4. 12-month hippocampal/ICV and ventricular/ICV MAE;
5. 36-month conversion AUPRC; and
6. first-MCI Uno C at 36 months.

Effect sizes, intervals, q-values, and split dispersion are interpreted together.
Supporting metrics, sensitivity, ablation, and subgroup analyses are descriptive
even when a nominal p-value is available. NRI and IDI are not calculated.

## 13. Sample-size and precision audit

The project uses the eligible finite cohort rather than claiming a prospective
power calculation. Eligibility before the new split is approximately:

| Task | Eligible data |
|---|---:|
| AD vs CN | 1,278 participants; 399 AD |
| MCI conversion at 36 months | 605 participants; 262 converters |
| First-MCI survival | 1,030 participants; 344 events |
| Concurrent ADAS13 | 9,531 rows; 2,358 participants |
| Future ADAS13 at 12 months | 4,441 rows; 1,597 participants |
| Historical horizon-pair table | 14,877 rows; 1,872 participants, including forbidden six-month pairs |

Every participant receives one out-of-fold prediction per repeat, so pooled
repeat-level evaluation uses the complete eligible task cohort rather than a
single 25% holdout. Individual outer folds will nevertheless have substantially
fewer than 100 conversion events. Fold calibration is therefore descriptive;
pooled out-of-fold calibration retains participant-level uncertainty. Final
reporting includes fold counts and achieved interval widths rather than
retroactive power based on observed effects.

## 14. Robustness and subgroup analyses

Locked sensitivity analyses cover removal of concurrent diagnosis, first-session
instead of all-session cognition, complete-case conventional predictors,
24/36-month horizons, performance stratified by field strength/site/ADNI phase,
and Cox proportional-hazards violations. Acquisition variables never enter a
predictive model.

Subgroups are sex, age under 65/65–74/75+, APOE4 status, 1.5T versus 3T, and ADNI
phase. Regression subgroup metrics require at least 50 test participants. Binary
metrics require at least 50 participants and 20 members of each class; survival
requires at least 50 participants and 20 events. Otherwise only counts are shown.
These are robustness checks, not fairness claims; ADNI is not population
representative.

## 15. Reporting and interpretation rules

- Report every prespecified endpoint, feature group, failure, and null or negative
  result.
- Lead with paired effects, intervals, and fold/repeat dispersion, not a ranked
  model table.
- Call the evaluation repeated nested internal cross-validation, never external
  testing or independent cohort replication.
- Do not call overlapping random ADNI splits independent replications.
- Do not interpret a null interval as proof of equivalence.
- Do not infer longitudinal utility from AD-versus-CN separation.
- Do not infer clinical usefulness from discrimination without a deployment and
  impact study.
- Restrict conclusions to the pinned encoder, representation, data construction,
  and ADNI population.

The manuscript will include cohort flow, partition characteristics, missingness,
event/follow-up counts, exact software/checkpoint provenance, hyperparameter
histories, calibration, failure analysis, and CLAIM/TRIPOD+AI checklists.

## 16. Freeze, deviations, and analysis aborts

Freezing requires named author approval, a clean committed worktree, passing
tests, immutable source/checkpoint hashes, and completion of every implementation
gate in `configs/protocol.json`. The protocol generator must continue to reject a
draft protocol.

After freeze, the original file is not edited. Deviations are appended with time,
reason, author, and whether test outcomes were known. An outcome-aware deviation
makes the affected analysis exploratory.

The affected analysis is aborted if partitions overlap, paired cohorts differ,
a needed class/event is absent, an outer-test fold was used for selection, or
provenance hashes disagree. An unfavorable result or model failure never
authorizes discarding a fold or changing a seed.

## 17. Required implementation before freeze

The scientific protocol is complete, but the current code is not yet compliant.
The following must be implemented and tested before any locked analysis command:

1. immutable integrity manifest for the existing embedding index and files;
2. three repeats of participant-level 4-outer-fold × 5-inner-fold nested CV;
3. refit preprocessing and the selected model on each full outer-development fold;
4. participant-equal fitting weights for repeated-session endpoints;
5. identical paired cohorts, expanded morphometry in future-pair tables, and
   explicit exclusion of six-month morphometry;
6. Uno C, IPCW Brier, fixed-time survival calibration, and Cox diagnostics;
7. smooth regression and binary calibration artifacts;
8. pooled out-of-fold reporting, repeat synthesis, and the global core FDR family;
9. design-only cohort/precision audit separated from performance evaluation; and
10. a clean commit and author protocol sign-off.

Until these are complete, `results/smoke/` remains implementation evidence only
and no script should be launched as a paper experiment.
