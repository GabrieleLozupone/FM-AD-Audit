# Do Frozen Neuroimaging Foundation Representations Add Diagnostic and Prognostic Value in Alzheimer Disease? A Longitudinal ADNI Audit

Manuscript status: initial journal-specific scaffold. Numerical Results remain
provisional until the locked repeated nested cross-validation and final paired
analysis are complete.

## Abstract

**Background:** Neuroimaging foundation models provide transferable magnetic
resonance imaging representations, but their incremental value beyond established
anatomical and clinical markers in Alzheimer disease remains unclear.
**Methods:** We designed a longitudinal audit of frozen Neuro-JEPA representations
extracted from T1-weighted MRI in the Alzheimer's Disease Neuroimaging Initiative.
Under locked repeated participant-level nested cross-validation, regularized linear
probes compare morphometric, demographic and clinical variables, frozen
Neuro-JEPA features, and their combination across diagnosis, concurrent cognitive
severity, future cognition and anatomy, MCI conversion, and censored time to
Alzheimer dementia. Incremental effects are estimated using paired
participant-level bootstrap intervals with fold and repeat dispersion. All
diagnostic-state and prognostic tasks are prespecified core endpoints.
**Results:** [To be generated exclusively from the frozen nested-CV protocol.]
Historical analyses motivate the hypothesis that the clearest
incremental signal will occur for concurrent cognitive severity, with smaller and
endpoint-dependent effects for short-horizon prognosis.
**Conclusions:** This study tests whether a frozen neuroimaging foundation
representation contributes information beyond conventional Alzheimer disease
variables, and identifies the diagnostic and prognostic settings in which such
representations are complementary, redundant, or insufficient.

Word count: approximately 160 words after removal of this note.

## Keywords

Alzheimer disease; neuroimaging foundation models; representation learning;
magnetic resonance imaging; longitudinal prediction; brain informatics; ADNI

## Introduction

Foundation models are increasingly used to derive general-purpose representations
from brain imaging. Their evaluation, however, is often dominated by static
classification tasks in which a lightweight prediction head is trained to
separate diagnostic groups. Such experiments establish that disease-related
information can be decoded from an embedding, but they do not establish whether
the embedding adds information beyond established anatomical or clinical markers.
This distinction is particularly important in Alzheimer disease, where age,
genetic risk, cognitive status, hippocampal atrophy, and ventricular enlargement
already provide strong and clinically interpretable signals.

This comparison also has a practical resource dimension. Once validated
morphometric and clinical variables have been derived, regularized tabular models
are inexpensive to train, tune, and audit, and their predictors have direct
clinical or anatomical meanings. In contrast, repeatedly fine-tuning large
volumetric encoders requires specialized accelerators and introduces additional
optimization and reproducibility burden. We therefore treat computational
parsimony as part of the evaluation question, while avoiding an environmental
impact claim unless energy use and carbon intensity are measured explicitly.

A representation intended for brain-health research should therefore be assessed
incrementally and longitudinally. First, its benefit should be measured against
matched conventional baselines rather than against imaging-free or weak
comparators. Second, visits from the same participant must not cross data splits.
Third, uncertainty should reflect repeated observations within participants.
Finally, evaluation should include difficult outcomes such as future cognitive
change, anatomical progression, conversion from mild cognitive impairment, and
censored time to dementia. Strong diagnostic separation at one visit need not
translate into useful prognostic information.

In this study, we audit frozen Neuro-JEPA representations extracted from
longitudinal T1-weighted MRI in ADNI. We compare explicitly separated
morphometric, demographic and clinical feature groups with Neuro-JEPA-only and
combined models using regularized linear probes. The protocol uses three locked
repeats of participant-level nested cross-validation with no historical seed,
and incremental effects are quantified by paired participant-level bootstrap
intervals plus fold/repeat dispersion. Positive, null,
and negative findings are retained to characterize where foundation
representations complement conventional Alzheimer disease variables and where
they do not. Because ADNI was already a Neuro-JEPA downstream benchmark, this is
a post-publication audit rather than a model-naive external test.

Our contributions are: (1) an incremental evaluation framework that separates
MRI morphometry from demographic and clinical state; (2) a multi-endpoint audit
spanning concurrent state, short- and longer-horizon prognosis, conversion, and
survival; (3) participant-aware inference across repeated nested folds; and (4) an
explicit analysis of the boundary between decodable disease information
and added prognostic value.

## Related work

### Neuroimaging foundation and self-supervised models

Large-scale self-supervised pretraining has produced transferable representations
for natural images and video. Joint-embedding predictive architectures are a
prominent recent branch of this literature: I-JEPA predicts target-block
representations from surrounding image context without reconstructing pixels,
while V-JEPA 2 extends latent prediction to video understanding, anticipation,
and planning. These results make JEPA a leading contemporary pretraining
paradigm, but not a universally superior method across datasets and tasks
([I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html);
[V-JEPA 2](https://arxiv.org/abs/2506.09985)).

Medical foundation models have shown particularly visible progress in
segmentation, including promptable volumetric models such as SAM-Med3D. More
recent brain MRI models have broadened evaluation to diagnosis, regression,
survival, and other image-level tasks. BrainIAC evaluates both fine-tuning and
frozen linear probing across heterogeneous brain MRI endpoints; BrainDINO reports
frozen transfer to ADNI disease staging; and AnatCL evaluates anatomically guided
pretraining across Alzheimer classification and clinical-assessment tasks
([SAM-Med3D](https://arxiv.org/abs/2310.15161);
[BrainIAC](https://www.nature.com/articles/s41593-026-02202-6);
[BrainDINO](https://arxiv.org/abs/2604.27277);
[AnatCL](https://doi.org/10.1016/j.patrec.2025.11.028)). These studies establish
that pretrained representations can be decoded or adapted for medical endpoints,
but they primarily compare encoders or training strategies rather than testing
incremental value over a complete conventional Alzheimer disease model.

Neuro-JEPA is an especially informative stress test of this unresolved question.
It applies latent predictive pretraining with a sparse mixture-of-experts
architecture to approximately 1.55 million T1-weighted, T2-weighted, and FLAIR
scans, and reports broad downstream gains over other neuroimaging foundation
models and a task-specific convolutional baseline. Its released benchmark already
includes ADNI Alzheimer classification, amyloid prediction, and MCI-to-Alzheimer
time-to-event evaluation. Those experiments predominantly assess task-specific
fine-tuning and relative image-encoder performance; they do not determine whether
a frozen Neuro-JEPA embedding adds information after explicit morphometric,
demographic, and current clinical state is known
([Neuro-JEPA](https://arxiv.org/abs/2606.14957)). We therefore select Neuro-JEPA
not because JEPA can be declared universally state of the art, but because it is
a recent, large-scale, neuroimaging-specific implementation of a leading
predictive-representation paradigm and provides a stringent contemporary test of
the incremental-value hypothesis.

### Alzheimer disease diagnosis and prognosis from MRI

MRI-based Alzheimer modeling has long covered diagnosis, conversion, cognition,
and anatomical progression. The TADPOLE challenge prospectively compared 92
algorithms for future diagnosis, ADAS13, and ventricular volume and found that no
single approach was best across outcomes; ADAS13 forecasting was particularly
difficult. This history makes simple carry-forward, current-state, and
conventional multimodal predictors necessary comparators rather than optional
baselines
([TADPOLE](https://www.melba-journal.org/papers/2021%3A019.html)).

Recent work has moved closer to the present study without answering the same
question. Ma and colleagues used baseline MRI and demographics to predict
diagnosis, concurrent cognition, and cognitive trajectories up to 36 months,
comparing a three-dimensional U-Net trained from scratch with a fine-tuned
MedicalNet model and testing externally. Their objective was MRI-based cognitive
estimation; current cognition, current diagnosis, and regional biomarkers were
deliberately excluded from the inputs. Demir and colleagues instead evaluated
frozen registration and segmentation foundation-model features with linear probes
on ADNI diagnosis, pairwise diagnostic worsening, and 36-month progression. Their
future task used multiple earlier MRI visits and did not quantify incremental
value beyond conventional clinical and morphometric state
([single-MRI Alzheimer outcome prediction](https://doi.org/10.1038/s43587-026-01121-2);
[out-of-the-box progression features](https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/papers/Demir_How_Useful_Are_Vision_Foundation_Model_Features_for_Out-of-the-Box_Disease_CVPRW_2026_paper.pdf)).

Conversely, recent clinical prediction studies have explicitly evaluated routine
clinical and MRI-derived variables for 12-month cognitive decline, including
calibration and decision analysis, but have not evaluated raw-MRI foundation
representations
([routine clinical and MRI prediction](https://doi.org/10.1038/s41598-026-43321-1)).
The remaining gap is therefore not whether MRI or pretrained features can predict
Alzheimer-related endpoints in isolation. It is whether a frozen contemporary
neuroimaging representation supplies reproducible, endpoint-dependent information
beyond variables that are cheaper to model, easier to validate, and more directly
interpretable.

### Evaluation and generalization in medical AI

The closest studies collectively cover frozen probing, task-specific fine-tuning,
scratch convolutional baselines, longitudinal prediction, and external testing,
but not their combination with matched incremental comparisons. This study does
not claim the first use of Neuro-JEPA on ADNI, the first frozen foundation-model
analysis of ADNI progression, or the first MRI prediction of future cognition.
Its contribution is the intersection of (i) a frozen externally pretrained
representation, (ii) prespecified conventional comparators, (iii) concurrent,
future, conversion, and censored survival endpoints, and (iv) paired
participant-level inference in a locked participant-disjoint internal test.

Computational footprint is secondary rather than rhetorical. Training time,
processor type, memory, data-center efficiency, and electricity carbon intensity
all affect emissions, so model size or GPU use alone cannot establish an
environmental advantage
([Green Algorithms](https://doi.org/10.1002/advs.202100707)). We will therefore
report downstream wall time, hardware, peak accelerator memory, and trainable
parameter counts where available. Unless energy and carbon intensity are measured,
the Discussion will claim only lower downstream computational requirements for
regularized conventional and frozen-probe models, not a quantified reduction in
environmental impact.

## Materials and methods

### Study design and data source

This retrospective computational study uses longitudinal T1-weighted MRI and
associated clinical data from ADNI. Imaging is selected from the local
longitudinal BIDS-style archive, while diagnosis, demographics, cognition, MRI
metadata, and morphometry are aligned from ADNI tabular exports. The exact cohort
is reconstructed from versioned derived manifests; source ADNI data are not
modified.

[Add ADNI governance, ethics, informed-consent language and official data-use
acknowledgement from current ADNI requirements. Add acquisition phases, date of
data access, and cohort flow counts.]

### Cohort construction and participant-level splitting

Sessions are the primary temporal unit. Task-specific rows represent a same-visit
observation, a baseline or first-MCI anchor, or a current-to-future pair matched
to a predefined horizon. Three locked repeats use four participant-level outer
folds and five inner folds. Within an outer fold, 75% of participants are used for
inner model selection and final refitting and 25% for one outer-test evaluation;
each participant is outer-test once per repeat. No historical split seed is used.

[Specify inclusion/exclusion criteria, visit matching tolerance, diagnostic label
construction, late converters, censoring, missing values, and split proportions
from the canonical scripts and manifests.]

### MRI preprocessing and frozen Neuro-JEPA representations

Each eligible T1-weighted MRI is processed using the released Neuro-JEPA input
pipeline and encoded by a frozen pretrained checkpoint. The encoder produces 576
tokens with 768 dimensions; the main analysis uses their 768-dimensional pooled
visit representation. No ADNI label, cognition, morphometry, or outcome is used
to update the encoder.

[Insert exact checkpoint identifier/revision, preprocessing operations, software
versions, extraction quality control, embedding exclusions, and hardware. Verify
all details against the extraction code, metadata summary, and official local
Neuro-JEPA reference.]

### Conventional and foundation feature groups

The prespecified groups are morphometry only, clinical and demographic variables,
their conventional union, frozen Neuro-JEPA only, and Neuro-JEPA combined with
conventional variables. An MRI-specific comparison adds Neuro-JEPA to morphometry
only. Variables are measured at or before the anchor visit. A concurrent target
is never included as a predictor; for future outcomes, its current value may be
included and is compared with a carry-forward baseline.

[Insert the final data dictionary as a table and define preprocessing, scaling,
encoding, missing-value handling, scanner covariates, and fit-on-training-only
operations.]

### Diagnostic and concurrent-state tasks

AD-versus-cognitively-normal classification and concurrent ADAS13, ADAS11, and
MMSE form the diagnostic-state core. No single endpoint is privileged over the
complete prespecified audit matrix.

### Longitudinal prognostic tasks

Prespecified prognostic tasks comprise 12-month cognitive change, future
hippocampus-to-intracranial-volume ratio, future lateral-ventricle-to-intracranial-
volume ratio, MCI conversion within 36 months, and censored time from the first MCI
anchor to Alzheimer dementia. Cognitive analyses at 24 and 36 months are
sensitivity analyses. Future-state prediction is compared with persistence or
carry-forward whenever applicable.

[Insert exact target definitions, horizon windows, event/censoring rules and
sample sizes.]

### Models and model selection

Continuous outcomes use Ridge regression, binary outcomes use regularized
logistic regression, and time-to-event outcomes use a regularized Cox model.
Predictors are standardized using parameters estimated from training data only.
Regularization is selected by tuning performance using a fixed grid shared
across feature groups. The full preprocessing and model pipeline is refitted on
the complete outer-development fold before its single outer-test evaluation.

[Insert grids, optimization software, threshold policy, convergence handling,
and exact primary metrics.]

### Statistical analysis

All primary comparisons are paired on the same test participants. Participants
are resampled with replacement 10,000 times; all rows belonging to a sampled
participant remain together. For repeated observations, regression performance is
reported with participant-equal weighting. Incremental value is the metric of the
combined model minus that of the conventional model; negative differences favor
the combined model for MAE, whereas positive differences favor it for AUROC,
AUPRC, and C-index. Secondary endpoint comparisons use Benjamini-Hochberg false
discovery rate control. Classification results include calibration, and each
analysis reports rows, participants, class counts or events, and censoring.

[Finalize confidence interval method, FDR families, locked-test synthesis,
calibration statistics, and any heterogeneity measure before protocol freeze.]

### Reproducibility

Code, configurations, synthetic examples, aggregate metrics, and non-identifying
derived outputs will be archived in a versioned public repository. ADNI MRI,
clinical tables, and participant-level embeddings will not be redistributed;
authorized users will be able to reconstruct derived artifacts from ADNI source
data.

## Results

### Cohort and task flow

[Table 1; do not populate until every locked outer fold is complete.]

### Diagnostic state and concurrent cognition

[Report AD-vs-CN as a sanity check, then the primary ADAS13 result and concurrent
MMSE/ADAS11. Lead with paired incremental effects and CIs, not model rankings.]

### Short-horizon cognitive prognosis

[Report 12-month effects and carry-forward; move 24/36-month detail to the
supplement.]

### Future anatomical state

[Report hippocampus/ICV primary anatomical endpoint and ventricles/ICV secondary
endpoint by horizon.]

### MCI conversion and survival

[Report discrimination, calibration, events, censoring, and null/negative results
without selective omission.]

### Locked-test and sensitivity analyses

[Forest plot of paired effects in the locked test and all prespecified
sensitivity analyses.]

## Discussion

### Principal findings

[Answer the title directly using the final evidence. Distinguish diagnostic,
concurrent, and prognostic value.]

### Brain-informatics implications

[Explain why endpoint-aware incremental auditing is needed for foundation brain
representations and why static decodability does not imply a sufficient disease
state or useful predicted trajectory.]

### Comparison with prior work

[Compare study design and evidence, avoiding unsupported claims of being first.]

### Strengths and limitations

[Include ADNI-only data, a single foundation encoder unless extended, correlated
endpoints, endpoint selection informed by historical analyses, repeated visits,
linear readouts, data-use restrictions, and absence of demonstrated clinical
utility.]

## Conclusions

[One calibrated paragraph tied to paired effects and fold/repeat dispersion.]

## List of abbreviations

AD: Alzheimer disease; ADAS: Alzheimer's Disease Assessment Scale; ADNI:
Alzheimer's Disease Neuroimaging Initiative; AUPRC: area under the precision-recall
curve; AUROC: area under the receiver operating characteristic curve; CI:
confidence interval; ICV: intracranial volume; MAE: mean absolute error; MCI: mild
cognitive impairment; MMSE: Mini-Mental State Examination; MRI: magnetic
resonance imaging.

## Declarations

### Availability of data and materials

The ADNI data analyzed in this study are available to qualified investigators
through the ADNI data-access process and cannot be redistributed by the authors.
Code, configurations, synthetic examples, and non-identifying aggregate outputs
supporting the manuscript will be made available at [repository and archived DOI].

### Competing interests

[To be completed by all authors.]

### Funding

[To be completed. State the funders' roles.]

### Authors' contributions

[To be completed using author initials and CRediT-consistent roles.]

### Acknowledgements

[Insert the current required ADNI acknowledgement and contributor statements.]

### Authors' information

Not applicable.

## References

Working primary-source bibliography; convert to the journal's numbered style and
verify complete metadata at manuscript freeze.

1. Assran M, Duval Q, Misra I, et al. Self-supervised learning from images with a
   joint-embedding predictive architecture. CVPR. 2023:15619-15629.
   https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html.
2. Assran M, Bardes A, Fan D, et al. V-JEPA 2: self-supervised video models enable
   understanding, prediction and planning. arXiv:2506.09985. 2025.
   https://doi.org/10.48550/arXiv.2506.09985.
3. Wang H, Guo S, Ye J, et al. SAM-Med3D: towards general-purpose segmentation
   models for volumetric medical images. arXiv:2310.15161. 2023.
   https://doi.org/10.48550/arXiv.2310.15161.
4. Tak D, Garomsa BA, Zapaishchykova A, et al. A generalizable foundation model
   for analysis of human brain MRI. Nature Neuroscience. 2026.
   https://doi.org/10.1038/s41593-026-02202-6.
5. Wu Y, Wang S, Li Y, et al. BrainDINO: a brain MRI foundation model for
   generalizable clinical representation learning. arXiv:2604.27277. 2026.
   https://doi.org/10.48550/arXiv.2604.27277.
6. Barbano CA, Brunello M, Dufumier B, Grangetto M. Anatomical foundation models
   for brain MRIs. Pattern Recognition Letters. 2026;199:178-184.
   https://doi.org/10.1016/j.patrec.2025.11.028.
7. Huang H, Chen L, Chen J, et al. Learning sparse latent predictive foundation
   model for multimodal neuroimaging. arXiv:2606.14957. 2026.
   https://doi.org/10.48550/arXiv.2606.14957.
8. Ma D, Pabalan C, Rajagopal A, et al. Predicting categorical and continuous
   Alzheimer's disease outcomes from a single MRI scan. Nature Aging.
   2026;6:1121-1137. https://doi.org/10.1038/s43587-026-01121-2.
9. Demir B, Tian L, Greer H, et al. How useful are vision foundation model
   features for out-of-the-box disease progression prediction? CVPR Workshops.
   2026.
10. Marinescu RV, Oxtoby NP, Young AL, et al. The Alzheimer's Disease Prediction
    Of Longitudinal Evolution (TADPOLE) Challenge: results after 1 year follow-up.
    J Mach Learn Biomed Imaging. 2021;2021:019.
11. Lannelongue L, Grealey J, Inouye M. Green Algorithms: quantifying the carbon
    footprint of computation. Advanced Science. 2021;8:2100707.
    https://doi.org/10.1002/advs.202100707.

[Add the 2026 routine-clinical/MRI cognitive-decline study and all conventional
morphometry, conversion, survival, calibration, leakage, and reporting references
after full-text metadata verification. Do not reuse the provisional reference
list from the older Stage 1 manuscript without checking relevance and metadata.]
