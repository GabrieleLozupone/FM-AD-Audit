# FM-AD Audit: Auditing Frozen Vision Foundation Models in Alzheimer Disease

This repository contains the reusable analysis code for:

> *Do Frozen Vision Foundation Models Add Diagnostic and Prognostic Value in
> Alzheimer Disease? A Longitudinal Study*

It evaluates whether frozen MRI representations from three vision foundation
models (Neuro-JEPA, DINOv3, and BrainIAC) add predictive information to
prespecified clinical, demographic, and morphometric variables. The public
repository contains the audit implementation and study protocols, not the ADNI
data, participant-level features, fitted outputs, manuscript source, or
presentation material.

## What is reproducible

The code makes the computational procedure inspectable and reusable. A user can
prepare a longitudinal dataset, provide Neuro-JEPA, DINOv3, or BrainIAC
embeddings or fixed embeddings from another model, and run the same
participant-aware comparisons, nested cross-validation, calibration, survival
analysis, and paired reporting.

The paper's exact numerical results are **not guaranteed to be reproducible from
a new ADNI download**. ADNI is a continuously maintained resource: available
images, corrected records, derived variables, and download selections can differ
between access dates. ADNI data-use restrictions also prevent redistribution of
the paper's participant-level source data and embeddings. Exact reconstruction
would additionally require the same historical ADNI export, image-selection
decisions, preprocessing software, model checkpoint, exclusions, and frozen
cohort manifests.

This is therefore a method-reproduction repository, not a redistributable data
snapshot. Users should report their ADNI access date, download/query definition,
software versions, model revision, preprocessing decisions, exclusions, and
cohort counts. Results obtained from a newly assembled cohort are a new
application of the audit and should not be described as an exact replication of
the published numbers.

## Repository contents

- `src/neurojepa_adni_audit/`: reusable cohort, feature, training, evaluation,
  calibration, survival, and reporting logic;
- `scripts/`: command-line entry points for preprocessing QC, embedding
  extraction, nested analyses, and paired reports;
- `configs/protocol.json`: the frozen Neuro-JEPA protocol (0.5.0) and estimands;
- `configs/protocol_multi_encoder.json`: the draft protocol 0.6.0 that records
  the DINOv3 and BrainIAC settings, families, and invariants;
- `configs/protocol_ridge_sensitivity.json`: protocol 0.6.0 with only the ridge
  grid extended, used by the post hoc ridge sensitivity analysis;
- `configs/nested_paired_report.json` and
  `configs/nested_paired_report_multi_encoder.json`: the paired-report
  specifications of the Neuro-JEPA and multi-encoder analyses;
- `configs/*.example.*`: portable configuration examples;
- `RUNBOOK.md`: a general workflow for a user-prepared dataset.

Tests, raw or derived participant data, embeddings, results, paper files,
presentations, caches, and local project notes are intentionally excluded from
the publication tree.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To extract Neuro-JEPA features, install the upstream encoder at the code revision
used by this study:

```bash
python -m pip install \
  "neurojepa @ git+https://github.com/NYUMedML/Neuro-JEPA.git@b8074740cd3e6ad72e5d032aa49617bdae327240"
```

The released Neuro-JEPA weights are gated. Obtain access from the model provider
and authenticate through Hugging Face without saving credentials in this
repository. The frozen model revision and checkpoint hashes used for the paper
are recorded in `configs/protocol.json`; model weights are not redistributed.

To extract DINOv3 features, install the optional dependency and request access
to the gated `facebook/dinov3-vitb16-pretrain-lvd1689m` model on Hugging Face:

```bash
python -m pip install -e ".[encoders]"
```

BrainIAC needs no extra package: `src/neurojepa_adni_audit/encoders/brainiac.py`
rebuilds the upstream MONAI backbone. Obtain the foundation checkpoint
`BrainIAC.ckpt` from the upstream repository
(<https://github.com/AIM-KannLab/BrainIAC>, commit
`ba60f45bed832ee0e5683c678caeaeefe2072f0d`); the extractor refuses a file
whose SHA-256 differs from the one in `configs/protocol_multi_encoder.json`.
The fine-tuned task checkpoints distributed with it are not used.

## Required user-prepared inputs

The audit does not download or preprocess a cohort automatically. At minimum,
the user supplies:

1. A visit-level CSV manifest with stable `participant_id`, `session_id`, and
   unique `image_id` keys; visit time; diagnosis or outcomes required by the
   selected task; and predictor columns.
2. One preprocessed T1-weighted NIfTI path per eligible visit when Neuro-JEPA
   embeddings are to be extracted.
3. An embedding-index CSV keyed by the same three identifiers, with
   `pooled_embedding_path` pointing to one finite one-dimensional NumPy array per
   image. Neuro-JEPA produces 768-dimensional pooled embeddings. Another model
   may use a different fixed dimension, supplied to the integrity checker with
   `--expected-dimension`.
4. Longitudinal pair tables for analyses that forecast a future visit. Pairs must
   preserve participant and visit identifiers, temporal order, requested
   horizon, and observed interval.

Clinical and morphometric columns can be missing at the row level; predictors
are imputed within training folds. Outcomes and embeddings are never imputed.
All visits from a participant must remain in the same fold.

## General MRI preprocessing specification

Users are responsible for implementing and validating their upstream imaging
pipeline. It should, at a minimum:

1. select one eligible T1-weighted scan per participant and visit using a
   documented, outcome-blind rule;
2. convert images to NIfTI while preserving identifiers and acquisition
   provenance;
3. apply a consistent correction, brain extraction, and spatial-normalization
   procedure appropriate to the chosen encoder;
4. visually inspect orientation, coverage, registration, artifacts, and gross
   failures before feature extraction;
5. derive morphometry with one documented segmentation pipeline and normalize
   the prespecified regional volumes by contemporaneous intracranial volume;
6. retain an exclusion log and versioned manifest rather than silently dropping
   failed scans.

For Neuro-JEPA, `scripts/extract_neurojepa_embeddings.py` then applies the model
input transform recorded in the frozen protocol: replace non-finite voxels,
orient to RAS, resample to 1 mm isotropic spacing, foreground crop, pad/crop and
resize to the encoder field of view, center crop to `96 x 108 x 96`, percentile
scale intensities to `[0, 1]`, and cast to `float32`. No augmentation is applied.
Run `scripts/qc_neurojepa_inputs.py` on a representative sample before a full
extraction. These operations do not replace upstream scan selection, artifact
QC, bias correction, brain extraction, registration, or morphometric
segmentation.

## Audit design

The included paper protocol uses participant-level repeated nested
cross-validation. Preprocessing of tabular predictors, feature selection, and
hyperparameter selection are fitted inside training data only. The outer held-out
predictions are compared on identical paired cohorts. Core tasks include AD vs
CN classification, same-visit cognition, future cognition and morphometry,
36-month MCI conversion, and first-MCI survival.

The six canonical feature groups are `morphometry_only`,
`clinical_demographic`, `conventional_full`, `neurojepa_only`,
`neurojepa_morphometry`, and `neurojepa_conventional`. When auditing another
encoder, the `neurojepa_*` labels should be interpreted as the supplied frozen
representation; record that substitution clearly in the derived protocol and
reports.

See [RUNBOOK.md](RUNBOOK.md) for the ordered, dataset-agnostic workflow.

## Encoders and protocol versions

All encoders read the same preprocessed T1-weighted images: N4-corrected,
skull-stripped, and affinely registered (ANTs) to the 1 mm MNI152 grid
(182 x 218 x 182). Each then applies only its own model-input transform. All
representations have 768 dimensions.

| Encoder | Pinned model | Representation |
|---|---|---|
| Neuro-JEPA | `NYUMedML/Neuro-JEPA`, revision `abdfc94aa3d2654a81a14afeb834dc04653b8976`, code `b8074740cd3e6ad72e5d032aa49617bdae327240` | mean of the 576 patch tokens |
| DINOv3 ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision `5931719e67bbdb9737e363e781fb0c67687896bc`, `model.safetensors` SHA-256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`, transformers 5.12.1 | CLS token of every 1 mm slice in a fixed MNI brain range (443 slices), averaged within each plane, then equal-weight mean of the axial, coronal, and sagittal vectors |
| BrainIAC (3D ViT-B, SimCLR) | upstream commit `ba60f45bed832ee0e5683c678caeaeefe2072f0d`, `BrainIAC.ckpt` SHA-256 `f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693`, MONAI 1.5.2 | primary: mean of the 216 patch tokens; secondary: upstream token 0 (`brainiac_token0`), the first patch token |

For BrainIAC only, background voxels below 0.1% of the volume's 99th-percentile
positive intensity are set to zero before the upstream resize and nonzero
intensity normalization, because the shared images carry near-zero
registration residue outside the brain. Upstream pins MONAI 1.3.2; an
outcome-blind check on 25 scans gave bit-identical model inputs and tokens
under MONAI 1.3.2 and 1.5.2.

Protocol status:

- `configs/protocol.json` (0.5.0) is frozen and unchanged. The Neuro-JEPA
  analyses were prespecified under 0.4.0 and 0.5.0.
- `configs/protocol_multi_encoder.json` (0.6.0) has status `draft` and was never
  frozen. The DINOv3 and BrainIAC analyses were added after the Neuro-JEPA
  results were known. They are a post hoc comparison under the unchanged
  design: same images, participants, folds, endpoints, predictor groups, grids,
  decoders, and statistics. The runners are therefore called with
  `--allow-draft-protocol`.
- The ridge sensitivity analysis was decided after the primary results, when
  inner cross-validation was seen to select the largest ridge penalty (1000)
  in most embedding-model regression fits. It is a post hoc sensitivity
  analysis, not a prespecified one, and it does not change the primary
  results. `configs/protocol_ridge_sensitivity.json` differs from 0.6.0 only in
  `protocol_version`, `ridge_alphas` (extended to 1e6), and an added
  `ridge_sensitivity` note; its status is also `draft`.

Both 0.6.0 files are kept byte-identical to the versions used for the runs,
because every run records the protocol SHA-256 and
`scripts/run_ridge_sensitivity.py` regenerates the ridge protocol from 0.6.0 and
stops on any difference. The following statements in them are known to be
inaccurate and are corrected here:

- `additional_encoders.brainiac.upstream_preprocessing_difference` says the
  shared inputs are "nonlinearly registered to MNI152". They were affinely
  registered. The BrainIAC deviation is affine (this study) versus rigid
  (BrainIAC pretraining) registration.
- `additional_encoders.brainiac.architecture` says "216 patch tokens plus CLS".
  The MONAI backbone has no CLS token, as `token_layout_finding` in the same
  file states.
- `study.title` is a working title, not the paper title.
- The 0.6.0 amendment and `implementation_gates_before_freeze_0_6_0` describe a
  freeze before the first DINOv3 or BrainIAC prediction. That freeze did not
  take place; see the status above.
- `multi_encoder_analysis.invariants` says that the encoder copies of
  `embedding_index.csv` and `horizon_matched_pairs.csv` differ only in the
  pooled-path columns. The token-path, token-count, and model-metadata columns
  are also rewritten (token paths blanked); all other cells are verified
  byte-identical.
- The `ridge_sensitivity.reason` field calls the analysis "declared". It is a
  post hoc sensitivity analysis.

## Data governance and licenses

Never commit ADNI images, clinical exports, participant identifiers,
participant-level embeddings, access tokens, or generated predictions. Users
must obtain their own ADNI and model access and comply with the corresponding
terms.

Original code in this repository is MIT licensed. The Neuro-JEPA, DINOv3, and
BrainIAC software and weights remain subject to their own licenses and access
conditions (BrainIAC is recorded in the protocol as licensed for non-commercial
academic research use only).
