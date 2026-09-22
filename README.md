# FM-AD Audit: Auditing Neuroimaging Foundation Models in Alzheimer Disease

This repository contains the reusable analysis code for:

> *Do Frozen Neuroimaging Foundation Representations Add Diagnostic and
> Prognostic Value in Alzheimer Disease? A Longitudinal ADNI Audit*

It evaluates whether a frozen MRI representation adds predictive information to
prespecified clinical, demographic, and morphometric variables. The public
repository contains the audit implementation and frozen study protocol, not the
ADNI data, participant-level features, fitted outputs, manuscript source, or
presentation material.

## What is reproducible

The code makes the computational procedure inspectable and reusable. A user can
prepare a longitudinal dataset, provide either Neuro-JEPA embeddings or fixed
embeddings from another model, and run the same participant-aware comparisons,
nested cross-validation, calibration, survival analysis, and paired reporting.

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
- `configs/protocol.json`: the frozen paper protocol and estimands;
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

## Data governance and licenses

Never commit ADNI images, clinical exports, participant identifiers,
participant-level embeddings, access tokens, or generated predictions. Users
must obtain their own ADNI and model access and comply with the corresponding
terms.

Original code in this repository is MIT licensed. The Neuro-JEPA software and
weights remain subject to their own licenses and access conditions.
