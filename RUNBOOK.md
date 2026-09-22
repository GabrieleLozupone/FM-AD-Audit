# General audit runbook

This runbook describes how to apply the longitudinal representation audit to a
user-prepared dataset. It deliberately does not assume access to the authors'
ADNI snapshot or local directory layout. Run commands from the repository root
and use paths appropriate to your environment.

The former project-specific runbook is retained locally at
`.publication-backup/RUNBOOK.pre-publication.md` and is excluded from Git.

## 1. Freeze the analysis definition

Before inspecting held-out results, copy `configs/protocol.json` and adapt it to
the new study. Record the cohort source and access date; eligibility and outcome
definitions; model and checkpoint revisions and hashes; preprocessing and
morphometry versions; feature groups, metrics, exclusions, model grids; and
participant-level split seeds. Give a modified protocol a new version rather
than presenting it as the paper protocol.

## 2. Prepare images and clinical data

Build a visit-level manifest without using outcomes to choose scans or to decide
whether preprocessing succeeded. Each row represents one imaging visit.

Required identifiers:

```text
participant_id, session_id, image_id, session_month
```

For Neuro-JEPA extraction, add `processed_t1_path`. For the paper feature groups,
prepare the applicable columns below:

```text
diagnosis_at_session
age_at_session, sex_baseline, education_years_baseline, apoe4_status
mmse_at_session, adas11_at_session, adas13_at_session
vol_total_intracranial
vol_hippocampus_total, vol_lateral_ventricles_total
vol_left_amygdala, vol_right_amygdala
vol_left_inferior_lateral_ventricle, vol_right_inferior_lateral_ventricle
vol_left_cerebral_cortex, vol_right_cerebral_cortex
```

Only columns needed by the selected endpoint are mandatory. Use stable
pseudonymous identifiers, unique image keys, numeric visit times in months, and
consistent units. Keep the source-identifier mapping outside the repository.

Recommended upstream MRI preparation is documented scan selection,
DICOM-to-NIfTI conversion, intensity non-uniformity correction, brain extraction,
spatial normalization appropriate to the encoder, visual QC, and a documented
segmentation pipeline for morphometry. Apply one consistent pipeline, record all
failures and versions, and avoid manual decisions informed by outcomes.

## 3. Extract or supply frozen embeddings

For Neuro-JEPA, first inspect a representative QC sample:

```bash
python scripts/qc_neurojepa_inputs.py \
  --manifest-path /path/to/manifest.csv \
  --image-path-column processed_t1_path \
  --output-dir /private/path/input_qc \
  --model-name-or-path NYUMedML/Neuro-JEPA \
  --device cuda
```

Then extract pooled representations using an immutable approved model revision:

```bash
python scripts/extract_neurojepa_embeddings.py \
  --manifest-path /path/to/manifest.csv \
  --image-path-column processed_t1_path \
  --output-dir /private/path/embeddings \
  --index-path /private/path/embedding_index.csv \
  --summary-path /private/path/embedding_summary.json \
  --model-name-or-path NYUMedML/Neuro-JEPA \
  --hf-revision YOUR_IMMUTABLE_REVISION \
  --feature-kind pooled --save-dtype float32 --device cuda --resume
```

For another frozen model, run its preprocessing and inference separately. Save
one finite, fixed-length one-dimensional `.npy` vector per image and create an
index with:

```text
participant_id, session_id, image_id, pooled_embedding_path
```

Use the same frozen encoder and pooling rule for all visits. Do not use outcome
labels during representation extraction.

## 4. Validate and freeze the embeddings

Validate exact key agreement, uniqueness, paths, dimensions, finite values, and
file hashes. The output directory must not already exist.

```bash
python scripts/build_embedding_integrity_manifest.py \
  --embedding-index-path /private/path/embedding_index.csv \
  --source-manifest-path /path/to/manifest.csv \
  --output-dir /private/path/embedding_integrity \
  --expected-dimension 768
```

Replace `768` with the dimension of a different model.

## 5. Prepare longitudinal pairs

Create outcome-blind pairs within participants. A follow-up must be strictly
later than its anchor. Select the nearest visit inside each prespecified horizon
window and retain the observed interval and horizon error.

If a progression-pair table already exists, attach embeddings and select
horizon-matched pairs with:

```bash
python scripts/build_state_transition_pairs.py \
  --progression-pairs-path /path/to/progression_pairs.csv \
  --embedding-index-path /private/path/embedding_index.csv \
  --pair-policy horizon_matched --horizons 12 24 36 \
  --horizon-tolerance-months 3 \
  --output-path /private/path/horizon_pairs.csv \
  --summary-path /private/path/horizon_pairs_summary.json
```

Inspect participant and pair counts, temporal ordering, follow-up distribution,
and missing outcomes before training.

## 6. Create participant-level nested folds

All visits from one participant must remain together. Learned preprocessing and
hyperparameter selection belong in inner training folds; outer folds are used
once for held-out evaluation.

```bash
python scripts/prepare_nested_cv.py \
  --protocol-path /path/to/your_protocol.json \
  --manifest-path /path/to/manifest.csv \
  --embedding-index-path /private/path/embedding_index.csv \
  --embedding-integrity-dir /private/path/embedding_integrity \
  --progression-pairs-path /path/to/progression_pairs.csv \
  --horizon-pairs-path /private/path/horizon_pairs.csv \
  --output-dir /private/path/nested_cv_inputs
```

Use `--allow-draft-protocol` and `--allow-dirty-worktree` only for development,
never for a locked analysis.

## 7. Audit cohorts, then run models

First generate outcome-blind cohort flows and confirm that each paired comparison
uses identical participants and observations:

```bash
python scripts/audit_nested_cohorts.py --help
```

Resolve unexplained key mismatches, duplicates, temporal errors, and
feature-group cohort differences. Then use the relevant nested entry points:

```bash
python scripts/run_nested_tabular_probe.py --help
python scripts/run_nested_survival_probe.py --help
python scripts/run_nested_nonlinear_probe.py --help
```

Run all prespecified feature groups on the same folds. The main incremental
contrasts are `morphometry_only` versus `neurojepa_morphometry`, and
`conventional_full` versus `neurojepa_conventional`. For another encoder, the
`neurojepa_*` labels refer operationally to the supplied embedding; document the
substitution. Never tune on outer-fold predictions. Outcomes and absent
embeddings are not imputed.

## 8. Produce paired reports

After all held-out predictions are complete, copy and adapt
`configs/nested_paired_report.example.json`, then run:

```bash
python scripts/report_nested_paired_results.py \
  --spec-path /path/to/paired_report.json \
  --output-dir /private/path/paired_report
```

Report absolute performance and paired incremental effects, uncertainty,
calibration, fold/repeat dispersion, cohort counts, exclusions, missingness, and
multiplicity correction. Repeated splits are correlated resamples, not
independent external replications.

## 9. Archive provenance privately

Privately retain hashes of manifests, embeddings, folds, protocol, code commit,
environment, commands, logs, and exclusion counts. Publicly release only
aggregate, disclosure-checked results allowed by the data-use agreement.

Never commit images, participant tables, identifiers, embeddings, predictions,
credentials, or generated outputs. With a new ADNI download, describe the work
as an application to the cohort assembled on the recorded access date, not as an
exact reproduction of the paper's numerical results.
