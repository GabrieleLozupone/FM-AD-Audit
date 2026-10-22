# General audit runbook

This runbook describes how to apply the longitudinal representation audit to a
user-prepared dataset. It deliberately does not assume access to the authors'
ADNI snapshot or local directory layout. Run commands from the repository root
and use paths appropriate to your environment.

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

## 10. Reproduce the multi-encoder extension (DINOv3, BrainIAC)

The extension evaluates DINOv3 and BrainIAC on the completed Neuro-JEPA audit
without changing its cohort, folds, endpoints, predictor groups, grids,
decoders, or statistics. It is a post hoc comparison under the draft protocol
`configs/protocol_multi_encoder.json` (0.6.0), so the runners are called with
`--allow-draft-protocol`. See the README for model pins and known protocol
wording errors.

The extension scripts use a fixed layout relative to the repository root.
Before starting, the Neuro-JEPA audit must be complete under
`results/paper_neurojepa_feature_audit/`:

```text
locked_inputs/            nested-CV inputs (manifest, pairs, subjects, folds,
                          embedding_index.csv with source_image_path)
experiments/              linear and survival runs
nonlinear_experiments/    MLP runs
paired_report/nested_paired_effects.csv
cohort_audit.json
```

New outputs are written under `results/fm_encoders/`; the Neuro-JEPA results
are only read. Install the encoder extra and obtain model access first
(README, Installation). Run the steps one at a time, in order.

1. Outcome-blind input QC on a small sample. The sample CSV needs
   `participant_id`, `session_id`, `image_id`, and `processed_t1_path`; each
   sampled participant's next locked session is added automatically.

   ```bash
   python scripts/qc_encoder_inputs.py --encoder brainiac \
     --sample-manifest-path /path/to/qc_sample.csv \
     --checkpoint-path /private/path/BrainIAC.ckpt --gpu 0
   python scripts/qc_encoder_inputs.py --encoder dinov3 \
     --sample-manifest-path /path/to/qc_sample.csv --gpu 0
   python scripts/qc_background_residue.py \
     --sample-manifest-path /path/to/qc_sample.csv
   ```

   Inspect `results/fm_encoders/qc/<encoder>_inputs/montage_*.png` and
   `qc_summary.json`, and `results/fm_encoders/qc/background_residue/`.

2. Extract embeddings for every locked image. Both extractors are resumable,
   check the pinned checkpoint hash, and must report 10,470/10,470 images for
   the paper cohort.

   ```bash
   python scripts/extract_brainiac_embeddings.py \
     --checkpoint-path /private/path/BrainIAC.ckpt --gpu 0
   python scripts/extract_dinov3_embeddings.py --gpu 0
   ```

   Outputs: `results/fm_encoders/{dinov3,brainiac,brainiac_token0}/embedding_index.csv`
   and `embedding_summary.json`; vectors under `results/fm_encoders/{dinov3,brainiac}/embeddings/`.

3. Validate the embeddings (the output directories must not exist yet):

   ```bash
   for rep in dinov3 brainiac brainiac_token0; do
     python scripts/build_embedding_integrity_manifest.py \
       --embedding-index-path results/fm_encoders/$rep/embedding_index.csv \
       --source-manifest-path results/paper_neurojepa_feature_audit/locked_inputs/manifest.csv \
       --output-dir results/fm_encoders/$rep/embedding_integrity \
       --expected-dimension 768
   done
   ```

4. Build each encoder's input directory. Design files are copied byte for byte,
   only the embedding columns are rewritten, and all 21 locked endpoint tables
   must load identical rows from both directories.

   ```bash
   for rep in dinov3 brainiac brainiac_token0; do
     python scripts/prepare_encoder_locked_inputs.py --representation $rep
   done
   ```

5. Audit the cohorts and require them to equal the Neuro-JEPA audit (only the
   protocol hash may differ):

   ```bash
   for rep in dinov3 brainiac brainiac_token0; do
     python scripts/audit_nested_cohorts.py \
       --protocol-path configs/protocol_multi_encoder.json --allow-draft-protocol \
       --input-dir results/fm_encoders/$rep/locked_inputs \
       --output-path results/fm_encoders/$rep/cohort_audit.json
   done
   python scripts/compare_cohort_audits.py \
     results/fm_encoders/{dinov3,brainiac,brainiac_token0}/cohort_audit.json
   ```

   Expected: `IDENTICAL` for all three.

6. Run the experiment matrices. Without `--execute` the script prints a
   read-only plan. The encoder-independent conventional groups are reused from
   the Neuro-JEPA root through read-only symlinks and are never refitted.

   ```bash
   python scripts/run_encoder_matrix.py --representation all
   python scripts/run_encoder_matrix.py --representation all --link-baselines
   python scripts/run_encoder_matrix.py --representation all --linear \
     --workers 32 --execute
   python scripts/run_encoder_matrix.py --representation all --mlp \
     --workers 3 --gpus 0,1,2 --execute
   python scripts/run_encoder_matrix.py --representation all --status
   ```

   Expected: 756 linear and survival runs and 240 MLP runs per representation
   (2,268 and 720 in total), under
   `results/fm_encoders/<rep>/{experiments,nonlinear_experiments}/`. Runs are
   resumable; delete any directory reported as `INCOMPLETE` and rerun.

7. Build the paired reports:

   ```bash
   python scripts/build_multi_encoder_report_spec.py
   python scripts/run_multi_encoder_report.py
   ```

   The first command checks every comparison id and family against the
   protocol, creates the standalone-contrast symlinks under
   `results/fm_encoders/paired_report_inputs/`, and refuses to overwrite
   `configs/nested_paired_report_multi_encoder.json` with different content
   (107 comparisons in 8 families). The second writes
   `results/fm_encoders/paired_report/nested_paired_effects.csv` (with the
   33-comparison all-encoder sensitivity q-value), `outer_fold_effects.csv`, and
   one folder per family. The published Neuro-JEPA effects are read, not
   recomputed.

## 11. Post hoc ridge regularization sensitivity

This analysis was decided after the primary results and is reported as a post
hoc sensitivity analysis; the primary results are unchanged. It refits the 8
regression core endpoints for Neuro-JEPA, DINOv3, and BrainIAC (patch mean)
with the ridge grid extended from 0.01-1000 to 0.01-1e6, using the unchanged
runner, inputs, folds, and targets. It requires steps 1-7 above.

```bash
python scripts/run_ridge_sensitivity.py              # read-only plan: 288 jobs
python scripts/run_ridge_sensitivity.py --execute --workers 32
python scripts/run_ridge_sensitivity.py --status
python scripts/run_ridge_sensitivity.py --report --workers 4
```

`--execute` first checks that `configs/protocol_ridge_sensitivity.json` is
exactly the derivation of `configs/protocol_multi_encoder.json`. Outputs under
`results/fm_encoders/ridge_sensitivity/`:

- `paired_report/nested_paired_effects.csv`: refitted contrasts with
  `q_full_family` (Benjamini-Hochberg over each original family, reusing the
  unchanged logistic and Cox p-values) next to the primary estimate, interval,
  and q-value;
- `alpha_selection.csv` and `alpha_summary.csv`: selected alpha per run under
  the original and extended grids;
- `identity_check.csv`: for every run whose selected alpha did not change,
  whether the refit reproduces the primary outer-test predictions (the report
  counts runs that agree within 1e-4).
