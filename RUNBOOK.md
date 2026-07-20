# Experiment execution runbook

This file records which scripts to run, in what order, and why. Run commands
from the repository root. The commands below use the linked historical inputs
for smoke testing; they do **not** produce confirmatory paper results.

## Workflow overview

| Order | Script | Purpose | Required? |
|---:|---|---|---|
| 1 | `python -m pytest` | Verify the environment and core feature/reporting logic. | Always |
| 2 | `scripts/check_linked_data.sh` | Verify that all local derived-data links resolve. | Local linked workflow |
| 3 | `qc_neurojepa_inputs.py` | Visually and numerically check preprocessing before extracting embeddings. | Before a new extraction |
| 4 | `extract_neurojepa_embeddings.py` | Create frozen pooled/token embeddings and their index. | Skip when valid embeddings already exist |
| 5 | `build_state_transition_pairs.py` | Join embeddings to longitudinal pairs and select forecast horizons. | Future morphometry |
| 6 | `prepare_paper_replications.py` | Create immutable split-specific inputs, hashes, and provenance. | After protocol freeze |
| 7 | `train_neurojepa_downstream_probe.py` | Run AD-vs-CN and 36-month MCI-conversion classification. | Paper tasks |
| 8 | `train_neurojepa_cognitive_probe.py` | Run same-visit and future cognitive Ridge probes. | Paper tasks |
| 9 | `train_neurojepa_future_state_probe.py` | Run horizon-matched future morphometry probes. | Paper tasks |
| 10 | `train_neurojepa_survival_probe.py` | Run first-MCI-to-AD Cox survival probes. | Paper task |
| 11 | `report_paired_results.py` | Compute paired effects, bootstrap CIs, sign p-values, and within-family FDR. | After matched models finish |

The task-training scripts in steps 6--9 are independent once their inputs are
ready. Their order follows the manuscript: diagnostic sanity check, concurrent
cognition, prognosis, conversion, and survival.

## 1. Activate and validate the environment

```bash
conda activate neurojepa-adni-audit
python -m pytest
bash scripts/check_linked_data.sh
```

Why: this catches missing Python dependencies, broken imports, and unavailable
linked data before any expensive computation. Do not continue after a failure.

For the existing linked smoke workflow, define the inputs once:

```bash
export AUDIT_MANIFEST="data/linked/manifest.csv"
export AUDIT_EMBEDDING_INDEX="data/linked/embedding_index.csv"
export AUDIT_PROGRESSION_PAIRS="data/linked/progression_pairs.csv"
```

## 2. QC before a new embedding extraction

Skip this and step 3 when `data/linked/embedding_index.csv` and all referenced
embedding files have already passed validation.

First request access to the gated Neuro-JEPA weights and authenticate as
described in `README.md`. Then run a small longitudinal sample:

```bash
python scripts/qc_neurojepa_inputs.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --output-dir results/smoke/input_qc \
  --model-name-or-path NYUMedML/Neuro-JEPA \
  --device cuda
```

Why: it checks orientation, spatial preprocessing, intensity statistics, token
shapes, and within/between-subject embedding similarities before committing to a
full-cohort GPU run. Inspect the generated images, CSV files, and summary JSON.

Before confirmatory extraction, also pass the frozen model revision with
`--hf-revision`; do not silently use a moving Hub revision.

## 3. Extract frozen Neuro-JEPA embeddings

Use `--feature-kind both` because pooled vectors support the main linear probes
and token embeddings support attentive sensitivity analyses. Private outputs are
kept below the ignored `data/private/` directory.

```bash
python scripts/extract_neurojepa_embeddings.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --output-dir data/private/neurojepa_embeddings \
  --index-path data/private/neurojepa_embedding_index.csv \
  --summary-path data/private/neurojepa_embedding_summary.json \
  --model-name-or-path NYUMedML/Neuro-JEPA \
  --feature-kind both \
  --save-dtype float32 \
  --device cuda \
  --resume
```

Why: downstream experiments must reuse one frozen representation rather than
reloading or updating the encoder separately for each endpoint. `--resume`
avoids recomputing completed scans. After extraction, point later commands at
the new index:

```bash
export AUDIT_EMBEDDING_INDEX="data/private/neurojepa_embedding_index.csv"
```

## 4. Build horizon-matched transition pairs

```bash
python scripts/build_state_transition_pairs.py \
  --progression-pairs-path "$AUDIT_PROGRESSION_PAIRS" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --pair-policy horizon_matched \
  --horizons 12 24 36 \
  --horizon-tolerance-months 3 \
  --output-path results/smoke/horizon_matched_pairs.csv \
  --summary-path results/smoke/horizon_matched_pairs_summary.json
```

Why: future anatomy must compare visits near prespecified forecast horizons,
not mix arbitrary adjacent-visit gaps. Review the summary counts and horizon
errors before training. For the linked smoke workflow, the already generated
`data/linked/horizon_matched_pairs.csv` can be used instead.

## 5. Run the historical smoke experiments

These commands test the end-to-end implementation. Write them under
`results/smoke/`, which is ignored by Git, and never copy their numerical output
into the paper.

### 5.1 AD-vs-CN diagnostic sanity check

```bash
python scripts/train_neurojepa_downstream_probe.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --output-dir results/smoke/ad_vs_cn \
  --task ad_vs_cn \
  --feature morphometry_only \
  --feature clinical_demographic \
  --feature conventional_full \
  --feature neurojepa_only \
  --feature neurojepa_morphometry \
  --feature neurojepa_conventional \
  --seed 20260623
```

Why: this establishes whether diagnostic state is decodable and whether the
embedding adds value beyond matched conventional inputs. It is a sanity check,
not the primary endpoint.

### 5.2 Same-visit cognition

```bash
python scripts/train_neurojepa_cognitive_probe.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --output-dir results/smoke/same_visit_cognition \
  --task same_visit \
  --target adas13 --target adas11 --target mmse \
  --feature morphometry_only \
  --feature clinical_demographic \
  --feature conventional_full \
  --feature neurojepa_only \
  --feature neurojepa_morphometry \
  --feature neurojepa_conventional \
  --seed 20260623
```

Why: same-visit ADAS13 is the planned primary endpoint; ADAS11 and MMSE are
secondary endpoints. All models use the same participant split and Ridge grid.

### 5.3 Future cognition

```bash
python scripts/train_neurojepa_cognitive_probe.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --output-dir results/smoke/future_cognition \
  --task future_change \
  --target adas13 --target adas11 --target mmse \
  --horizon-months 12 24 36 \
  --horizon-tolerance-months 3 \
  --feature conventional_full \
  --feature neurojepa_conventional \
  --seed 20260623
```

Why: this tests short-horizon prognostic value against current cognitive state
and the carry-forward baseline. Twelve months is the main horizon; 24 and 36
months are sensitivity analyses.

### 5.4 Future morphometry

```bash
python scripts/train_neurojepa_future_state_probe.py \
  --table-path data/linked/horizon_matched_pairs.csv \
  --output-dir results/smoke/future_morphometry \
  --regression-target hippocampus_total_icv \
  --regression-target lateral_ventricles_total_icv \
  --feature covariates_morphometry \
  --feature jepa_only \
  --feature jepa_covariates_morphometry \
  --seed 20260623
```

If step 4 generated a fresh table, replace `--table-path` with
`results/smoke/horizon_matched_pairs.csv`.

Why: hippocampal and ventricular volume must be normalized by ICV and evaluated
at matched horizons, including persistence/carry-forward comparators.

### 5.5 MCI conversion within 36 months

```bash
python scripts/train_neurojepa_downstream_probe.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --output-dir results/smoke/mci_conversion_36m \
  --task mci_conversion_36m \
  --conversion-window-months 36 \
  --stable-min-followup-months 36 \
  --mci-anchor-policy first \
  --feature conventional_full \
  --feature neurojepa_conventional \
  --seed 20260623
```

Why: this is the fixed-horizon binary prognosis endpoint. Requiring 36 months
of follow-up prevents short-followed participants from being mislabeled stable.

### 5.6 First-MCI survival

```bash
python scripts/train_neurojepa_survival_probe.py \
  --manifest-path "$AUDIT_MANIFEST" \
  --embedding-index-path "$AUDIT_EMBEDDING_INDEX" \
  --output-dir results/smoke/first_mci_survival \
  --mci-anchor-policy first \
  --model cox_elastic_net \
  --feature covariates \
  --feature pooled \
  --include-covariates \
  --seed 20260623
```

Why: survival analysis retains late conversions and right censoring instead of
reducing follow-up to a binary 36-month label. The first MCI visit is the primary
anchor; `--mci-anchor-policy all` belongs only in a sensitivity analysis.

## 6. Locked repeated nested CV after protocol freeze

The legacy `prepare_paper_replications.py` command implements a single
train/validation/test assignment and therefore **must not be used for the final
paper protocol**. Generate and validate the pooled-embedding inventory first:

```bash
python scripts/build_embedding_integrity_manifest.py
```

This command fails closed on duplicate or mismatched keys, absent files, shapes
other than `(768,)`, non-finite values, and an existing output directory. It
writes the index/source hashes plus one SHA-256 per pooled `.npy` file.

After protocol freeze and author sign-off, create the three-repeat, four-outer,
five-inner participant assignments with:

```bash
python scripts/prepare_nested_cv.py
```

Each participant is outer-test exactly once per repeat. Within every outer
development partition, each participant is inner-validation exactly once. The
generator checks disjointness and binds copies of all inputs to the protocol and
embedding-integrity hashes. For implementation-only smoke generation before
freeze, both `--allow-draft-protocol` and `--allow-dirty-worktree` are required.

The tabular classification, cognition, and future-morphometry runner consumes these
assignments, refits preprocessing independently inside each inner fold, selects
the locked hyperparameter by mean inner-fold performance, and refits on the
complete outer-development partition:

```bash
python scripts/run_nested_tabular_probe.py \
  --task ad_vs_cn \
  --feature-group conventional_full \
  --feature-group neurojepa_conventional
```

For future cognition, add `--task future_cognition`, the source column through
`--target`, and the locked horizon through `--horizon-months`. Future morphometry
uses `--task future_morphometry` with targets such as
`hippocampus_total_icv`. The first-MCI Cox runner is separate:

```bash
python scripts/run_nested_survival_probe.py \
  --feature-group conventional_full \
  --feature-group neurojepa_conventional
```

Before any experiment, generate the outcome-blind cohort audit:

```bash
python scripts/audit_nested_cohorts.py
```

After every core endpoint has all outer-fold predictions, synthesize paired
effects, correlated-repeat bootstrap intervals, fold/repeat dispersion, and the
global BH family with:

```bash
python scripts/report_nested_paired_results.py \
  --spec-path configs/nested_paired_report.example.json \
  --output-dir results/paper_neurojepa_feature_audit/paired_report
```

The complete resumable matrix contains 20 tabular endpoint/horizon commands plus
first-MCI survival, each across all six feature groups. Inspect it without
executing anything:

```bash
python scripts/run_locked_experiment_matrix.py \
  --input-dir results/paper_neurojepa_feature_audit/locked_inputs \
  --output-dir results/paper_neurojepa_feature_audit/experiments
```

Only after every preflight gate passes, add `--execute`. Completed fold outputs
are verified and skipped on resume; incomplete directories stop the matrix rather
than being overwritten.

The final read-only gate is:

```bash
python scripts/preflight_locked_run.py \
  --input-dir results/paper_neurojepa_feature_audit/locked_inputs \
  --cohort-audit-path results/paper_neurojepa_feature_audit/cohort_audit.json
```

It must report `"ready": true`; a draft protocol, missing author sign-off, dirty
or unborn Git revision, stale input hash, invalid embedding inventory, or
unreconciled cohort flow blocks execution.

Do not launch these commands without development flags until the protocol is
frozen. During development, restrict smoke runs to one repeat and one outer fold
and pass `--allow-draft-protocol`.

No locked analysis command is currently authorized.

## 7. Do not launch confirmatory paper runs yet

The repository is not yet at protocol-freeze status. Confirmatory runs require
all of the following first:

1. Verify the end-to-end command matrix from a clean committed worktree.
2. Freeze `configs/protocol.json` after named author sign-off and archive its hash.
3. Regenerate the design-only cohort/precision audit and all immutable inputs
   from the frozen protocol and clean commit.

Until those items are complete, `results/smoke/` is implementation evidence
only. Final numerical claims must come exclusively from the three immutable
repeated nested-CV runs under `results/paper_neurojepa_feature_audit/`.

## Output checks after every script

Before starting the next stage, check that:

- the command exited successfully and its summary JSON is readable;
- train, validation, and test all contain participants and required classes or
  events;
- no participant appears in more than one split;
- selected features and source columns match the intended task;
- prediction row counts match the recorded task table;
- no ADNI identifiers, images, or participant-level embeddings are staged for
  Git.
