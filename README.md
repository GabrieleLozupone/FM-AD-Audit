# Neuro-JEPA ADNI Longitudinal Audit

Focused experimental repository for the paper:

> Do Frozen Neuroimaging Foundation Representations Add Diagnostic and
> Prognostic Value in Alzheimer Disease? A Longitudinal ADNI Audit

This repository was extracted from the private NeuroSTAR research workspace. It
contains only the frozen-feature experimental path needed for the paper:

- AD versus CN and 36-month MCI conversion classification;
- same-visit and future cognitive Ridge probes;
- horizon-matched future cognition and morphometry;
- first-MCI Cox survival analysis;
- prespecified conventional/Neuro-JEPA feature groups;
- participant-aware paired reporting.

Stage 1 progression alignment, generative modeling and Tasks G--J are outside the
paper scope.

## Current migration state

The repository currently links to the already generated NeuroSTAR derived tables
and pooled Neuro-JEPA embeddings. These links are local conveniences and are
ignored by Git. No ADNI data or participant-level embeddings should be committed.

The paper-facing API now includes canonical feature groups, immutable embedding
integrity checks, participant-level repeated nested CV, inner-fold preprocessing
and selection, complete outer-development refitting, participant-equal weights,
censoring-aware survival evaluation, calibration, and correlated-repeat paired
reporting and a reconciled exclusion-reason cohort flow. Protocol version 0.4.0
is frozen with named author approval; the remaining gate is the clean-worktree
end-to-end preflight over regenerated locked inputs.

## Local linked inputs

After running the migration script, the following symlinks are created:

```text
data/linked/manifest.csv
data/linked/embedding_index.csv
data/linked/embeddings
data/linked/progression_pairs.csv
data/linked/horizon_matched_pairs.csv
```

The embedding index contains absolute paths to the existing `.npy` files, so
embedding extraction is not required for the current local continuation.

Validate the links with:

```bash
bash scripts/check_linked_data.sh
```

## Installation

Create one Conda environment for this audit and the pinned Neuro-JEPA encoder:

```bash
conda create -n neurojepa-adni-audit -c conda-forge python=3.11 pip git -y
conda activate neurojepa-adni-audit
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pip install \
  "neurojepa @ git+https://github.com/NYUMedML/Neuro-JEPA.git@b8074740cd3e6ad72e5d032aa49617bdae327240"
```

The pinned Neuro-JEPA package installs its official Torch, MONAI, Safetensors,
TorchMetrics, and Hugging Face Hub dependencies. Confirm that the environment is
usable with:

```bash
python -c "import neurojepa, torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest
```

### Neuro-JEPA model access

The [Neuro-JEPA weights](https://huggingface.co/NYUMedML/Neuro-JEPA) are gated.
Request access with an institutional Hugging Face account, accept the model
terms, and authenticate without storing a token in this repository:

```bash
hf auth login
hf auth whoami
```

After approval, verify that the released backbone can be downloaded and loaded:

```bash
python - <<'PY'
import torch
from neurojepa.utils.init_utils import load_backbone_from_hf

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_backbone_from_hf("NYUMedML/Neuro-JEPA", device=device)
model.eval()
print(f"Neuro-JEPA loaded on {device}")
PY
```

Model weights are not redistributed by this repository. Keep the code commit and
Hugging Face model revision used for final extraction recorded in the frozen
paper protocol. A compatible NVIDIA driver is required separately for GPU use.

## Existing experimental entry points

See [RUNBOOK.md](RUNBOOK.md) for the ordered execution workflow, the reason for
each script, and the boundary between smoke tests and confirmatory paper runs.

```bash
python scripts/prepare_paper_replications.py --help
python scripts/build_embedding_integrity_manifest.py --help
python scripts/prepare_nested_cv.py --help
python scripts/run_nested_tabular_probe.py --help
python scripts/run_nested_survival_probe.py --help
python scripts/audit_nested_cohorts.py --help
python scripts/report_nested_paired_results.py --help
python scripts/run_locked_experiment_matrix.py --help
python scripts/preflight_locked_run.py --help
python scripts/train_neurojepa_downstream_probe.py --help
python scripts/train_neurojepa_cognitive_probe.py --help
python scripts/train_neurojepa_future_state_probe.py --help
python scripts/train_neurojepa_survival_probe.py --help
python scripts/report_paired_results.py --help
```

The classification and cognition entry points accept the six canonical paper
feature groups directly. For example, a smoke run of the primary same-visit
ADAS13 comparison is configured with:

```bash
python scripts/train_neurojepa_cognitive_probe.py \
  --manifest-path data/linked/manifest.csv \
  --embedding-index-path data/linked/embedding_index.csv \
  --task same_visit --target adas13 \
  --feature conventional_full --feature neurojepa_conventional
```

Canonical groups are `morphometry_only`, `clinical_demographic`,
`conventional_full`, `neurojepa_only`, `neurojepa_morphometry`, and
`neurojepa_conventional`. The probe records the resolved source columns in each
`summary.json`; morphometry uses explicit hippocampus/ICV and lateral-ventricle/
ICV ratios, and task-aware checks reject outcome columns before fitting.

Outputs belong under `results/` and are ignored by Git. The confirmatory runs
must use paper-specific copies of the manifest/pair tables; the linked historical
tables are for migration validation and smoke tests only.

## Reproducibility boundary

ADNI MRI, clinical exports, participant identifiers and participant-level
embeddings cannot be redistributed. A later milestone will add the public ADNI
input reconstruction, MRI preprocessing and morphometry pipeline. Until then,
the experimental code can be validated from the linked derived inputs and from
the synthetic tests.

## Licenses

This repository's original code is MIT licensed. Neuro-JEPA code is MIT licensed;
its released model weights use CC BY-NC-ND 4.0. Users must separately satisfy
ADNI and Neuro-JEPA access and usage terms.
