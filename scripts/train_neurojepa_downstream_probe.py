#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.probes.downstream import (
    DEFAULT_COVARIATE_COLUMNS,
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_LOGISTIC_C_VALUES,
    DEFAULT_OUTPUT_DIR,
    run,
)
from neurojepa_adni_audit.features.progression_aligned import STAGE1_FEATURE_MODES
from neurojepa_adni_audit.features.groups import FEATURE_GROUPS


STAGE1_COVARIATE_FEATURE_MODES = tuple(f"{feature}_covariates" for feature in STAGE1_FEATURE_MODES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Neuro-JEPA downstream probes on frozen ADNI embeddings."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=None)
    parser.add_argument("--replication", type=str, default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--task", choices=["ad_vs_cn", "mci_conversion_36m"], default="ad_vs_cn")
    parser.add_argument(
        "--conversion-window-months",
        type=float,
        default=36.0,
        help="For mci_conversion_36m, label MCI anchor visits as converters if AD appears within this many months.",
    )
    parser.add_argument(
        "--stable-min-followup-months",
        type=float,
        default=36.0,
        help="For mci_conversion_36m, require this much observed follow-up after an MCI anchor before calling a never-AD subject stable.",
    )
    parser.add_argument(
        "--allow-short-stable-followup",
        action="store_true",
        help="For mci_conversion_36m, keep never-AD MCI anchors even when observed follow-up is shorter than --stable-min-followup-months.",
    )
    parser.add_argument(
        "--mci-anchor-policy",
        choices=["all", "first"],
        default="all",
        help="For mci_conversion_36m, use all eligible MCI anchor visits or only the first eligible anchor per subject.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        choices=[
            "pooled",
            "tokens_attn",
            "covariates",
            *FEATURE_GROUPS,
            *STAGE1_FEATURE_MODES,
            *STAGE1_COVARIATE_FEATURE_MODES,
        ],
        default=None,
        help="Canonical paper feature group or legacy feature mode. Can be repeated. Default: pooled.",
    )
    parser.add_argument("--include-covariates", action="store_true")
    parser.add_argument("--covariate-columns", nargs="*", default=DEFAULT_COVARIATE_COLUMNS)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument(
        "--logistic-c",
        type=float,
        nargs="*",
        default=DEFAULT_LOGISTIC_C_VALUES,
        help="Validation-selected inverse L2 regularization grid for logistic probes.",
    )
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=["auroc", "auprc", "balanced_accuracy", "loss"],
        default="auprc",
        help="Validation metric used to select the attentive-head checkpoint.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "reduce_on_plateau", "cosine"],
        default="reduce_on_plateau",
        help="Learning-rate schedule for attentive-head training.",
    )
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Default: cuda if available else cpu.")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="PCI bus-ID ordered GPU index to expose for attentive-head training.",
    )
    parser.add_argument("--list-gpus", action="store_true", help="List GPUs in PCI bus-ID order and exit.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
