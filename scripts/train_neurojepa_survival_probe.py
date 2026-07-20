#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.survival.probe import (
    DEFAULT_COVARIATE_COLUMNS,
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_L1_RATIOS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PENALIZERS,
    DEFAULT_SURVIVAL_HORIZONS,
    DEFAULT_SURVIVAL_OUTPUT_DIR,
    run,
)
from neurojepa_adni_audit.features.progression_aligned import STAGE1_FEATURE_MODES
from neurojepa_adni_audit.features.groups import FEATURE_GROUPS


STAGE1_COVARIATE_FEATURE_MODES = tuple(f"{feature}_covariates" for feature in STAGE1_FEATURE_MODES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train survival probes for MCI time-to-AD conversion."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SURVIVAL_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=None)
    parser.add_argument("--replication", type=str, default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument(
        "--feature",
        action="append",
        choices=["pooled", "covariates", *FEATURE_GROUPS, *STAGE1_FEATURE_MODES, *STAGE1_COVARIATE_FEATURE_MODES],
        default=None,
        help="Canonical paper feature group or legacy feature mode. Can be repeated. Default: pooled.",
    )
    parser.add_argument("--include-covariates", action="store_true")
    parser.add_argument("--covariate-columns", nargs="*", default=DEFAULT_COVARIATE_COLUMNS)
    parser.add_argument(
        "--mci-anchor-policy",
        choices=["first", "all"],
        default="first",
        help="Use the first MCI anchor per subject for primary Task C, or all MCI anchors as a sensitivity run.",
    )
    parser.add_argument("--penalizer", type=float, nargs="*", default=DEFAULT_PENALIZERS)
    parser.add_argument("--l1-ratio", type=float, nargs="*", default=DEFAULT_L1_RATIOS)
    parser.add_argument(
        "--model",
        action="append",
        choices=["cox_elastic_net", "logistic_hazard", "deephit_single"],
        default=None,
        help="Survival model backend. Can be passed multiple times. Default: cox_elastic_net.",
    )
    parser.add_argument(
        "--num-durations",
        type=int,
        default=12,
        help="Number of discrete duration bins for pycox neural survival models.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[128],
        help="Hidden layer sizes for pycox neural survival models.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout for pycox neural survival models.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for pycox neural survival models.")
    parser.add_argument("--epochs", type=int, default=256, help="Maximum epochs for pycox neural survival models.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for pycox neural survival models.")
    parser.add_argument("--patience", type=int, default=24, help="Early stopping patience for pycox neural survival models.")
    parser.add_argument("--device", default="cpu", help="Torch device for pycox neural survival models, e.g. `cpu` or `cuda:0`.")
    parser.add_argument("--deephit-alpha", type=float, default=0.2, help="DeepHit likelihood/ranking loss mixture alpha.")
    parser.add_argument("--deephit-sigma", type=float, default=0.1, help="DeepHit ranking loss smoothing sigma.")
    parser.add_argument(
        "--survival-horizons",
        type=float,
        nargs="*",
        default=DEFAULT_SURVIVAL_HORIZONS,
        help="Month horizons for neural survival risk/survival outputs and time-dependent metrics.",
    )
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
