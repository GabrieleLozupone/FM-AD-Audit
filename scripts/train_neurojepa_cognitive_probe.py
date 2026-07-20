#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.probes.cognitive import (
    DEFAULT_ALPHAS,
    DEFAULT_COGNITIVE_OUTPUT_DIR,
    DEFAULT_COVARIATE_COLUMNS,
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_FUTURE_HORIZONS,
    DEFAULT_MANIFEST_PATH,
    run,
)
from neurojepa_adni_audit.features.progression_aligned import STAGE1_FEATURE_MODES
from neurojepa_adni_audit.features.groups import FEATURE_GROUPS


STAGE1_COVARIATE_FEATURE_MODES = tuple(f"{feature}_covariates" for feature in STAGE1_FEATURE_MODES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ridge regression probes for cognitive severity and change.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COGNITIVE_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=None)
    parser.add_argument("--replication", type=str, default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument(
        "--task",
        choices=["same_visit", "future_change"],
        default="same_visit",
        help="Run D1 same-visit severity or D2 future cognitive change prediction.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Cognitive target column or alias. Aliases: mmse, adas11, adas13. Default: all three same-visit targets.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        choices=[
            "pooled",
            "pooled_covariates",
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
    parser.add_argument(
        "--horizon-months",
        type=float,
        nargs="*",
        default=DEFAULT_FUTURE_HORIZONS,
        help="For --task future_change, forecast windows in months. Default: 12 24 36.",
    )
    parser.add_argument(
        "--horizon-tolerance-months",
        type=float,
        default=3.0,
        help="For --task future_change, match the nearest future visit within this tolerance.",
    )
    parser.add_argument(
        "--row-policy",
        choices=["all_sessions", "first_session"],
        default="all_sessions",
        help="Use every target-available session or only each subject's first session.",
    )
    parser.add_argument("--alpha", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--seed", type=int, default=20260624)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
