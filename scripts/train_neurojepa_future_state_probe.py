#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.transitions.future import (
    CLASSIFICATION_TARGETS,
    DEFAULT_ALPHAS,
    DEFAULT_HORIZON_MATCHED_TABLE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STATE_COVARIATE_COLUMNS,
    DEFAULT_STATE_TRANSITION_PATH,
    FEATURE_MODES,
    REGRESSION_TARGETS,
    run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task F future clinical/anatomical state probes.")
    parser.add_argument(
        "--table-path",
        type=Path,
        default=DEFAULT_HORIZON_MATCHED_TABLE if DEFAULT_HORIZON_MATCHED_TABLE.exists() else DEFAULT_STATE_TRANSITION_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=None)
    parser.add_argument("--replication", type=str, default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--feature", action="append", choices=FEATURE_MODES, default=None)
    parser.add_argument("--regression-target", action="append", choices=sorted(REGRESSION_TARGETS), default=None)
    parser.add_argument("--classification-target", action="append", choices=sorted(CLASSIFICATION_TARGETS), default=None)
    parser.add_argument("--covariate-columns", nargs="*", default=DEFAULT_STATE_COVARIATE_COLUMNS)
    parser.add_argument("--alpha", type=float, nargs="*", default=DEFAULT_ALPHAS)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
