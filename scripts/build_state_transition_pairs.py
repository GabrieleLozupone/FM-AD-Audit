#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.transitions.state import (
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_HORIZONS,
    DEFAULT_PROGRESSION_PAIRS_PATH,
    DEFAULT_STATE_TRANSITION_PATH,
    DEFAULT_STATE_TRANSITION_SUMMARY_PATH,
    PAIR_POLICIES,
    build_run,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NeuroSTAR model-ready state transition pairs.")
    parser.add_argument("--progression-pairs-path", type=Path, default=DEFAULT_PROGRESSION_PAIRS_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_STATE_TRANSITION_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_STATE_TRANSITION_SUMMARY_PATH)
    parser.add_argument(
        "--pair-policy",
        choices=PAIR_POLICIES,
        default="adjacent",
        help="Transition-pair generation policy. Default preserves the original adjacent observed pairs.",
    )
    parser.add_argument(
        "--min-delta-months",
        type=float,
        default=6.0,
        help="Minimum current-to-future gap for generated non-adjacent pairs.",
    )
    parser.add_argument(
        "--max-delta-months",
        type=float,
        default=None,
        help="Optional maximum current-to-future gap for generated non-adjacent pairs.",
    )
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="*",
        default=DEFAULT_HORIZONS,
        help="Requested horizons for --pair-policy horizon_matched.",
    )
    parser.add_argument(
        "--horizon-tolerance-months",
        type=float,
        default=3.0,
        help="Allowed absolute mismatch around each requested horizon for horizon_matched pairs.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_run(parse_args())
