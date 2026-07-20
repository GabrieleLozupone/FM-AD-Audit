#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.nested_cv import prepare_nested_cv_artifacts
from neurojepa_adni_audit.paths import (
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_HORIZON_PAIRS_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROGRESSION_PAIRS_PATH,
)
from neurojepa_adni_audit.protocol import load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable participant-level repeated nested-CV artifacts."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs" / "protocol.json")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--progression-pairs-path", type=Path, default=DEFAULT_PROGRESSION_PAIRS_PATH)
    parser.add_argument("--horizon-pairs-path", type=Path, default=DEFAULT_HORIZON_PAIRS_PATH)
    parser.add_argument(
        "--embedding-integrity-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "embedding_integrity_manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_nested_cv_inputs",
    )
    parser.add_argument(
        "--allow-draft-protocol",
        action="store_true",
        help="Permit implementation testing while the protocol is draft. Never use for paper runs.",
    )
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Permit implementation testing without a clean commit. Never use for paper runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    output = prepare_nested_cv_artifacts(
        protocol=protocol,
        manifest_path=args.manifest_path,
        embedding_index_path=args.embedding_index_path,
        embedding_integrity_dir=args.embedding_integrity_dir,
        progression_pairs_path=args.progression_pairs_path,
        horizon_pairs_path=args.horizon_pairs_path,
        output_dir=args.output_dir,
        project_root=PROJECT_ROOT,
        allow_dirty_worktree=args.allow_dirty_worktree,
    )
    print(f"Prepared immutable repeated nested-CV inputs: {output}")


if __name__ == "__main__":
    main()
