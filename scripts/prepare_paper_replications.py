#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.paths import (
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_HORIZON_PAIRS_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROGRESSION_PAIRS_PATH,
)
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.replications import prepare_replication


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create immutable paper-specific manifests and pair tables for each replication."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs" / "protocol.json")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--progression-pairs-path", type=Path, default=DEFAULT_PROGRESSION_PAIRS_PATH)
    parser.add_argument("--horizon-pairs-path", type=Path, default=DEFAULT_HORIZON_PAIRS_PATH)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_neurojepa_feature_audit",
    )
    parser.add_argument("--replication", action="append", default=None, help="Replication name; repeatable.")
    parser.add_argument(
        "--allow-draft-protocol",
        action="store_true",
        help="Permit artifact-generation testing while the protocol is draft. Never use for paper runs.",
    )
    parser.add_argument(
        "--allow-dirty-worktree",
        action="store_true",
        help="Permit generator testing without a clean commit. Never use for paper runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    names = args.replication or [item.name for item in protocol.replications]
    for name in names:
        replication = protocol.replication(name)
        output = prepare_replication(
            protocol=protocol,
            replication=replication,
            manifest_path=args.manifest_path,
            embedding_index_path=args.embedding_index_path,
            progression_pairs_path=args.progression_pairs_path,
            horizon_pairs_path=args.horizon_pairs_path,
            output_root=args.output_root,
            project_root=PROJECT_ROOT,
            allow_dirty_worktree=args.allow_dirty_worktree,
        )
        print(f"Prepared immutable replication inputs: {output}")


if __name__ == "__main__":
    main()
