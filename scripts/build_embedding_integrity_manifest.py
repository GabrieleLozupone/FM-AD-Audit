#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.embedding_manifest import build_embedding_integrity_manifest
from neurojepa_adni_audit.paths import DEFAULT_EMBEDDING_INDEX_PATH, DEFAULT_MANIFEST_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and hash all pooled Neuro-JEPA embeddings.")
    parser.add_argument("--embedding-index-path", type=Path, default=DEFAULT_EMBEDDING_INDEX_PATH)
    parser.add_argument("--source-manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "embedding_integrity_manifest",
    )
    parser.add_argument("--expected-dimension", type=int, default=768)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = build_embedding_integrity_manifest(
        embedding_index_path=args.embedding_index_path,
        source_manifest_path=args.source_manifest_path,
        output_dir=args.output_dir,
        expected_dimension=args.expected_dimension,
    )
    print(f"Wrote immutable embedding integrity manifest: {output}")


if __name__ == "__main__":
    main()
