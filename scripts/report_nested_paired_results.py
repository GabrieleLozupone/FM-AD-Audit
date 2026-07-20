#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.evaluation.nested_reporting import run_nested_paired_report
from neurojepa_adni_audit.protocol import load_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description="Report paired effects across nested outer folds and repeats.")
    parser.add_argument("--spec-path", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    effects, _dispersion = run_nested_paired_report(
        spec_path=args.spec_path, protocol=protocol, output_dir=args.output_dir
    )
    print(f"Wrote {len(effects)} nested paired endpoint effect(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
