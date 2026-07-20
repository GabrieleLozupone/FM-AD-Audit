#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.evaluation.reporting import run_paired_report
from neurojepa_adni_audit.protocol import load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired participant-bootstrap paper effects and FDR.")
    parser.add_argument("--spec-path", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs" / "protocol.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-draft-protocol",
        action="store_true",
        help="Permit reporter testing with a draft protocol. Never use for paper results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    frame, _metadata = run_paired_report(spec_path=args.spec_path, protocol=protocol, output_dir=args.output_dir)
    print(frame.to_string(index=False))
    print(f"Wrote paired report to {args.output_dir}")


if __name__ == "__main__":
    main()
