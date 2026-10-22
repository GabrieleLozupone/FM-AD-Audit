#!/usr/bin/env python3
"""Build the locked nested-CV inputs for one additional encoder (protocol 0.6.0).

Copies the 0.4.0 locked inputs into results/fm_encoders/<representation>/locked_inputs/:
  - manifest, progression pairs, subjects and fold assignments byte for byte;
  - embedding_index.csv and horizon_matched_pairs.csv with only the encoder columns
    rewritten (pooled path -> encoder file; token columns blanked; model metadata);
  - the encoder's embedding integrity manifest.
Then loads every locked endpoint from both directories with the runners' loaders and
requires identical rows. Refuses to overwrite an existing output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.encoders.locked_inputs import REPRESENTATIONS, build_encoder_locked_inputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--representation", required=True, choices=REPRESENTATIONS)
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=PROJECT_ROOT / "results/paper_neurojepa_feature_audit/locked_inputs",
        help="0.4.0 locked inputs; read only.",
    )
    parser.add_argument("--fm-root", type=Path, default=PROJECT_ROOT / "results/fm_encoders")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <fm-root>/<representation>/locked_inputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = build_encoder_locked_inputs(
        representation=args.representation,
        parent_dir=args.parent_dir,
        encoder_root=args.fm_root / args.representation,
        protocol_path=args.protocol_path,
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
    )
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    loads = provenance["encoder_inputs"]["table_load_equivalence"]
    for label, record in loads.items():
        print(f"  {label:<45} rows={record['rows']:>6} participants={record['participants']:>5}  identical")
    print(f"Wrote {output_dir} ({len(loads)} endpoint loads equivalent to the 0.4.0 inputs)")


if __name__ == "__main__":
    main()
