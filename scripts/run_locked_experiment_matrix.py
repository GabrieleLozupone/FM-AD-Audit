#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.features.groups import FEATURE_GROUPS
from neurojepa_adni_audit.protocol import load_protocol


def build_commands(
    *, protocol_path: Path, input_dir: Path, output_dir: Path, allow_draft: bool
) -> list[list[str]]:
    common = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_nested_tabular_probe.py"),
        "--protocol-path",
        str(protocol_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--resume",
    ]
    feature_arguments = [argument for feature in FEATURE_GROUPS for argument in ("--feature-group", feature)]
    if allow_draft:
        common.append("--allow-draft-protocol")
    specifications: list[list[str]] = [
        ["--task", "ad_vs_cn"],
        ["--task", "mci_conversion_36m"],
    ]
    specifications.extend(
        ["--task", "same_visit_cognition", "--target", target]
        for target in ("adas13_at_session", "adas11_at_session", "mmse_at_session")
    )
    specifications.extend(
        [
            "--task",
            "future_cognition",
            "--target",
            target,
            "--horizon-months",
            str(horizon),
        ]
        for target in ("adas13_at_session", "adas11_at_session", "mmse_at_session")
        for horizon in (12, 24, 36)
    )
    specifications.extend(
        [
            "--task",
            "future_morphometry",
            "--target",
            target,
            "--horizon-months",
            str(horizon),
        ]
        for target in ("hippocampus_total_icv", "lateral_ventricles_total_icv")
        for horizon in (12, 24, 36)
    )
    commands = [common + feature_arguments + specification for specification in specifications]
    survival = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_nested_survival_probe.py"),
        "--protocol-path",
        str(protocol_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--resume",
        *feature_arguments,
    ]
    if allow_draft:
        survival.append("--allow-draft-protocol")
    commands.append(survival)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan or execute the complete locked repeated nested-CV experiment matrix."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute; default is a read-only plan.")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(
        args.protocol_path,
        require_frozen=args.execute and not args.allow_draft_protocol,
    )
    provenance = json.loads((args.input_dir / "provenance.json").read_text())
    if provenance.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("Experiment inputs were generated from a different protocol hash.")
    commands = build_commands(
        protocol_path=args.protocol_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        allow_draft=args.allow_draft_protocol,
    )
    print(f"Experiment matrix: {len(commands)} endpoint commands × {len(FEATURE_GROUPS)} feature groups")
    for index, command in enumerate(commands, start=1):
        print(f"[{index:02d}/{len(commands):02d}] {shlex.join(command)}", flush=True)
        if args.execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not args.execute:
        print("Plan only. Re-run with --execute after protocol freeze and final approval.")


if __name__ == "__main__":
    main()
