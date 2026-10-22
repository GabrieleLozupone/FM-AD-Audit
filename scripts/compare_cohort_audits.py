#!/usr/bin/env python3
"""Require an encoder's cohort audit to equal the Neuro-JEPA one (protocol 0.6.0 invariants).

The audit is design-only (counts, outcomes and missingness per task, repeat and fold),
so for identical data and splits the two JSON documents must match exactly. Only
`protocol_sha256` may differ. Exits non-zero and lists every difference otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IGNORED_TOP_LEVEL = {"protocol_sha256"}


def differences(reference: Any, candidate: Any, path: str = "$") -> list[str]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        found = []
        for key in sorted(set(reference) | set(candidate)):
            if path == "$" and key in IGNORED_TOP_LEVEL:
                continue
            if key not in reference or key not in candidate:
                found.append(f"{path}.{key}: present in only one audit")
            else:
                found.extend(differences(reference[key], candidate[key], f"{path}.{key}"))
        return found
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [f"{path}: length {len(reference)} != {len(candidate)}"]
        found = []
        for index, (a, b) in enumerate(zip(reference, candidate)):
            found.extend(differences(a, b, f"{path}[{index}]"))
        return found
    return [] if reference == candidate else [f"{path}: {reference!r} != {candidate!r}"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", type=Path, nargs="+", help="Encoder cohort audit JSON file(s).")
    parser.add_argument(
        "--reference", type=Path, default=PROJECT_ROOT / "results/paper_neurojepa_feature_audit/cohort_audit.json"
    )
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    failed = False
    for path in args.candidate:
        found = differences(reference, json.loads(path.read_text(encoding="utf-8")))
        tasks = len(reference.get("tasks", []))
        if found:
            failed = True
            print(f"DIFFERENT {path}: {len(found)} difference(s)")
            for line in found[:50]:
                print(f"  {line}")
        else:
            print(f"IDENTICAL {path} ({tasks} tasks; only protocol_sha256 ignored)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
