#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.replications import runtime_provenance, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only preflight for locked confirmatory runs.")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit-path", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol_path)
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("protocol_frozen", protocol.status == "frozen", f"status={protocol.status}")
    check(
        "author_signoff",
        bool(protocol.payload.get("approved_by")) and bool(protocol.payload.get("frozen_at")),
        f"approved_by={protocol.payload.get('approved_by', [])}",
    )
    runtime = runtime_provenance(PROJECT_ROOT)
    check("git_commit_available", runtime["git_commit"] is not None, str(runtime["git_commit"]))
    check("git_worktree_clean", not runtime["git_status_porcelain"], str(runtime["git_status_porcelain"]))

    provenance_path = args.input_dir / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        check(
            "input_protocol_hash",
            provenance.get("protocol", {}).get("sha256") == protocol.sha256,
            str(provenance.get("protocol", {}).get("sha256")),
        )
        check(
            "inputs_generated_from_frozen_protocol",
            provenance.get("protocol", {}).get("status") == "frozen",
            str(provenance.get("protocol", {}).get("status")),
        )
        integrity = provenance.get("embedding_integrity", {})
        check("embedding_integrity_valid", integrity.get("status") == "valid", str(integrity.get("status")))
        records_path = args.input_dir / "pooled_embeddings_integrity.csv"
        expected_hash = integrity.get("pooled_file_manifest", {}).get("sha256")
        check(
            "embedding_manifest_hash",
            records_path.exists() and sha256_file(records_path) == expected_hash,
            str(expected_hash),
        )
    else:
        check("nested_input_provenance", False, f"missing {provenance_path}")

    if args.cohort_audit_path.exists():
        audit = json.loads(args.cohort_audit_path.read_text())
        flows = [task.get("cohort_flow", {}) for task in audit.get("tasks", [])]
        check("audit_protocol_hash", audit.get("protocol_sha256") == protocol.sha256, str(audit.get("protocol_sha256")))
        check(
            "audit_is_design_only",
            audit.get("design_only") is True
            and audit.get("comparative_test_performance_computed") is False,
            "no comparative performance",
        )
        check(
            "cohort_flows_reconciled",
            bool(flows) and all(flow.get("reconciled") is True for flow in flows),
            f"flows={len(flows)}",
        )
    else:
        check("cohort_audit", False, f"missing {args.cohort_audit_path}")

    passed = all(item["passed"] for item in checks)
    print(json.dumps({"ready": passed, "checks": checks}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
