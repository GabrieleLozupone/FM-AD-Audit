#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.evaluation.cohort_audit import summarize_nested_cohort
from neurojepa_adni_audit.evaluation.exclusions import (
    ad_vs_cn_flow,
    future_cognition_flow,
    future_morphometry_flow,
    mci_conversion_flow,
    same_visit_cognition_flow,
    survival_flow,
)
from neurojepa_adni_audit.features.groups import (
    add_morphometry_ratios,
    feature_columns,
    transition_feature_columns,
)
from neurojepa_adni_audit.nested_cv import validate_nested_fold_assignments
from neurojepa_adni_audit.probes.cognitive import (
    load_cognitive_task_table,
    load_future_change_task_tables,
)
from neurojepa_adni_audit.probes.downstream import load_task_table
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.survival.probe import load_survival_task_table
from neurojepa_adni_audit.tables.pairs import enrich_horizon_pairs
from neurojepa_adni_audit.transitions.future import make_regression_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an outcome-blind nested-CV cohort audit.")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "results/paper_nested_cv_inputs_enriched_draft"
    )
    parser.add_argument(
        "--output-path", type=Path, default=PROJECT_ROOT / "results/design_only_cohort_audit.json"
    )
    parser.add_argument("--allow-draft-protocol", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite cohort audit: {args.output_path}")
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    provenance = json.loads((args.input_dir / "provenance.json").read_text())
    if provenance.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("Cohort inputs were generated from a different protocol hash.")
    assignments = pd.read_csv(args.input_dir / "nested_fold_assignments.csv", low_memory=False)
    validate_nested_fold_assignments(assignments, protocol)
    manifest_path = args.input_dir / "manifest.csv"
    index_path = args.input_dir / "embedding_index.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    audits = []

    for task in ("ad_vs_cn", "mci_conversion_36m"):
        table = load_task_table(
            manifest_path,
            index_path,
            task,
            mci_anchor_policy="first" if task == "mci_conversion_36m" else "all",
        )
        table = add_morphometry_ratios(table)
        audit = summarize_nested_cohort(
                table=table,
                assignments=assignments,
                task_name=task,
                outcome_column="label",
                missing_columns=feature_columns(task, "conventional_full"),
            )
        audit["cohort_flow"] = (
            ad_vs_cn_flow(manifest, table)
            if task == "ad_vs_cn"
            else mci_conversion_flow(manifest, table)
        )
        audits.append(audit)

    survival = add_morphometry_ratios(
        load_survival_task_table(manifest_path, index_path, mci_anchor_policy="first")
    )
    survival_audit = summarize_nested_cohort(
            table=survival,
            assignments=assignments,
            task_name="first_mci_survival",
            outcome_column="event",
            missing_columns=feature_columns("first_mci_survival", "conventional_full"),
        )
    survival_audit["cohort_flow"] = survival_flow(manifest, survival)
    audits.append(survival_audit)

    for target in ("adas13_at_session", "adas11_at_session", "mmse_at_session"):
        same = add_morphometry_ratios(
            load_cognitive_task_table(manifest_path, index_path, target, "all_sessions")
        )
        same_audit = summarize_nested_cohort(
                table=same,
                assignments=assignments,
                task_name=f"same_visit_cognition:{target}",
                outcome_column=None,
                missing_columns=feature_columns("same_visit_cognition", "conventional_full"),
            )
        same_audit["cohort_flow"] = same_visit_cognition_flow(manifest, same, target)
        audits.append(same_audit)
        future_tables = load_future_change_task_tables(
            manifest_path,
            index_path,
            target,
            "all_sessions",
            [12.0, 24.0, 36.0],
            3.0,
        )
        for horizon, raw_future in future_tables.items():
            future = add_morphometry_ratios(raw_future)
            future_audit = summarize_nested_cohort(
                    table=future,
                    assignments=assignments,
                    task_name=f"future_cognition:{target}:{horizon:g}m",
                    outcome_column=None,
                    missing_columns=feature_columns("future_cognition", "conventional_full"),
                )
            future_audit["cohort_flow"] = future_cognition_flow(
                manifest, future, target
            )
            audits.append(future_audit)

    pairs = enrich_horizon_pairs(
        pd.read_csv(args.input_dir / "horizon_matched_pairs.csv", low_memory=False), manifest
    )
    for target in ("hippocampus_total_icv", "lateral_ventricles_total_icv"):
        for horizon in (12.0, 24.0, 36.0):
            selected = pairs[
                pd.to_numeric(pairs["requested_horizon_months"], errors="coerce") == horizon
            ].copy()
            eligible = add_morphometry_ratios(
                make_regression_values(selected, target), prefix="current_"
            )
            morphometry_audit = summarize_nested_cohort(
                    table=eligible,
                    assignments=assignments,
                    task_name=f"future_morphometry:{target}:{horizon:g}m",
                    outcome_column=None,
                    missing_columns=transition_feature_columns(
                        "future_morphometry", "conventional_full"
                    ),
                )
            morphometry_audit["cohort_flow"] = future_morphometry_flow(
                manifest, selected, eligible
            )
            audits.append(morphometry_audit)

    payload = {
        "schema_version": "1.0",
        "protocol_sha256": protocol.sha256,
        "design_only": True,
        "comparative_test_performance_computed": False,
        "tasks": audits,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote design-only cohort audit: {args.output_path}")


if __name__ == "__main__":
    main()
