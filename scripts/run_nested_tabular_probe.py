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

from neurojepa_adni_audit.features.groups import FEATURE_GROUPS, add_morphometry_ratios, uses_neurojepa
from neurojepa_adni_audit.evaluation.calibration import gaussian_smooth_calibration
from neurojepa_adni_audit.nested_cv import validate_nested_fold_assignments
from neurojepa_adni_audit.nested_training import participant_equal_weights, run_nested_tabular_fold
from neurojepa_adni_audit.probes.cognitive import (
    load_cognitive_task_table,
    load_future_change_task_table,
)
from neurojepa_adni_audit.probes.downstream import load_pooled_matrix, load_task_table, validate_feature_paths
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.tables.pairs import enrich_horizon_pairs
from neurojepa_adni_audit.transitions.future import make_regression_values
from neurojepa_adni_audit.transitions.state import load_matrix


DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "paper_nested_cv_inputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inner selection and outer-development refit for tabular nested-CV probes."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs" / "protocol.json")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--assignments-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--embedding-index-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "nested_tabular")
    parser.add_argument(
        "--task",
        choices=[
            "ad_vs_cn",
            "mci_conversion_36m",
            "same_visit_cognition",
            "future_cognition",
            "future_morphometry",
        ],
        required=True,
    )
    parser.add_argument("--target", default=None, help="Required for cognition tasks.")
    parser.add_argument("--horizon-months", type=float, default=12.0)
    parser.add_argument("--horizon-tolerance-months", type=float, default=3.0)
    parser.add_argument("--feature-group", action="append", choices=FEATURE_GROUPS, default=None)
    parser.add_argument("--repeat", action="append", default=None)
    parser.add_argument("--outer-fold", type=int, action="append", default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _verify_bound_inputs(args: argparse.Namespace, protocol: object) -> None:
    provenance_path = args.input_dir / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Nested-CV provenance not found: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    protocol.require_artifact_protocol(
        provenance.get("protocol", {}).get("sha256"), "nested_cv_inputs"
    )


def _load_completed_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load completed run summary: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Completed run summary must be a JSON object: {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    _verify_bound_inputs(args, protocol)
    assignments_path = args.assignments_path or args.input_dir / "nested_fold_assignments.csv"
    manifest_path = args.manifest_path or args.input_dir / "manifest.csv"
    embedding_index_path = args.embedding_index_path or args.input_dir / "embedding_index.csv"
    assignments = pd.read_csv(assignments_path, dtype={"participant_id": "string"}, low_memory=False)
    validate_nested_fold_assignments(assignments, protocol)

    if args.task in {"same_visit_cognition", "future_cognition", "future_morphometry"}:
        if not args.target:
            raise ValueError("--target is required for cognition and morphometry tasks.")
        if args.task == "same_visit_cognition":
            table = load_cognitive_task_table(
                manifest_path, embedding_index_path, args.target, row_policy="all_sessions"
            )
            target = args.target
        elif args.task == "future_cognition":
            table = load_future_change_task_table(
                manifest_path,
                embedding_index_path,
                args.target,
                row_policy="all_sessions",
                horizon_months=args.horizon_months,
                tolerance_months=args.horizon_tolerance_months,
            )
            target = "target_change"
        else:
            manifest = pd.read_csv(manifest_path, low_memory=False)
            pairs = pd.read_csv(args.input_dir / "horizon_matched_pairs.csv", low_memory=False)
            pairs = enrich_horizon_pairs(pairs, manifest)
            pairs = pairs[
                pd.to_numeric(pairs["requested_horizon_months"], errors="coerce")
                == float(args.horizon_months)
            ].copy()
            table = make_regression_values(pairs, args.target)
            table = add_morphometry_ratios(table, prefix="current_")
            target = "target_change"
        kind = "regression"
        grid = [float(value) for value in protocol.payload["ridge_alphas"]]
        repeated_visits = True
    else:
        table = load_task_table(
            manifest_path,
            embedding_index_path,
            args.task,
            mci_anchor_policy="first" if args.task == "mci_conversion_36m" else "all",
        )
        target = "label"
        kind = "classification"
        grid = [float(value) for value in protocol.payload["logistic_c_values"]]
        repeated_visits = False
    endpoint_target = args.target if args.target is not None else target
    if args.task in {"future_cognition", "future_morphometry"}:
        horizon_label = f"{args.horizon_months:g}m".replace(".", "p")
        endpoint_target = f"{endpoint_target}_{horizon_label}"
    if args.task != "future_morphometry":
        table = add_morphometry_ratios(table)

    features = args.feature_group or list(FEATURE_GROUPS)
    if any(uses_neurojepa(feature) for feature in features):
        if args.task == "future_morphometry":
            table = table[
                table["current_pooled_embedding_path"].notna()
                & table["current_pooled_embedding_path"].map(lambda value: Path(str(value)).is_file())
            ].reset_index(drop=True)
            if table.empty:
                raise ValueError("No future-morphometry rows have usable anchor embeddings.")
        else:
            table = validate_feature_paths(table, "pooled")
        pooled = (
            load_matrix(table["current_pooled_embedding_path"])
            if args.task == "future_morphometry"
            else load_pooled_matrix(table)
        )
    else:
        pooled = None

    repeats = args.repeat or [replication.name for replication in protocol.replications]
    outer_folds = args.outer_fold or list(range(int(protocol.payload["partitioning"]["outer_folds"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    for repeat in repeats:
        replication = protocol.replication(repeat)
        for outer_fold in outer_folds:
            for feature_group in features:
                run_dir = (
                    args.output_dir
                    / args.task
                    / endpoint_target
                    / feature_group
                    / repeat
                    / f"outer_{outer_fold}"
                )
                if run_dir.exists():
                    complete = all(
                        (run_dir / filename).exists()
                        for filename in (
                            "summary.json",
                            "outer_test_predictions.csv",
                            "selected_inner_oof_predictions.csv",
                            "outer_test_calibration_smooth.csv",
                        )
                    )
                    if args.resume and complete:
                        run_summaries.append(_load_completed_summary(run_dir))
                        print(f"SKIP complete output={run_dir}", flush=True)
                        continue
                    raise FileExistsError(f"Refusing to overwrite nested probe output: {run_dir}")
                print(
                    f"START task={args.task} target={target} feature={feature_group} "
                    f"repeat={repeat} outer={outer_fold}",
                    flush=True,
                )
                summary, predictions, inner_oof = run_nested_tabular_fold(
                    table=table,
                    assignments=assignments,
                    full_pooled_matrix=pooled,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    task=args.task,
                    target=target,
                    feature_group=feature_group,
                    kind=kind,
                    hyperparameters=grid,
                    seed=replication.split_seed,
                    repeated_visits=repeated_visits,
                )
                summary.update(
                    {
                        "endpoint_target": endpoint_target,
                        "protocol_sha256": protocol.sha256,
                        "manifest_path": str(manifest_path.resolve()),
                        "embedding_index_path": str(embedding_index_path.resolve()),
                    }
                )
                run_dir.mkdir(parents=True)
                predictions.to_csv(run_dir / "outer_test_predictions.csv", index=False)
                inner_oof.to_csv(run_dir / "selected_inner_oof_predictions.csv", index=False)
                if kind == "classification":
                    calibration = gaussian_smooth_calibration(
                        predictions["truth"].to_numpy(),
                        predictions["prediction"].to_numpy(),
                        probability=True,
                    )
                else:
                    calibration_truth = (
                        predictions["future_target_value"].to_numpy()
                        if "future_target_value" in predictions
                        else predictions["truth"].to_numpy()
                    )
                    calibration_prediction = (
                        predictions["predicted_future_value"].to_numpy()
                        if "predicted_future_value" in predictions
                        else predictions["prediction"].to_numpy()
                    )
                    calibration_weights = (
                        participant_equal_weights(predictions) if repeated_visits else None
                    )
                    calibration = gaussian_smooth_calibration(
                        calibration_truth,
                        calibration_prediction,
                        sample_weight=calibration_weights,
                    )
                calibration.to_csv(run_dir / "outer_test_calibration_smooth.csv", index=False)
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                run_summaries.append(summary)
                print(f"DONE output={run_dir}", flush=True)
    (args.output_dir / f"{args.task}_{endpoint_target}_run_index.json").write_text(
        json.dumps({"runs": run_summaries}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
