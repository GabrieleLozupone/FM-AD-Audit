#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.evaluation.calibration import gaussian_smooth_calibration
from neurojepa_adni_audit.features.groups import add_morphometry_ratios, uses_neurojepa
from neurojepa_adni_audit.nested_cv import validate_nested_fold_assignments
from neurojepa_adni_audit.nested_training import participant_equal_weights
from neurojepa_adni_audit.nonlinear_training import (
    MLPTrainingSpec,
    resolve_device,
    run_nested_mlp_fold,
)
from neurojepa_adni_audit.probes.cognitive import (
    load_cognitive_task_table,
    load_future_change_task_table,
)
from neurojepa_adni_audit.probes.downstream import (
    load_pooled_matrix,
    load_task_table,
    validate_feature_paths,
)
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.tables.pairs import enrich_horizon_pairs
from neurojepa_adni_audit.transitions.future import make_regression_values
from neurojepa_adni_audit.transitions.state import load_matrix


AUDIT_ROOT = PROJECT_ROOT / "results" / "paper_neurojepa_feature_audit"
DEFAULT_INPUT_DIR = AUDIT_ROOT / "locked_inputs"
DEFAULT_OUTPUT_DIR = AUDIT_ROOT / "nonlinear_experiments"
COMPLETE_FILES = (
    "summary.json",
    "outer_test_predictions.csv",
    "selected_inner_oof_predictions.csv",
    "outer_test_calibration_smooth.csv",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked nested-CV PyTorch MLP analysis for one tabular endpoint."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--target")
    parser.add_argument("--horizon-months", type=float, default=12.0)
    parser.add_argument("--feature-group", action="append")
    parser.add_argument("--repeat", action="append")
    parser.add_argument("--outer-fold", type=int, action="append")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    return parser.parse_args(argv)


def _load_summary(run_dir: Path) -> dict:
    value = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Completed summary must be an object: {run_dir / 'summary.json'}")
    return value


def _verify_inputs(input_dir: Path, protocol: object) -> str:
    provenance_path = input_dir / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Nested-CV provenance not found: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_hash = provenance.get("protocol", {}).get("sha256")
    protocol.require_artifact_protocol(source_hash, "nested_cv_inputs")
    return str(source_hash)


def _locked_endpoint(protocol: object, task: str, endpoint_target: str) -> dict:
    matches = [
        endpoint
        for endpoint in protocol.payload["nonlinear_decoder_analysis"]["endpoints"]
        if endpoint["task"] == task and endpoint["target"] == endpoint_target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{task}/{endpoint_target} is not one of the ten locked nonlinear endpoints."
        )
    return matches[0]


def _training_spec(protocol: object) -> tuple[MLPTrainingSpec, list[float]]:
    nonlinear = protocol.payload["nonlinear_decoder_analysis"]
    architecture = nonlinear["architecture"]
    optimization = nonlinear["optimization"]
    hidden = architecture["hidden_layers"]
    if len(hidden) != 1:
        raise ValueError("The locked nonlinear implementation requires one hidden layer.")
    spec = MLPTrainingSpec(
        hidden_units=int(hidden[0]),
        dropout=float(architecture["dropout"]),
        learning_rate=float(optimization["learning_rate"]),
        batch_size=int(optimization["batch_size"]),
        epochs=int(optimization["epochs"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
    )
    return spec, [float(value) for value in optimization["weight_decay_grid"]]


def _prepare_task(args: argparse.Namespace, manifest_path: Path, embedding_index_path: Path):
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
                tolerance_months=3.0,
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
        repeated_visits = False
    endpoint_target = args.target if args.target is not None else target
    if args.task in {"future_cognition", "future_morphometry"}:
        endpoint_target = f"{endpoint_target}_{args.horizon_months:g}m".replace(".0m", "m")
    if args.task != "future_morphometry":
        table = add_morphometry_ratios(table)
    return table, target, endpoint_target, kind, repeated_visits


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be positive.")
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    source_protocol_hash = _verify_inputs(args.input_dir, protocol)
    manifest_path = args.input_dir / "manifest.csv"
    embedding_index_path = args.input_dir / "embedding_index.csv"
    assignments = pd.read_csv(
        args.input_dir / "nested_fold_assignments.csv",
        dtype={"participant_id": "string"},
        low_memory=False,
    )
    validate_nested_fold_assignments(assignments, protocol)
    table, target, endpoint_target, kind, repeated_visits = _prepare_task(
        args, manifest_path, embedding_index_path
    )
    endpoint = _locked_endpoint(protocol, args.task, endpoint_target)
    nonlinear = protocol.payload["nonlinear_decoder_analysis"]
    locked_features = [
        *nonlinear["required_feature_groups"],
        *nonlinear["descriptive_feature_groups"],
    ]
    features = args.feature_group or locked_features
    if len(features) != len(set(features)) or not set(features).issubset(set(locked_features)):
        raise ValueError(f"Feature groups must be unique and locked: {locked_features}")
    if any(uses_neurojepa(feature) for feature in features):
        if args.task == "future_morphometry":
            table = table[
                table["current_pooled_embedding_path"].notna()
                & table["current_pooled_embedding_path"].map(
                    lambda value: Path(str(value)).is_file()
                )
            ].reset_index(drop=True)
            if table.empty:
                raise ValueError("No future-morphometry rows have usable anchor embeddings.")
            pooled = load_matrix(table["current_pooled_embedding_path"])
        else:
            table = validate_feature_paths(table, "pooled")
            pooled = load_pooled_matrix(table)
    else:
        pooled = None

    spec, weight_decays = _training_spec(protocol)
    resolved_device = resolve_device(args.device)
    runtime = {
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if resolved_device == "cuda"
            else None
        ),
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "torch_threads": args.torch_threads,
    }
    repeats = args.repeat or [item.name for item in protocol.replications]
    outer_folds = args.outer_fold or list(
        range(int(protocol.payload["partitioning"]["outer_folds"]))
    )
    total = len(repeats) * len(outer_folds) * len(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    completed = 0
    print(
        f"MLP endpoint={endpoint['id']} rows={len(table)} "
        f"participants={table['participant_id'].nunique()} device={resolved_device} "
        f"runs={total} grid={weight_decays}",
        flush=True,
    )
    for repeat in repeats:
        replication = protocol.replication(repeat)
        for outer_fold in outer_folds:
            for feature_group in features:
                completed += 1
                run_dir = (
                    args.output_dir
                    / args.task
                    / endpoint_target
                    / feature_group
                    / repeat
                    / f"outer_{outer_fold}"
                )
                if run_dir.exists():
                    complete = all((run_dir / name).exists() for name in COMPLETE_FILES)
                    if args.resume and complete:
                        summary = _load_summary(run_dir)
                        summaries.append(summary)
                        print(
                            f"[{completed:03d}/{total:03d}] SKIP {repeat}/o{outer_fold} "
                            f"{feature_group}",
                            flush=True,
                        )
                        continue
                    raise FileExistsError(f"Refusing to overwrite nonlinear output: {run_dir}")
                print(
                    f"[{completed:03d}/{total:03d}] START {repeat}/o{outer_fold} "
                    f"{feature_group}",
                    flush=True,
                )
                started = time.monotonic()
                summary, predictions, inner_oof = run_nested_mlp_fold(
                    table=table,
                    assignments=assignments,
                    full_pooled_matrix=pooled,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    task=args.task,
                    target=target,
                    feature_group=feature_group,
                    kind=kind,
                    weight_decays=weight_decays,
                    training_spec=spec,
                    seed=replication.split_seed,
                    repeated_visits=repeated_visits,
                    device=resolved_device,
                    torch_threads=args.torch_threads,
                )
                summary.update(
                    {
                        "endpoint_id": endpoint["id"],
                        "endpoint_target": endpoint_target,
                        "protocol_sha256": protocol.sha256,
                        "source_protocol_sha256": source_protocol_hash,
                        "manifest_path": str(manifest_path.resolve()),
                        "embedding_index_path": str(embedding_index_path.resolve()),
                        "runtime": runtime,
                    }
                )
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
                    calibration = gaussian_smooth_calibration(
                        calibration_truth,
                        calibration_prediction,
                        sample_weight=(
                            participant_equal_weights(predictions)
                            if repeated_visits
                            else None
                        ),
                    )
                run_dir.mkdir(parents=True)
                predictions.to_csv(run_dir / "outer_test_predictions.csv", index=False)
                inner_oof.to_csv(
                    run_dir / "selected_inner_oof_predictions.csv", index=False
                )
                calibration.to_csv(
                    run_dir / "outer_test_calibration_smooth.csv", index=False
                )
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                summaries.append(summary)
                print(
                    f"[{completed:03d}/{total:03d}] DONE  {repeat}/o{outer_fold} "
                    f"{feature_group} wd={summary['selected_weight_decay']:g} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    index_path = args.output_dir / f"{args.task}_{endpoint_target}_mlp_run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol.sha256,
                "source_protocol_sha256": source_protocol_hash,
                "endpoint_id": endpoint["id"],
                "runs": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
