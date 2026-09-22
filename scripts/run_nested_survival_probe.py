#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.features.groups import FEATURE_GROUPS, add_morphometry_ratios, uses_neurojepa
from neurojepa_adni_audit.nested_cv import validate_nested_fold_assignments
from neurojepa_adni_audit.probes.downstream import load_pooled_matrix, validate_feature_paths
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.survival.nested import run_nested_survival_fold
from neurojepa_adni_audit.survival.probe import load_survival_task_table


DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "paper_nested_cv_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested first-MCI penalized Cox probes.")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/nested_survival")
    parser.add_argument("--feature-group", action="append", choices=FEATURE_GROUPS, default=None)
    parser.add_argument("--repeat", action="append", default=None)
    parser.add_argument("--outer-fold", type=int, action="append", default=None)
    parser.add_argument("--allow-draft-protocol", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol_path, require_frozen=not args.allow_draft_protocol)
    provenance = json.loads((args.input_dir / "provenance.json").read_text())
    protocol.require_artifact_protocol(
        provenance.get("protocol", {}).get("sha256"), "nested_cv_inputs"
    )
    assignments = pd.read_csv(args.input_dir / "nested_fold_assignments.csv", low_memory=False)
    validate_nested_fold_assignments(assignments, protocol)
    table = load_survival_task_table(
        args.input_dir / "manifest.csv", args.input_dir / "embedding_index.csv", mci_anchor_policy="first"
    )
    table = add_morphometry_ratios(table)
    features = args.feature_group or list(FEATURE_GROUPS)
    if any(uses_neurojepa(feature) for feature in features):
        table = validate_feature_paths(table, "pooled")
        pooled = load_pooled_matrix(table)
    else:
        pooled = None
    repeats = args.repeat or [item.name for item in protocol.replications]
    outer_folds = args.outer_fold or list(range(int(protocol.payload["partitioning"]["outer_folds"])))
    horizons = [float(value) for value in protocol.payload["horizons"]["survival_evaluation_months"]]
    for repeat in repeats:
        for outer_fold in outer_folds:
            for feature in features:
                run_dir = (
                    args.output_dir
                    / "first_mci_survival"
                    / "time_to_ad"
                    / feature
                    / repeat
                    / f"outer_{outer_fold}"
                )
                if run_dir.exists():
                    complete = all(
                        (run_dir / filename).exists()
                        for filename in (
                            "summary.json",
                            "outer_test_predictions.csv",
                            "inner_model_selection.csv",
                            "outer_test_calibration.csv",
                            "development_scaled_schoenfeld_residuals.csv",
                        )
                    )
                    if args.resume and complete:
                        print(f"SKIP complete output={run_dir}", flush=True)
                        continue
                    raise FileExistsError(f"Refusing to overwrite nested survival output: {run_dir}")
                print(f"START survival feature={feature} repeat={repeat} outer={outer_fold}", flush=True)
                summary, predictions, history, calibration, schoenfeld = run_nested_survival_fold(
                    table=table,
                    assignments=assignments,
                    full_pooled_matrix=pooled,
                    repeat=repeat,
                    outer_fold=outer_fold,
                    feature_group=feature,
                    penalizers=[float(value) for value in protocol.payload["cox_penalizers"]],
                    l1_ratios=[float(value) for value in protocol.payload["cox_l1_ratios"]],
                    horizons=horizons,
                    log=lambda message: print(
                        f"SURVIVAL feature={feature} repeat={repeat} outer={outer_fold} {message}",
                        flush=True,
                    ),
                    show_fit_progress=True,
                )
                summary["protocol_sha256"] = protocol.sha256
                run_dir.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(prefix=f".{run_dir.name}_", dir=run_dir.parent)
                )
                try:
                    predictions.to_csv(temporary / "outer_test_predictions.csv", index=False)
                    history.to_csv(temporary / "inner_model_selection.csv", index=False)
                    calibration.to_csv(temporary / "outer_test_calibration.csv", index=False)
                    schoenfeld.to_csv(
                        temporary / "development_scaled_schoenfeld_residuals.csv", index=False
                    )
                    (temporary / "summary.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                    temporary.rename(run_dir)
                except Exception:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise
                print(f"DONE output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
