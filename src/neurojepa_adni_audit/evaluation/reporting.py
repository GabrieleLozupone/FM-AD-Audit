from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from neurojepa_adni_audit.evaluation.paired import (
    benjamini_hochberg,
    paired_subject_bootstrap_weighted,
)
from neurojepa_adni_audit.protocol import Protocol


LOWER_IS_BETTER = {"subject_equal_mae", "brier_score", "log_loss"}
HIGHER_IS_BETTER = {"auroc", "auprc"}
SUPPORTED_METRICS = LOWER_IS_BETTER | HIGHER_IS_BETTER


def _weighted_mae(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    error = np.abs(frame["target"].to_numpy(float) - frame[prediction].to_numpy(float))
    return float(np.average(error, weights=weights))


def _weighted_brier(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    error = (frame["target"].to_numpy(float) - frame[prediction].to_numpy(float)) ** 2
    return float(np.average(error, weights=weights))


def _weighted_log_loss(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    target = frame["target"].to_numpy(float)
    probability = np.clip(frame[prediction].to_numpy(float), 1e-7, 1.0 - 1e-7)
    losses = -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    return float(np.average(losses, weights=weights))


def _weighted_auroc(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    included = weights > 0
    if len(np.unique(frame.loc[included, "target"])) < 2:
        return float("nan")
    return float(
        roc_auc_score(
            frame["target"].to_numpy(), frame[prediction].to_numpy(), sample_weight=weights
        )
    )


def _weighted_auprc(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    included = weights > 0
    if len(np.unique(frame.loc[included, "target"])) < 2:
        return float("nan")
    return float(
        average_precision_score(
            frame["target"].to_numpy(), frame[prediction].to_numpy(), sample_weight=weights
        )
    )


METRICS: dict[str, Callable[[pd.DataFrame, str, np.ndarray], float]] = {
    "subject_equal_mae": _weighted_mae,
    "brier_score": _weighted_brier,
    "log_loss": _weighted_log_loss,
    "auroc": _weighted_auroc,
    "auprc": _weighted_auprc,
}


def _load_prediction_pair(spec: dict[str, Any]) -> pd.DataFrame:
    required = {
        "combined_predictions",
        "conventional_predictions",
        "pair_keys",
        "target_column",
        "combined_prediction_column",
        "conventional_prediction_column",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Comparison `{spec.get('id', '?')}` is missing: {', '.join(missing)}")
    keys = [str(value) for value in spec["pair_keys"]]
    if "participant_id" not in keys:
        raise ValueError("pair_keys must include `participant_id`.")
    combined = pd.read_csv(spec["combined_predictions"], low_memory=False)
    conventional = pd.read_csv(spec["conventional_predictions"], low_memory=False)
    split = str(spec.get("split", "test"))
    if "subject_split" in combined:
        combined = combined[combined["subject_split"].astype(str) == split].copy()
    if "subject_split" in conventional:
        conventional = conventional[conventional["subject_split"].astype(str) == split].copy()
    required_combined = set(keys) | {spec["target_column"], spec["combined_prediction_column"]}
    required_conventional = set(keys) | {spec["target_column"], spec["conventional_prediction_column"]}
    for label, table, columns in (
        ("combined", combined, required_combined),
        ("conventional", conventional, required_conventional),
    ):
        absent = sorted(columns - set(table.columns))
        if absent:
            raise ValueError(f"{label} predictions are missing columns: {', '.join(absent)}")
        if table.duplicated(keys).any():
            raise ValueError(f"{label} predictions contain duplicate pair keys.")

    left = combined[keys + [spec["target_column"], spec["combined_prediction_column"]]].rename(
        columns={spec["target_column"]: "target_combined", spec["combined_prediction_column"]: "combined"}
    )
    right = conventional[keys + [spec["target_column"], spec["conventional_prediction_column"]]].rename(
        columns={
            spec["target_column"]: "target_conventional",
            spec["conventional_prediction_column"]: "conventional",
        }
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(
            f"Prediction pairing is incomplete: combined={len(left)}, conventional={len(right)}, paired={len(paired)}"
        )
    if not np.allclose(paired["target_combined"], paired["target_conventional"], equal_nan=True):
        raise ValueError("Targets differ between paired prediction files.")
    paired["target"] = paired["target_combined"]
    return paired[[*keys, "target", "combined", "conventional"]]


def run_paired_report(
    *,
    spec_path: Path,
    protocol: Protocol,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = spec_path.read_bytes()
    payload = json.loads(raw)
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("Report specification must contain a non-empty `comparisons` list.")
    ids = [str(item.get("id")) for item in comparisons]
    if len(ids) != len(set(ids)):
        raise ValueError("Comparison IDs must be unique.")

    rows: list[dict[str, Any]] = []
    default_replicates = int(protocol.payload["bootstrap"]["replicates"])
    for comparison in comparisons:
        metric_name = str(comparison["metric"])
        if metric_name not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported paired metric `{metric_name}`.")
        replication = protocol.replication(str(comparison["replication"]))
        frame = _load_prediction_pair(comparison)
        result = paired_subject_bootstrap_weighted(
            frame,
            METRICS[metric_name],
            "combined",
            "conventional",
            replicates=int(comparison.get("bootstrap_replicates", default_replicates)),
            seed=int(comparison.get("bootstrap_seed", replication.split_seed)),
        )
        row = {
            "id": str(comparison["id"]),
            "family": str(comparison["family"]),
            "replication": replication.name,
            "analysis_role": replication.analysis_role,
            "metric": metric_name,
            "difference_definition": "combined-minus-conventional",
            "favorable_direction": "negative" if metric_name in LOWER_IS_BETTER else "positive",
            **asdict(result),
        }
        rows.append(row)

    result_frame = pd.DataFrame(rows)
    result_frame["fdr_adjusted_p_value"] = np.nan
    for _, indices in result_frame.groupby("family", sort=False).groups.items():
        positions = list(indices)
        adjusted = benjamini_hochberg(result_frame.loc[positions, "bootstrap_sign_p_value"].tolist())
        result_frame.loc[positions, "fdr_adjusted_p_value"] = adjusted

    metadata = {
        "protocol_sha256": protocol.sha256,
        "protocol_version": protocol.payload["protocol_version"],
        "protocol_status": protocol.status,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": hashlib.sha256(raw).hexdigest(),
        "difference_definition": "combined-minus-conventional",
        "ci_method": "paired participant bootstrap percentile interval",
        "p_value_method": "two-sided bootstrap sign probability with finite-sample correction",
        "fdr_method": "Benjamini-Hochberg within named family",
        "comparisons": result_frame.to_dict(orient="records"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_dir / "paired_effects.csv", index=False)
    (output_dir / "paired_effects.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result_frame, metadata
