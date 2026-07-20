from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.probes.downstream import (
    DEFAULT_COVARIATE_COLUMNS,
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    first_baseline_rows,
    load_pooled_matrix,
    require_sklearn,
    split_rows,
    validate_feature_paths,
)
from neurojepa_adni_audit.features.covariates import validate_covariate_columns
from neurojepa_adni_audit.features.progression_aligned import STAGE1_FEATURE_MODES, is_stage1_feature
from neurojepa_adni_audit.features.groups import (
    add_morphometry_ratios,
    feature_columns,
    is_feature_group,
    uses_neurojepa,
    validate_no_target_leakage,
)
from neurojepa_adni_audit.features.tabular import build_pooled_or_covariate_feature_matrix
from neurojepa_adni_audit.metrics import compute_regression_metric
from neurojepa_adni_audit.sklearn_utils import require_regression_sklearn
from neurojepa_adni_audit.paper_run import bind_paper_run, require_protocol_grid


DEFAULT_COGNITIVE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "neurojepa_cognitive_probe"
DEFAULT_TARGETS = ["mmse_at_session", "adas11_at_session", "adas13_at_session"]
TARGET_ALIASES = {
    "mmse": "mmse_at_session",
    "adas11": "adas11_at_session",
    "adas13": "adas13_at_session",
}
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
DEFAULT_FUTURE_HORIZONS = [12.0, 24.0, 36.0]


def log_progress(message: str) -> None:
    print(message, flush=True)


def resolve_targets(targets: list[str] | None) -> list[str]:
    selected = targets or DEFAULT_TARGETS
    resolved = [TARGET_ALIASES.get(target.lower(), target) for target in selected]
    return list(dict.fromkeys(resolved))


def load_cognitive_task_table(
    manifest_path: Path,
    embedding_index_path: Path,
    target: str,
    row_policy: str,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"Embedding index not found: {embedding_index_path}. Finish Neuro-JEPA extraction first."
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    index = pd.read_csv(embedding_index_path, low_memory=False)
    if target not in manifest.columns:
        raise ValueError(f"Target column `{target}` not found in manifest.")
    if "image_id" not in manifest.columns or "image_id" not in index.columns:
        raise ValueError("Both manifest and embedding index must contain `image_id`.")

    table = manifest.copy()
    if row_policy == "first_session":
        table = first_baseline_rows(table)
    elif row_policy != "all_sessions":  # pragma: no cover
        raise ValueError(f"Unsupported row policy: {row_policy}")

    table[target] = pd.to_numeric(table[target], errors="coerce")
    table = table[table[target].notna()].copy()
    table = table.merge(
        index.drop_duplicates("image_id"),
        on=["participant_id", "session_id", "image_id"],
        how="inner",
        suffixes=("", "_embedding"),
    )
    if "subject_split" not in table.columns:
        raise ValueError("Merged cognitive task table does not contain `subject_split`.")
    table = table[table["subject_split"].isin(["train", "val", "test"])].copy()
    return table.reset_index(drop=True)


def find_future_visit(
    subject_rows: pd.DataFrame,
    anchor_index: int,
    target: str,
    horizon_months: float,
    tolerance_months: float,
) -> pd.Series | None:
    anchor_month = float(subject_rows.at[anchor_index, "session_month_numeric"])
    desired_month = anchor_month + float(horizon_months)
    future = subject_rows[
        (subject_rows["session_month_numeric"] > anchor_month)
        & subject_rows[target].notna()
        & ((subject_rows["session_month_numeric"] - desired_month).abs() <= float(tolerance_months))
    ].copy()
    if future.empty:
        return None
    future["horizon_abs_error"] = (future["session_month_numeric"] - desired_month).abs()
    future = future.sort_values(["horizon_abs_error", "session_month_numeric", "session_id"])
    return future.iloc[0]


def load_future_change_task_table(
    manifest_path: Path,
    embedding_index_path: Path,
    target: str,
    row_policy: str,
    horizon_months: float,
    tolerance_months: float,
) -> pd.DataFrame:
    return load_future_change_task_tables(
        manifest_path=manifest_path,
        embedding_index_path=embedding_index_path,
        target=target,
        row_policy=row_policy,
        horizons_months=[horizon_months],
        tolerance_months=tolerance_months,
    )[float(horizon_months)]


def load_future_change_task_tables(
    manifest_path: Path,
    embedding_index_path: Path,
    target: str,
    row_policy: str,
    horizons_months: list[float],
    tolerance_months: float,
) -> dict[float, pd.DataFrame]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"Embedding index not found: {embedding_index_path}. Finish Neuro-JEPA extraction first."
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    index = pd.read_csv(embedding_index_path, low_memory=False)
    if target not in manifest.columns:
        raise ValueError(f"Target column `{target}` not found in manifest.")
    required = {"participant_id", "session_id", "image_id", "session_month", "subject_split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    table = manifest.copy()
    table["session_month_numeric"] = pd.to_numeric(table["session_month"], errors="coerce")
    table[target] = pd.to_numeric(table[target], errors="coerce")
    table = table[table["session_month_numeric"].notna()].copy()
    table = table.sort_values(["participant_id", "session_month_numeric", "session_id"]).reset_index(drop=True)
    if row_policy == "first_session":
        anchor_candidates = first_baseline_rows(table)
    elif row_policy == "all_sessions":
        anchor_candidates = table
    else:  # pragma: no cover
        raise ValueError(f"Unsupported row policy: {row_policy}")
    anchor_keys = set(
        zip(
            anchor_candidates["participant_id"].astype(str),
            anchor_candidates["session_id"].astype(str),
            anchor_candidates["image_id"].astype(str),
        )
    )

    horizons = list(dict.fromkeys(float(value) for value in horizons_months))
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Future cognition horizons must be positive.")
    rows_by_horizon: dict[float, list[dict[str, Any]]] = {horizon: [] for horizon in horizons}
    for _participant_id, subject_rows in table.groupby("participant_id", sort=False):
        subject_rows = subject_rows.sort_values(["session_month_numeric", "session_id"]).reset_index(drop=True)
        records = subject_rows.to_dict(orient="records")
        months = subject_rows["session_month_numeric"].to_numpy(dtype=float)
        target_values = subject_rows[target].to_numpy(dtype=float)
        for anchor_index, anchor in enumerate(records):
            anchor_key = (
                str(anchor["participant_id"]),
                str(anchor["session_id"]),
                str(anchor["image_id"]),
            )
            if anchor_key not in anchor_keys or not np.isfinite(target_values[anchor_index]):
                continue
            future_mask = (months > months[anchor_index]) & np.isfinite(target_values)
            if not future_mask.any():
                continue
            for horizon_months in horizons:
                desired_month = months[anchor_index] + horizon_months
                errors = np.abs(months - desired_month)
                candidate_indices = np.flatnonzero(future_mask & (errors <= float(tolerance_months)))
                if not len(candidate_indices):
                    continue
                future_index = int(candidate_indices[np.argmin(errors[candidate_indices])])
                future = records[future_index]
                row = dict(anchor)
                current_value = float(target_values[anchor_index])
                future_value = float(target_values[future_index])
                row["anchor_session_id"] = anchor["session_id"]
                row["anchor_image_id"] = anchor["image_id"]
                row["future_session_id"] = future["session_id"]
                row["future_image_id"] = future["image_id"]
                row["anchor_month"] = float(months[anchor_index])
                row["future_month"] = float(months[future_index])
                row["requested_horizon_months"] = horizon_months
                row["observed_horizon_months"] = float(months[future_index] - months[anchor_index])
                row["horizon_error_months"] = float(
                    months[future_index] - months[anchor_index] - horizon_months
                )
                row["current_target_value"] = current_value
                row["future_target_value"] = future_value
                row["target_change"] = future_value - current_value
                rows_by_horizon[horizon_months].append(row)

    outputs: dict[float, pd.DataFrame] = {}
    embedding_index = index.drop_duplicates("image_id")
    for horizon_months, rows in rows_by_horizon.items():
        horizon_table = pd.DataFrame(rows)
        if horizon_table.empty:
            raise ValueError(
                f"No future-change rows found for target `{target}` at {horizon_months:g} months "
                f"with +/- {tolerance_months:g} month tolerance."
            )
        horizon_table = horizon_table.merge(
            embedding_index,
            on=["participant_id", "session_id", "image_id"],
            how="inner",
            suffixes=("", "_embedding"),
        )
        horizon_table = horizon_table[
            horizon_table["subject_split"].isin(["train", "val", "test"])
        ].copy()
        outputs[horizon_months] = horizon_table.reset_index(drop=True)
    return outputs


def build_feature_matrix(
    table: pd.DataFrame,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    return build_pooled_or_covariate_feature_matrix(
        table,
        feature,
        include_covariates,
        covariate_columns,
        load_pooled_matrix,
        standardize_pooled=False,
        covariate_only_error="No usable covariate columns were found for `--feature covariates`.",
    )


def compute_metrics_by_split(table: pd.DataFrame, y_all: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    metrics = []
    for split in ("train", "val", "test"):
        mask = table["subject_split"].to_numpy() == split
        metrics.append(asdict(compute_regression_metric(split, y_all[mask], predictions[mask])))
    return metrics


def build_model(alpha: float) -> Any:
    SimpleImputer, Ridge, _mae, _mse, _r2, Pipeline, StandardScaler = require_regression_sklearn()
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="svd")),
        ]
    )


def select_alpha(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alphas: list[float],
) -> tuple[float, list[dict[str, float]]]:
    (_SimpleImputer, _Ridge, mean_absolute_error, _mse, _r2, _Pipeline, _StandardScaler) = require_regression_sklearn()
    history = []
    best_alpha = float(alphas[0])
    best_mae = np.inf
    for alpha in alphas:
        model = build_model(float(alpha))
        model.fit(x_all[train_mask], y_all[train_mask])
        val_pred = model.predict(x_all[val_mask])
        val_mae = float(mean_absolute_error(y_all[val_mask], val_pred))
        history.append({"alpha": float(alpha), "val_mae": val_mae})
        if val_mae < best_mae:
            best_mae = val_mae
            best_alpha = float(alpha)
    return best_alpha, history


def build_prediction_table(table: pd.DataFrame, target: str, predictions: np.ndarray) -> pd.DataFrame:
    columns = [
        "participant_id",
        "session_id",
        "image_id",
        "subject_split",
        "diagnosis_at_session",
        "session_month",
        target,
    ]
    columns = [column for column in columns if column in table.columns]
    output = table[columns].copy()
    output["target_column"] = target
    output["target_value"] = table[target].to_numpy(dtype=float)
    output["prediction"] = predictions
    output["residual"] = output["prediction"] - output["target_value"]
    return output


def build_future_prediction_table(table: pd.DataFrame, target: str, delta_predictions: np.ndarray) -> pd.DataFrame:
    columns = [
        "participant_id",
        "anchor_session_id",
        "future_session_id",
        "anchor_image_id",
        "future_image_id",
        "subject_split",
        "diagnosis_at_session",
        "anchor_month",
        "future_month",
        "requested_horizon_months",
        "observed_horizon_months",
        "horizon_error_months",
        "current_target_value",
        "future_target_value",
        "target_change",
    ]
    columns = [column for column in columns if column in table.columns]
    output = table[columns].copy()
    output["target_column"] = target
    output["predicted_change"] = delta_predictions
    output["predicted_future_value"] = output["current_target_value"] + output["predicted_change"]
    output["carry_forward_future_value"] = output["current_target_value"]
    output["change_residual"] = output["predicted_change"] - output["target_change"]
    output["future_value_residual"] = output["predicted_future_value"] - output["future_target_value"]
    output["carry_forward_residual"] = output["carry_forward_future_value"] - output["future_target_value"]
    return output


def train_ridge_probe(
    table: pd.DataFrame,
    target: str,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
    alphas: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")

    x_all, cov_info = build_feature_matrix(table, feature, include_covariates, covariate_columns)
    y_all = table[target].to_numpy(dtype=np.float32)
    train_mask = table["subject_split"].to_numpy() == "train"
    val_mask = table["subject_split"].to_numpy() == "val"
    best_alpha, alpha_history = select_alpha(x_all, y_all, train_mask, val_mask, alphas)
    model = build_model(best_alpha)
    model.fit(x_all[train_mask], y_all[train_mask])
    predictions = model.predict(x_all)

    summary = {
        "task": "same_visit_cognitive_severity",
        "target": target,
        "model": "ridge_regression",
        "feature": feature,
        "include_covariates": include_covariates,
        "covariates": cov_info,
        "row_policy": None,
        "n_input_features": int(x_all.shape[1]),
        "selected_alpha": float(best_alpha),
        "alpha_selection_metric": "validation_mae",
        "alpha_history": alpha_history,
        "metrics": compute_metrics_by_split(table, y_all, predictions),
    }
    return summary, build_prediction_table(table, target, predictions)


def train_future_change_probe(
    table: pd.DataFrame,
    target: str,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
    alphas: list[float],
    horizon_months: float,
    tolerance_months: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")

    x_all, cov_info = build_feature_matrix(table, feature, include_covariates, covariate_columns)
    y_delta = table["target_change"].to_numpy(dtype=np.float32)
    train_mask = table["subject_split"].to_numpy() == "train"
    val_mask = table["subject_split"].to_numpy() == "val"
    best_alpha, alpha_history = select_alpha(x_all, y_delta, train_mask, val_mask, alphas)
    model = build_model(best_alpha)
    model.fit(x_all[train_mask], y_delta[train_mask])
    delta_predictions = model.predict(x_all)
    future_predictions = table["current_target_value"].to_numpy(dtype=np.float32) + delta_predictions
    carry_forward = table["current_target_value"].to_numpy(dtype=np.float32)
    future_values = table["future_target_value"].to_numpy(dtype=np.float32)

    summary = {
        "task": "future_cognitive_change",
        "target": target,
        "model": "ridge_regression",
        "feature": feature,
        "include_covariates": include_covariates,
        "covariates": cov_info,
        "row_policy": None,
        "horizon_months": float(horizon_months),
        "horizon_tolerance_months": float(tolerance_months),
        "n_input_features": int(x_all.shape[1]),
        "selected_alpha": float(best_alpha),
        "alpha_selection_metric": "validation_mae_on_target_change",
        "alpha_history": alpha_history,
        "change_metrics": compute_metrics_by_split(table, y_delta, delta_predictions),
        "future_value_metrics": compute_metrics_by_split(table, future_values, future_predictions),
        "carry_forward_future_value_metrics": compute_metrics_by_split(table, future_values, carry_forward),
    }
    return summary, build_future_prediction_table(table, target, delta_predictions)


def run_probe(
    target: str,
    feature: str,
    args: Any,
    progress_label: str | None = None,
) -> dict[str, Any]:
    stage1_covariate_modes = {f"{mode}_covariates": mode for mode in STAGE1_FEATURE_MODES}
    canonical_group = is_feature_group(feature)
    if canonical_group:
        selected_covariates = feature_columns("same_visit_cognition", feature)
        validate_no_target_leakage("same_visit_cognition", target, selected_covariates)
        model_feature = "pooled" if uses_neurojepa(feature) else "covariates"
        include_covariates = uses_neurojepa(feature) and bool(selected_covariates)
    else:
        selected_covariates = args.covariate_columns
        model_feature = stage1_covariate_modes.get(feature, "pooled" if feature == "pooled_covariates" else feature)
        include_covariates = (
            feature == "pooled_covariates"
            or feature in stage1_covariate_modes
            or (feature == "pooled" and bool(args.include_covariates))
            or (is_stage1_feature(feature) and bool(args.include_covariates))
        )
    table = load_cognitive_task_table(
        manifest_path=args.manifest_path,
        embedding_index_path=args.embedding_index_path,
        target=target,
        row_policy=args.row_policy,
    )
    table = add_morphometry_ratios(table)
    table = validate_feature_paths(table, model_feature)
    if progress_label is not None:
        split_counts = {split: int((table["subject_split"] == split).sum()) for split in ("train", "val", "test")}
        log_progress(f"{progress_label} table ready: rows={len(table)} splits={split_counts}")
    if target in selected_covariates:
        raise ValueError(f"Refusing to use target column `{target}` as a covariate.")
    summary, predictions = train_ridge_probe(
        table=table,
        target=target,
        feature=model_feature,
        include_covariates=include_covariates,
        covariate_columns=selected_covariates,
        alphas=args.alpha,
    )
    summary["row_policy"] = args.row_policy
    summary["n_rows"] = int(len(table))
    summary["split_counts"] = {split: int((table["subject_split"] == split).sum()) for split in ("train", "val", "test")}
    summary["feature_group"] = feature if canonical_group else None
    summary["source_covariate_columns"] = selected_covariates
    summary["paper_run"] = getattr(args, "paper_run_metadata", None)

    feature_name = feature if canonical_group or (include_covariates and feature in stage1_covariate_modes) else (
        "covariates" if model_feature == "covariates" else "pooled_covariates" if include_covariates and model_feature == "pooled" else model_feature
    )
    run_dir = args.output_dir / f"same_visit_{target}_{feature_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return {"run_dir": str(run_dir), **summary}


def run_future_change_probe(
    target: str,
    feature: str,
    horizon_months: float,
    args: Any,
    progress_label: str | None = None,
) -> dict[str, Any]:
    stage1_covariate_modes = {f"{mode}_covariates": mode for mode in STAGE1_FEATURE_MODES}
    canonical_group = is_feature_group(feature)
    if canonical_group:
        selected_covariates = feature_columns("future_cognition", feature)
        validate_no_target_leakage("future_cognition", target, selected_covariates)
        model_feature = "pooled" if uses_neurojepa(feature) else "covariates"
        include_covariates = uses_neurojepa(feature) and bool(selected_covariates)
    else:
        selected_covariates = args.covariate_columns
        model_feature = stage1_covariate_modes.get(feature, "pooled" if feature == "pooled_covariates" else feature)
        include_covariates = (
            feature == "pooled_covariates"
            or feature in stage1_covariate_modes
            or (feature == "pooled" and bool(args.include_covariates))
            or (is_stage1_feature(feature) and bool(args.include_covariates))
        )
    table = load_future_change_task_table(
        manifest_path=args.manifest_path,
        embedding_index_path=args.embedding_index_path,
        target=target,
        row_policy=args.row_policy,
        horizon_months=horizon_months,
        tolerance_months=args.horizon_tolerance_months,
    )
    table = add_morphometry_ratios(table)
    table = validate_feature_paths(table, model_feature)
    if progress_label is not None:
        split_counts = {split: int((table["subject_split"] == split).sum()) for split in ("train", "val", "test")}
        log_progress(
            f"{progress_label} table ready: rows={len(table)} splits={split_counts} "
            f"horizon={horizon_months:g}m tolerance=+/-{args.horizon_tolerance_months:g}m"
        )
    if target in selected_covariates:
        raise ValueError(f"Refusing to use target column `{target}` as a covariate.")
    summary, predictions = train_future_change_probe(
        table=table,
        target=target,
        feature=model_feature,
        include_covariates=include_covariates,
        covariate_columns=selected_covariates,
        alphas=args.alpha,
        horizon_months=horizon_months,
        tolerance_months=args.horizon_tolerance_months,
    )
    summary["row_policy"] = args.row_policy
    summary["n_rows"] = int(len(table))
    summary["split_counts"] = {split: int((table["subject_split"] == split).sum()) for split in ("train", "val", "test")}
    summary["feature_group"] = feature if canonical_group else None
    summary["source_covariate_columns"] = selected_covariates
    summary["paper_run"] = getattr(args, "paper_run_metadata", None)

    feature_name = feature if canonical_group or (include_covariates and feature in stage1_covariate_modes) else (
        "covariates" if model_feature == "covariates" else "pooled_covariates" if include_covariates and model_feature == "pooled" else model_feature
    )
    horizon_name = f"{horizon_months:g}m".replace(".", "p")
    run_dir = args.output_dir / f"future_change_{target}_{horizon_name}_{feature_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return {"run_dir": str(run_dir), **summary}


def write_comparison(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        metric_key = "metrics" if summary["task"] == "same_visit_cognitive_severity" else "future_value_metrics"
        carry_metrics = {
            metric["split"]: metric for metric in summary.get("carry_forward_future_value_metrics", [])
        }
        change_metrics = {metric["split"]: metric for metric in summary.get("change_metrics", [])}
        for metric in summary[metric_key]:
            if metric["split"] == "test":
                carry_metric = carry_metrics.get("test", {})
                change_metric = change_metrics.get("test", {})
                rows.append(
                    {
                        "task": summary["task"],
                        "target": summary["target"],
                        "horizon_months": summary.get("horizon_months"),
                        "model": summary["model"],
                        "feature": (
                            "pooled_covariates"
                            if summary["feature"] == "pooled" and summary["include_covariates"]
                            else f"{summary['feature']}_covariates"
                            if str(summary["feature"]).startswith("stage1_") and summary["include_covariates"]
                            else summary["feature"]
                        ),
                        "n_input_features": summary["n_input_features"],
                        "selected_alpha": summary["selected_alpha"],
                        "test_n": metric["n"],
                        "test_mae": metric["mae"],
                        "test_rmse": metric["rmse"],
                        "test_r2": metric["r2"],
                        "test_spearman": metric["spearman"],
                        "test_change_mae": change_metric.get("mae"),
                        "test_change_r2": change_metric.get("r2"),
                        "carry_forward_test_mae": carry_metric.get("mae"),
                        "carry_forward_test_r2": carry_metric.get("r2"),
                        "mae_improvement_vs_carry_forward": (
                            carry_metric.get("mae") - metric["mae"]
                            if carry_metric.get("mae") is not None and metric["mae"] is not None
                            else None
                        ),
                        "run_dir": summary["run_dir"],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)


def run(args: Any) -> None:
    paper_context = None
    if args.protocol_path is not None:
        if not args.replication:
            raise ValueError("`--replication` is required with `--protocol-path`.")
        paper_context = bind_paper_run(
            protocol_path=args.protocol_path,
            replication_name=args.replication,
            manifest_path=args.manifest_path,
            embedding_index_path=args.embedding_index_path,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        require_protocol_grid(
            args.alpha,
            paper_context.protocol.payload["ridge_alphas"],
            "Ridge alpha grid",
        )
        args.seed = paper_context.replication.split_seed
    args.paper_run_metadata = paper_context.metadata() if paper_context is not None else None
    validate_covariate_columns(args.covariate_columns)
    if not args.alpha:
        raise ValueError("At least one `--alpha` value is required.")
    targets = resolve_targets(args.target)
    features = args.feature or ["pooled"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    if args.task == "same_visit":
        run_specs = [(target, None, feature) for target in targets for feature in features]
        total_runs = len(run_specs)
        log_progress(f"Starting {total_runs} cognitive probe run(s): task=same_visit output_dir={args.output_dir}")
        for run_index, (target, _horizon_months, feature) in enumerate(run_specs, start=1):
            progress_label = f"[{run_index}/{total_runs}] target={target} feature={feature}"
            log_progress(f"{progress_label} START remaining={total_runs - run_index}")
            summary = run_probe(target=target, feature=feature, args=args, progress_label=progress_label)
            summary["paper_run"] = paper_context.metadata() if paper_context is not None else None
            summaries.append(summary)
            log_progress(f"{progress_label} DONE run_dir={summary['run_dir']} remaining={total_runs - run_index}")
    else:
        if not args.horizon_months:
            raise ValueError("At least one `--horizon-months` value is required for --task future_change.")
        run_specs = [
            (target, float(horizon_months), feature)
            for target in targets
            for horizon_months in args.horizon_months
            for feature in features
        ]
        total_runs = len(run_specs)
        log_progress(f"Starting {total_runs} cognitive probe run(s): task=future_change output_dir={args.output_dir}")
        for run_index, (target, horizon_months, feature) in enumerate(run_specs, start=1):
            progress_label = f"[{run_index}/{total_runs}] target={target} horizon={horizon_months:g}m feature={feature}"
            log_progress(f"{progress_label} START remaining={total_runs - run_index}")
            summary = run_future_change_probe(
                target=target,
                feature=feature,
                horizon_months=horizon_months,
                args=args,
                progress_label=progress_label,
            )
            summary["paper_run"] = paper_context.metadata() if paper_context is not None else None
            summaries.append(summary)
            log_progress(f"{progress_label} DONE run_dir={summary['run_dir']} remaining={total_runs - run_index}")

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump({"runs": summaries}, handle, indent=2)
    write_comparison(args.output_dir, summaries)
    print(f"Wrote {len(summaries)} cognitive probe run(s) to {args.output_dir}")
