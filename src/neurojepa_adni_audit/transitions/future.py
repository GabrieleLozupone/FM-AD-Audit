from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.paths import PROJECT_ROOT
from neurojepa_adni_audit.features.covariates import build_covariate_transform
from neurojepa_adni_audit.features.progression_aligned import (
    STAGE1_FEATURE_MODES,
    filter_stage1_available_rows,
    is_stage1_feature,
    load_stage1_feature_matrix,
    stage1_feature_info,
)
from neurojepa_adni_audit.metrics import choose_threshold, compute_grouped_regression_metric, compute_metrics
from neurojepa_adni_audit.metrics import require_regression_metric_deps
from neurojepa_adni_audit.sklearn_utils import require_ridge_logistic_sklearn
from neurojepa_adni_audit.transitions.state import (
    DEFAULT_ALPHAS,
    DEFAULT_STATE_COVARIATE_COLUMNS,
    DEFAULT_STATE_TRANSITION_PATH,
    load_jsonable,
    load_matrix,
    make_horizon_bins,
    read_csv,
)
from neurojepa_adni_audit.features.groups import (
    FEATURE_GROUPS,
    add_morphometry_ratios,
    is_feature_group,
    transition_feature_columns,
    uses_neurojepa,
    validate_no_target_leakage,
)
from neurojepa_adni_audit.paper_run import bind_paper_table_run, require_protocol_grid


DEFAULT_HORIZON_MATCHED_TABLE = PROJECT_ROOT / "data" / "neurostar_adni_t1w_state_transition_pairs_horizon_matched.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "neurojepa_future_state_probe"
FEATURE_MODES = [
    *FEATURE_GROUPS,
    "covariates_morphometry",
    "jepa_only",
    "jepa_covariates_morphometry",
    *STAGE1_FEATURE_MODES,
    *(f"{feature}_covariates_morphometry" for feature in STAGE1_FEATURE_MODES),
]
DEFAULT_REGRESSION_TARGETS = [
    "hippocampus_total_icv",
    "lateral_ventricles_total_icv",
    "mmse",
    "adas11",
    "adas13",
]
DEFAULT_CLASSIFICATION_TARGETS = ["future_ad", "mci_to_ad"]
REGRESSION_TARGETS = {
    "hippocampus_total": ("current_vol_hippocampus_total", "next_vol_hippocampus_total", "volume"),
    "lateral_ventricles_total": (
        "current_vol_lateral_ventricles_total",
        "next_vol_lateral_ventricles_total",
        "volume",
    ),
    "hippocampus_total_icv": ("current_vol_hippocampus_total", "next_vol_hippocampus_total", "icv_ratio"),
    "lateral_ventricles_total_icv": (
        "current_vol_lateral_ventricles_total",
        "next_vol_lateral_ventricles_total",
        "icv_ratio",
    ),
    "mmse": ("current_mmse_at_session", "next_mmse_at_session", "cognitive"),
    "adas11": ("current_adas11_at_session", "next_adas11_at_session", "cognitive"),
    "adas13": ("current_adas13_at_session", "next_adas13_at_session", "cognitive"),
    "cdrsb": ("current_cdrsb_at_session", "next_cdrsb_at_session", "cognitive"),
}
CLASSIFICATION_TARGETS = {
    "future_ad": "future diagnosis is AD versus non-AD",
    "mci_to_ad": "current MCI visit transitions to future AD versus future non-AD",
}


def require_sklearn() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    SimpleImputer, LogisticRegression, Ridge, Pipeline, StandardScaler = require_ridge_logistic_sklearn()
    mean_absolute_error, mean_squared_error, r2_score = require_regression_metric_deps()
    return SimpleImputer, LogisticRegression, Ridge, mean_absolute_error, mean_squared_error, r2_score, Pipeline, StandardScaler


def regression_metrics_by_split_and_group(
    table: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    split_values = table["subject_split"].astype(str).to_numpy()
    horizon = make_horizon_bins(table["delta_months"]).astype(str).to_numpy()
    diagnosis = table["current_diagnosis_at_session"].astype(str).fillna("missing").to_numpy()
    for split in ("train", "val", "test"):
        split_mask = split_values == split
        rows.append(asdict(compute_grouped_regression_metric(split, "all", y_true[split_mask], y_pred[split_mask])))
        for group in sorted(set(diagnosis[split_mask])):
            group_mask = split_mask & (diagnosis == group)
            rows.append(asdict(compute_grouped_regression_metric(split, f"diagnosis={group}", y_true[group_mask], y_pred[group_mask])))
        for label in ["0-9m", "9-18m", "18-30m", "30-42m", "42m+"]:
            group_mask = split_mask & (horizon == label)
            if group_mask.any():
                rows.append(asdict(compute_grouped_regression_metric(split, f"horizon={label}", y_true[group_mask], y_pred[group_mask])))
    return rows


def build_features(
    table: pd.DataFrame,
    feature: str,
    covariate_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    train_mask = table["subject_split"].to_numpy() == "train"
    cov_info: dict[str, Any] = {"included": False, "numeric_columns": [], "categorical_columns": []}
    cov_all = None
    if is_feature_group(feature):
        selected_columns = transition_feature_columns("future_morphometry", feature)
        validate_no_target_leakage("future_morphometry", "future_target_value", selected_columns)
        include_covariates = bool(selected_columns)
        if include_covariates:
            _transform, numeric, categorical, cov_all = build_covariate_transform(
                train_rows=table[train_mask],
                all_rows=table,
                covariate_columns=selected_columns,
            )
            if cov_all is None:
                raise ValueError(f"No usable columns found for canonical feature group `{feature}`.")
            cov_info = {
                "included": True,
                "numeric_columns": numeric,
                "categorical_columns": categorical,
                "source_columns": selected_columns,
            }
        if uses_neurojepa(feature):
            embeddings = load_matrix(table["current_pooled_embedding_path"])
            if cov_all is not None:
                embeddings = np.concatenate([embeddings, cov_all], axis=1)
            return embeddings.astype(np.float32), cov_info
        return cov_all.astype(np.float32), cov_info
    stage1_covariate_modes = {f"{mode}_covariates_morphometry": mode for mode in STAGE1_FEATURE_MODES}
    stage1_feature = stage1_covariate_modes.get(feature, feature)
    if feature in {"covariates_morphometry", "jepa_covariates_morphometry"} or feature in stage1_covariate_modes:
        _transform, numeric, categorical, cov_all = build_covariate_transform(
            train_rows=table[train_mask],
            all_rows=table,
            covariate_columns=covariate_columns,
        )
        if cov_all is None:
            raise ValueError("No usable covariates/morphometry were found.")
        cov_info = {"included": True, "numeric_columns": numeric, "categorical_columns": categorical}
    if feature in {"jepa_only", "jepa_covariates_morphometry"}:
        current_embeddings = load_matrix(table["current_pooled_embedding_path"])
    if feature == "jepa_only":
        return current_embeddings.astype(np.float32), cov_info
    if feature == "covariates_morphometry":
        return cov_all.astype(np.float32), cov_info
    if feature == "jepa_covariates_morphometry":
        return np.concatenate([current_embeddings, cov_all], axis=1).astype(np.float32), cov_info
    if is_stage1_feature(stage1_feature):
        current_stage1 = load_stage1_feature_matrix(table, stage1_feature, image_id_column="current_image_id")
        cov_info["stage1"] = stage1_feature_info(stage1_feature)
        if feature in stage1_covariate_modes:
            return np.concatenate([current_stage1, cov_all], axis=1).astype(np.float32), cov_info
        return current_stage1.astype(np.float32), cov_info
    raise ValueError(f"Unsupported feature mode: {feature}")


def make_regression_values(table: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in REGRESSION_TARGETS:
        raise ValueError(f"Unsupported regression target `{target}`.")
    current_col, next_col, target_family = REGRESSION_TARGETS[target]
    required = {current_col, next_col}
    if target_family == "icv_ratio":
        required |= {"current_vol_total_intracranial", "next_vol_total_intracranial"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Regression target `{target}` needs missing columns: {missing}")
    task = table.copy()
    current = pd.to_numeric(task[current_col], errors="coerce")
    future = pd.to_numeric(task[next_col], errors="coerce")
    if target_family == "icv_ratio":
        current_icv = pd.to_numeric(task["current_vol_total_intracranial"], errors="coerce")
        next_icv = pd.to_numeric(task["next_vol_total_intracranial"], errors="coerce")
        current = current / current_icv
        future = future / next_icv
    task["current_target_value"] = current
    task["future_target_value"] = future
    task["target_change"] = future - current
    task = task[task["current_target_value"].notna() & task["future_target_value"].notna()].copy()
    return task.reset_index(drop=True)


def build_ridge(alpha: float) -> Any:
    SimpleImputer, _LogisticRegression, Ridge, _mae, _mse, _r2, Pipeline, StandardScaler = require_sklearn()
    return Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(alpha), solver="svd"))])


def select_alpha(x_all: np.ndarray, y_delta: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, alphas: list[float]) -> tuple[float, list[dict[str, float]]]:
    (_SimpleImputer, _LogisticRegression, _Ridge, mean_absolute_error, _mse, _r2, _Pipeline, _StandardScaler) = require_sklearn()
    history = []
    best_alpha = float(alphas[0])
    best_mae = np.inf
    for alpha in alphas:
        model = build_ridge(alpha)
        model.fit(x_all[train_mask], y_delta[train_mask])
        pred = model.predict(x_all[val_mask])
        mae = float(mean_absolute_error(y_delta[val_mask], pred))
        history.append({"alpha": float(alpha), "val_change_mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    return best_alpha, history


def group_mean_delta_predictions(table: pd.DataFrame, current: np.ndarray, future: np.ndarray) -> np.ndarray:
    train_mask = table["subject_split"].to_numpy() == "train"
    delta = future - current
    global_delta = float(np.mean(delta[train_mask]))
    predicted_delta = np.full(len(table), global_delta, dtype=np.float32)
    groups = table["current_diagnosis_at_session"].astype(str).fillna("missing").to_numpy()
    for group in sorted(set(groups[train_mask])):
        group_train_mask = train_mask & (groups == group)
        if group_train_mask.any():
            predicted_delta[groups == group] = float(np.mean(delta[group_train_mask]))
    return current + predicted_delta


def train_regression_target(
    full_table: pd.DataFrame,
    target: str,
    feature: str,
    covariate_columns: list[str],
    alphas: list[float],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    table = make_regression_values(full_table, target)
    stage1_covariate_modes = {f"{mode}_covariates_morphometry": mode for mode in STAGE1_FEATURE_MODES}
    if feature in {"jepa_only", "jepa_covariates_morphometry"} or (
        is_feature_group(feature) and uses_neurojepa(feature)
    ):
        table = table[
            table["current_pooled_embedding_path"].notna()
            & table["current_pooled_embedding_path"].map(lambda value: Path(str(value)).exists())
        ].copy()
    if is_stage1_feature(feature) or feature in stage1_covariate_modes:
        table = filter_stage1_available_rows(table[table["current_image_id"].notna()].copy(), image_id_column="current_image_id")
    table = table[table["subject_split"].isin(["train", "val", "test"])].reset_index(drop=True)
    split_values = table["subject_split"].to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    if not train_mask.any() or not val_mask.any() or not (split_values == "test").any():
        raise ValueError(f"Target `{target}` does not have train/val/test coverage.")
    x_all, cov_info = build_features(table, feature, covariate_columns)
    current = table["current_target_value"].to_numpy(dtype=np.float32)
    future = table["future_target_value"].to_numpy(dtype=np.float32)
    y_delta = future - current
    best_alpha, alpha_history = select_alpha(x_all, y_delta, train_mask, val_mask, alphas)
    model = build_ridge(best_alpha)
    model.fit(x_all[train_mask], y_delta[train_mask])
    predicted_delta = model.predict(x_all).astype(np.float32)
    predicted_future = current + predicted_delta
    carry_forward = current.copy()
    diagnosis_mean = group_mean_delta_predictions(table, current, future)
    current_col, next_col, target_family = REGRESSION_TARGETS[target]
    summary = {
        "task": "future_state_regression",
        "target": target,
        "target_family": target_family,
        "current_source_column": current_col,
        "future_source_column": next_col,
        "model": "ridge_residual_change",
        "feature": feature,
        "feature_group": feature if is_feature_group(feature) else None,
        "covariates": cov_info,
        "n_rows": int(len(table)),
        "n_input_features": int(x_all.shape[1]),
        "selected_alpha": float(best_alpha),
        "alpha_selection_metric": "validation_change_mae",
        "alpha_history": alpha_history,
        "future_value_metrics": regression_metrics_by_split_and_group(table, future, predicted_future),
        "change_metrics": regression_metrics_by_split_and_group(table, y_delta, predicted_delta),
        "carry_forward_future_value_metrics": regression_metrics_by_split_and_group(table, future, carry_forward),
        "diagnosis_mean_delta_future_value_metrics": regression_metrics_by_split_and_group(table, future, diagnosis_mean),
    }
    columns = [
        "pair_id",
        "participant_id",
        "current_session_id",
        "next_session_id",
        "subject_split",
        "delta_months",
        "requested_horizon_months",
        "current_diagnosis_at_session",
        "next_diagnosis_at_session",
    ]
    predictions = table[[column for column in columns if column in table.columns]].copy()
    predictions["target"] = target
    predictions["current_target_value"] = current
    predictions["future_target_value"] = future
    predictions["target_change"] = y_delta
    predictions["predicted_change"] = predicted_delta
    predictions["predicted_future_value"] = predicted_future
    predictions["carry_forward_future_value"] = carry_forward
    predictions["diagnosis_mean_delta_future_value"] = diagnosis_mean
    return load_jsonable(summary), predictions, table


def make_classification_table(full_table: pd.DataFrame, target: str) -> pd.DataFrame:
    table = full_table.copy()
    table = table[table["next_diagnosis_at_session"].notna()].copy()
    if target == "future_ad":
        table["label"] = table["next_diagnosis_at_session"].astype(str).eq("AD").astype(int)
    elif target == "mci_to_ad":
        table = table[table["current_diagnosis_at_session"].astype(str).eq("MCI")].copy()
        table["label"] = table["next_diagnosis_at_session"].astype(str).eq("AD").astype(int)
    else:
        raise ValueError(f"Unsupported classification target `{target}`.")
    return table.reset_index(drop=True)


def build_logistic(c_value: float, seed: int) -> Any:
    SimpleImputer, LogisticRegression, _Ridge, _mae, _mse, _r2, Pipeline, StandardScaler = require_sklearn()
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def select_c_value(x_all: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, alphas: list[float], seed: int) -> tuple[float, list[dict[str, float | None]]]:
    from sklearn.metrics import average_precision_score

    history = []
    best_c = 1.0 / float(alphas[0])
    best_auprc = -np.inf
    for alpha in alphas:
        c_value = 1.0 / float(alpha)
        model = build_logistic(c_value, seed)
        model.fit(x_all[train_mask], labels[train_mask])
        prob = model.predict_proba(x_all[val_mask])[:, 1]
        auprc = float(average_precision_score(labels[val_mask], prob)) if len(np.unique(labels[val_mask])) == 2 else None
        history.append({"alpha_equivalent": float(alpha), "c": float(c_value), "val_auprc": auprc})
        score = -np.inf if auprc is None else auprc
        if score > best_auprc:
            best_auprc = score
            best_c = float(c_value)
    return best_c, history


def prior_probabilities(table: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    train_mask = table["subject_split"].to_numpy() == "train"
    global_prior = float(np.mean(labels[train_mask]))
    out = np.full(len(table), global_prior, dtype=np.float32)
    groups = (
        table["current_diagnosis_at_session"].astype(str).fillna("missing")
        + "|"
        + make_horizon_bins(table["delta_months"]).astype(str).fillna("missing")
    ).to_numpy()
    for group in sorted(set(groups[train_mask])):
        group_train_mask = train_mask & (groups == group)
        if group_train_mask.sum() >= 10:
            out[groups == group] = float(np.mean(labels[group_train_mask]))
    return out


def classification_metrics_by_split(table: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    rows = []
    split_values = table["subject_split"].astype(str).to_numpy()
    horizon = make_horizon_bins(table["delta_months"]).astype(str).to_numpy()
    diagnosis = table["current_diagnosis_at_session"].astype(str).fillna("missing").to_numpy()
    for split in ("train", "val", "test"):
        split_mask = split_values == split
        rows.append({"group": "all", **asdict(compute_metrics(split, labels[split_mask], probabilities[split_mask], threshold))})
        for group in sorted(set(diagnosis[split_mask])):
            group_mask = split_mask & (diagnosis == group)
            rows.append({"group": f"diagnosis={group}", **asdict(compute_metrics(split, labels[group_mask], probabilities[group_mask], threshold))})
        for label in ["0-9m", "9-18m", "18-30m", "30-42m", "42m+"]:
            group_mask = split_mask & (horizon == label)
            if group_mask.any():
                rows.append({"group": f"horizon={label}", **asdict(compute_metrics(split, labels[group_mask], probabilities[group_mask], threshold))})
    return rows


def train_classification_target(
    full_table: pd.DataFrame,
    target: str,
    feature: str,
    covariate_columns: list[str],
    alphas: list[float],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    table = make_classification_table(full_table, target)
    stage1_covariate_modes = {f"{mode}_covariates_morphometry": mode for mode in STAGE1_FEATURE_MODES}
    if feature in {"jepa_only", "jepa_covariates_morphometry"} or (
        is_feature_group(feature) and uses_neurojepa(feature)
    ):
        table = table[
            table["current_pooled_embedding_path"].notna()
            & table["current_pooled_embedding_path"].map(lambda value: Path(str(value)).exists())
        ].copy()
    if is_stage1_feature(feature) or feature in stage1_covariate_modes:
        table = filter_stage1_available_rows(table[table["current_image_id"].notna()].copy(), image_id_column="current_image_id")
    table = table[table["subject_split"].isin(["train", "val", "test"])].reset_index(drop=True)
    labels = table["label"].to_numpy(dtype=np.int64)
    split_values = table["subject_split"].to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    if not train_mask.any() or not val_mask.any() or not (split_values == "test").any():
        raise ValueError(f"Target `{target}` does not have train/val/test coverage.")
    if len(np.unique(labels[train_mask])) < 2:
        raise ValueError(f"Training split for `{target}` has only one class.")
    x_all, cov_info = build_features(table, feature, covariate_columns)
    best_c, c_history = select_c_value(x_all, labels, train_mask, val_mask, alphas, seed)
    model = build_logistic(best_c, seed)
    model.fit(x_all[train_mask], labels[train_mask])
    probabilities = model.predict_proba(x_all)[:, 1].astype(np.float32)
    val_threshold = choose_threshold(labels[val_mask], probabilities[val_mask])
    carry_forward_prob = table["current_diagnosis_at_session"].astype(str).eq("AD").to_numpy(dtype=np.float32)
    if target == "mci_to_ad":
        carry_forward_prob = np.zeros(len(table), dtype=np.float32)
    prior_prob = prior_probabilities(table, labels)
    summary = {
        "task": "future_state_classification",
        "target": target,
        "target_description": CLASSIFICATION_TARGETS[target],
        "model": "balanced_logistic_regression",
        "feature": feature,
        "feature_group": feature if is_feature_group(feature) else None,
        "covariates": cov_info,
        "n_rows": int(len(table)),
        "n_positive": int(labels.sum()),
        "n_input_features": int(x_all.shape[1]),
        "selected_c": float(best_c),
        "regularization_selection_metric": "validation_auprc",
        "regularization_history": c_history,
        "threshold": float(val_threshold),
        "metrics": classification_metrics_by_split(table, labels, probabilities, val_threshold),
        "carry_forward_metrics": classification_metrics_by_split(table, labels, carry_forward_prob, 0.5),
        "diagnosis_horizon_prior_metrics": classification_metrics_by_split(table, labels, prior_prob, 0.5),
    }
    columns = [
        "pair_id",
        "participant_id",
        "current_session_id",
        "next_session_id",
        "subject_split",
        "delta_months",
        "requested_horizon_months",
        "current_diagnosis_at_session",
        "next_diagnosis_at_session",
    ]
    predictions = table[[column for column in columns if column in table.columns]].copy()
    predictions["target"] = target
    predictions["label"] = labels
    predictions["probability_positive"] = probabilities
    predictions["predicted_label"] = (probabilities >= val_threshold).astype(int)
    predictions["carry_forward_probability_positive"] = carry_forward_prob
    predictions["diagnosis_horizon_prior_probability_positive"] = prior_prob
    return load_jsonable(summary), predictions, table


def write_run(run_dir: Path, summary: dict[str, Any], predictions: pd.DataFrame, table: pd.DataFrame) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(load_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    table.to_csv(run_dir / "task_table.csv", index=False)


def summarize_comparison(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        if summary["task"] == "future_state_regression":
            model = next(row for row in summary["future_value_metrics"] if row["split"] == "test" and row["group"] == "all")
            carry = next(row for row in summary["carry_forward_future_value_metrics"] if row["split"] == "test" and row["group"] == "all")
            prior = next(row for row in summary["diagnosis_mean_delta_future_value_metrics"] if row["split"] == "test" and row["group"] == "all")
            rows.append(
                {
                    "task": summary["task"],
                    "target": summary["target"],
                    "feature": summary["feature"],
                    "n_rows": summary["n_rows"],
                    "test_mae": model["mae"],
                    "test_rmse": model["rmse"],
                    "test_r2": model["r2"],
                    "carry_forward_test_mae": carry["mae"],
                    "diagnosis_mean_delta_test_mae": prior["mae"],
                    "mae_improvement_vs_carry_forward": None if carry["mae"] is None else carry["mae"] - model["mae"],
                }
            )
        else:
            model = next(row for row in summary["metrics"] if row["split"] == "test" and row["group"] == "all")
            carry = next(row for row in summary["carry_forward_metrics"] if row["split"] == "test" and row["group"] == "all")
            prior = next(row for row in summary["diagnosis_horizon_prior_metrics"] if row["split"] == "test" and row["group"] == "all")
            rows.append(
                {
                    "task": summary["task"],
                    "target": summary["target"],
                    "feature": summary["feature"],
                    "n_rows": summary["n_rows"],
                    "n_positive": summary["n_positive"],
                    "test_auroc": model["auroc"],
                    "test_auprc": model["auprc"],
                    "test_balanced_accuracy": model["balanced_accuracy"],
                    "carry_forward_test_auroc": carry["auroc"],
                    "diagnosis_horizon_prior_test_auroc": prior["auroc"],
                    "auprc_improvement_vs_prior": None if prior["auprc"] is None else model["auprc"] - prior["auprc"],
                }
            )
    return pd.DataFrame(rows)


def run(args: Any) -> None:
    paper_context = None
    if args.protocol_path is not None:
        if not args.replication:
            raise ValueError("`--replication` is required with `--protocol-path`.")
        paper_context = bind_paper_table_run(
            protocol_path=args.protocol_path,
            replication_name=args.replication,
            table_path=args.table_path,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        require_protocol_grid(args.alpha, paper_context.protocol.payload["ridge_alphas"], "Ridge alpha grid")
        args.seed = paper_context.replication.split_seed
    features = args.feature or FEATURE_MODES
    if args.regression_target is None and args.classification_target is None:
        regression_targets = DEFAULT_REGRESSION_TARGETS
        classification_targets = DEFAULT_CLASSIFICATION_TARGETS
    else:
        regression_targets = args.regression_target or []
        classification_targets = args.classification_target or []
    table = read_csv(args.table_path)
    table = add_morphometry_ratios(table, prefix="current_")
    table = table[table["subject_split"].isin(["train", "val", "test"])].reset_index(drop=True)
    if getattr(args, "limit_rows", None) is not None and int(args.limit_rows) > 0:
        per_split = max(int(args.limit_rows) // 3, 1)
        table = (
            table.sort_values(["participant_id", "current_session_id", "next_session_id"])
            .groupby("subject_split", group_keys=False)
            .head(per_split)
            .reset_index(drop=True)
        )
    summaries = []
    for feature in features:
        for target in regression_targets:
            summary, predictions, task_table = train_regression_target(
                full_table=table,
                target=target,
                feature=feature,
                covariate_columns=args.covariate_columns,
                alphas=args.alpha,
            )
            summary["paper_run"] = paper_context.metadata() if paper_context is not None else None
            run_dir = args.output_dir / f"regression_{target}_{feature}"
            write_run(run_dir, summary, predictions, task_table)
            summaries.append({"run_dir": str(run_dir), **summary})
            print(f"Wrote Task F regression run: {run_dir}")
        for target in classification_targets:
            summary, predictions, task_table = train_classification_target(
                full_table=table,
                target=target,
                feature=feature,
                covariate_columns=args.covariate_columns,
                alphas=args.alpha,
                seed=args.seed,
            )
            summary["paper_run"] = paper_context.metadata() if paper_context is not None else None
            run_dir = args.output_dir / f"classification_{target}_{feature}"
            write_run(run_dir, summary, predictions, task_table)
            summaries.append({"run_dir": str(run_dir), **summary})
            print(f"Wrote Task F classification run: {run_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all_summaries.json").write_text(
        json.dumps(load_jsonable(summaries), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    comparison = summarize_comparison(summaries)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    print(f"Wrote comparison: {args.output_dir / 'comparison.csv'}")
