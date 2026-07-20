from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from neurojepa_adni_audit.features.covariates import build_covariate_transform
from neurojepa_adni_audit.features.groups import (
    feature_columns,
    transition_feature_columns,
    uses_neurojepa,
    validate_no_target_leakage,
)
from neurojepa_adni_audit.nested_cv import NestedFoldSpec, materialize_split


TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class InnerFoldResult:
    hyperparameter: float
    inner_fold: int
    score: float
    n_train_rows: int
    n_validation_rows: int
    n_train_participants: int
    n_validation_participants: int


def participant_equal_weights(rows: pd.DataFrame) -> np.ndarray:
    if "participant_id" not in rows:
        raise ValueError("participant_id is required for participant-equal weights.")
    counts = rows.groupby("participant_id")["participant_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


def attach_nested_split(
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    spec: NestedFoldSpec,
) -> pd.DataFrame:
    if table["participant_id"].isna().any():
        raise ValueError("Task table contains null participant IDs.")
    split = materialize_split(assignments, spec)[["participant_id", "subject_split"]].copy()
    split["participant_id"] = split["participant_id"].astype(str)
    work = table.drop(columns=["subject_split"], errors="ignore").copy()
    work["participant_id"] = work["participant_id"].astype(str)
    work["_nested_row_id"] = np.arange(len(work), dtype=int)
    merged = work.merge(split, on="participant_id", how="inner", validate="many_to_one")
    expected = set(work["participant_id"])
    assigned = set(merged["participant_id"])
    missing = expected - assigned
    if missing and spec.inner_fold is None:
        raise ValueError(f"{len(missing)} task participants have no outer-fold assignment.")
    if merged.empty:
        raise ValueError("Nested split contains no eligible task rows.")
    return merged.sort_values("_nested_row_id").reset_index(drop=True)


def _standardize_embeddings(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `scikit-learn`.") from exc
    scaler = StandardScaler()
    return (
        scaler.fit_transform(train).astype(np.float32),
        scaler.transform(evaluation).astype(np.float32),
    )


def build_fold_matrices(
    *,
    train_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    full_pooled_matrix: np.ndarray | None,
    task: str,
    feature_group: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    covariates = (
        transition_feature_columns(task, feature_group)
        if task == "future_morphometry"
        else feature_columns(task, feature_group)
    )
    validate_no_target_leakage(task, target, covariates)
    blocks_train: list[np.ndarray] = []
    blocks_evaluation: list[np.ndarray] = []
    metadata: dict[str, Any] = {"covariate_columns": covariates, "uses_neurojepa": uses_neurojepa(feature_group)}

    if covariates:
        combined = pd.concat([train_rows, evaluation_rows], ignore_index=True, sort=False)
        _transform, numeric, categorical, transformed = build_covariate_transform(
            train_rows=train_rows,
            all_rows=combined,
            covariate_columns=covariates,
        )
        if transformed is None:
            raise ValueError(f"No usable conventional predictors for feature group `{feature_group}`.")
        blocks_train.append(transformed[: len(train_rows)])
        blocks_evaluation.append(transformed[len(train_rows) :])
        metadata.update({"numeric_columns": numeric, "categorical_columns": categorical})

    if uses_neurojepa(feature_group):
        if full_pooled_matrix is None:
            raise ValueError(f"Feature group `{feature_group}` requires pooled embeddings.")
        train_embedding = full_pooled_matrix[train_rows["_nested_row_id"].to_numpy(dtype=int)]
        evaluation_embedding = full_pooled_matrix[
            evaluation_rows["_nested_row_id"].to_numpy(dtype=int)
        ]
        train_embedding, evaluation_embedding = _standardize_embeddings(
            train_embedding, evaluation_embedding
        )
        blocks_train.insert(0, train_embedding)
        blocks_evaluation.insert(0, evaluation_embedding)

    if not blocks_train:
        raise ValueError(f"Feature group `{feature_group}` produced no predictors.")
    return (
        np.concatenate(blocks_train, axis=1),
        np.concatenate(blocks_evaluation, axis=1),
        metadata,
    )


def _subject_equal_mae(rows: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    errors = pd.DataFrame(
        {"participant_id": rows["participant_id"].to_numpy(), "error": np.abs(truth - prediction)}
    )
    return float(errors.groupby("participant_id")["error"].mean().mean())


def _classification_score(task: str, truth: np.ndarray, probability: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `scikit-learn`.") from exc
    if len(np.unique(truth)) != 2:
        raise ValueError("Every classification validation fold must contain both classes.")
    if task == "ad_vs_cn":
        return float(roc_auc_score(truth, probability))
    if task == "mci_conversion_36m":
        return float(average_precision_score(truth, probability))
    raise ValueError(f"Unsupported nested classification task: {task}")


def _choose_balanced_accuracy_threshold(truth: np.ndarray, probability: np.ndarray) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `scikit-learn`.") from exc
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    scores = [balanced_accuracy_score(truth, probability >= threshold) for threshold in candidates]
    best = max(scores)
    return float(min(threshold for threshold, score in zip(candidates, scores) if score == best))


def _build_estimator(kind: TaskKind, hyperparameter: float, seed: int) -> Any:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `scikit-learn`.") from exc
    if kind == "regression":
        return Ridge(alpha=float(hyperparameter), solver="svd")
    return LogisticRegression(
        C=float(hyperparameter),
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=int(seed),
    )


def run_nested_tabular_fold(
    *,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    full_pooled_matrix: np.ndarray | None,
    repeat: str,
    outer_fold: int,
    task: str,
    target: str,
    feature_group: str,
    kind: TaskKind,
    hyperparameters: list[float],
    seed: int,
    repeated_visits: bool,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not hyperparameters or any(float(value) <= 0 for value in hyperparameters):
        raise ValueError("Hyperparameter grid must be non-empty and positive.")
    outer = attach_nested_split(table, assignments, NestedFoldSpec(repeat, outer_fold))
    development = outer[outer["subject_split"] == "train"].copy()
    test = outer[outer["subject_split"] == "test"].copy()
    if development.empty or test.empty:
        raise ValueError("Outer development and test partitions must both be non-empty.")
    if kind == "classification" and (
        development[target].nunique() != 2 or test[target].nunique() != 2
    ):
        raise ValueError("Outer development and test partitions must both contain both classes.")

    n_inner = int(
        assignments.loc[
            (assignments["repeat"] == repeat)
            & (assignments["selected_outer_fold"] == outer_fold),
            "inner_folds",
        ].iloc[0]
    )
    histories: list[InnerFoldResult] = []
    oof_by_parameter: dict[float, list[pd.DataFrame]] = {}
    for hyperparameter in map(float, hyperparameters):
        oof_parts = []
        for inner_fold in range(n_inner):
            inner = attach_nested_split(
                table, assignments, NestedFoldSpec(repeat, outer_fold, inner_fold)
            )
            train_rows = inner[inner["subject_split"] == "train"].copy()
            validation_rows = inner[inner["subject_split"] == "val"].copy()
            x_train, x_validation, _metadata = build_fold_matrices(
                train_rows=train_rows,
                evaluation_rows=validation_rows,
                full_pooled_matrix=full_pooled_matrix,
                task=task,
                feature_group=feature_group,
                target=target,
            )
            estimator = _build_estimator(kind, hyperparameter, seed + inner_fold)
            weights = participant_equal_weights(train_rows) if repeated_visits else None
            estimator.fit(x_train, train_rows[target].to_numpy(), sample_weight=weights)
            if kind == "classification":
                prediction = estimator.predict_proba(x_validation)[:, 1]
                score = _classification_score(
                    task, validation_rows[target].to_numpy(dtype=int), prediction
                )
            else:
                prediction = estimator.predict(x_validation)
                score = -_subject_equal_mae(
                    validation_rows,
                    validation_rows[target].to_numpy(dtype=float),
                    prediction,
                )
            histories.append(
                InnerFoldResult(
                    hyperparameter=hyperparameter,
                    inner_fold=inner_fold,
                    score=float(score),
                    n_train_rows=len(train_rows),
                    n_validation_rows=len(validation_rows),
                    n_train_participants=train_rows["participant_id"].nunique(),
                    n_validation_participants=validation_rows["participant_id"].nunique(),
                )
            )
            oof_parts.append(
                pd.DataFrame(
                    {
                        "_nested_row_id": validation_rows["_nested_row_id"].to_numpy(),
                        "truth": validation_rows[target].to_numpy(),
                        "prediction": prediction,
                        "participant_id": validation_rows["participant_id"].to_numpy(),
                        "inner_fold": inner_fold,
                    }
                )
            )
        oof_by_parameter[hyperparameter] = oof_parts

    history = pd.DataFrame([asdict(item) for item in histories])
    means = history.groupby("hyperparameter", as_index=False)["score"].mean()
    if kind == "regression":
        selected = means.sort_values(["score", "hyperparameter"], ascending=[False, False]).iloc[0]
    else:
        selected = means.sort_values(["score", "hyperparameter"], ascending=[False, True]).iloc[0]
    best_hyperparameter = float(selected["hyperparameter"])
    selected_oof = pd.concat(oof_by_parameter[best_hyperparameter], ignore_index=True)

    x_development, x_test, feature_metadata = build_fold_matrices(
        train_rows=development,
        evaluation_rows=test,
        full_pooled_matrix=full_pooled_matrix,
        task=task,
        feature_group=feature_group,
        target=target,
    )
    final_estimator = _build_estimator(kind, best_hyperparameter, seed)
    final_weights = participant_equal_weights(development) if repeated_visits else None
    final_estimator.fit(
        x_development, development[target].to_numpy(), sample_weight=final_weights
    )
    if kind == "classification":
        outer_prediction = final_estimator.predict_proba(x_test)[:, 1]
        threshold = _choose_balanced_accuracy_threshold(
            selected_oof["truth"].to_numpy(dtype=int),
            selected_oof["prediction"].to_numpy(dtype=float),
        )
        outer_score = _classification_score(
            task, test[target].to_numpy(dtype=int), outer_prediction
        )
    else:
        outer_prediction = final_estimator.predict(x_test)
        threshold = None
        outer_score = -_subject_equal_mae(
            test, test[target].to_numpy(dtype=float), outer_prediction
        )

    prediction_columns = [
        column
        for column in (
            "participant_id",
            "session_id",
            "image_id",
            "anchor_session_id",
            "future_session_id",
            "current_target_value",
            "future_target_value",
            "pair_id",
            "current_session_id",
            "current_image_id",
            "next_session_id",
            "next_image_id",
        )
        if column in test
    ]
    predictions = test[prediction_columns].copy()
    predictions["repeat"] = repeat
    predictions["outer_fold"] = outer_fold
    predictions["target"] = target
    predictions["truth"] = test[target].to_numpy()
    predictions["prediction"] = outer_prediction
    if task in {"future_cognition", "future_morphometry"} and "current_target_value" in predictions:
        predictions["predicted_change"] = outer_prediction
        predictions["predicted_future_value"] = (
            predictions["current_target_value"].to_numpy(dtype=float) + outer_prediction
        )
        predictions["carry_forward_future_value"] = predictions["current_target_value"]
    if threshold is not None:
        predictions["threshold_inner_oof"] = threshold
        predictions["predicted_label"] = (outer_prediction >= threshold).astype(int)
        predictions["predicted_label_threshold_0_5"] = (outer_prediction >= 0.5).astype(int)

    summary = {
        "repeat": repeat,
        "outer_fold": outer_fold,
        "task": task,
        "target": target,
        "kind": kind,
        "feature_group": feature_group,
        "selected_hyperparameter": best_hyperparameter,
        "selection_score": float(selected["score"]),
        "selection_metric_value": (
            float(-selected["score"]) if kind == "regression" else float(selected["score"])
        ),
        "selection_metric": (
            "mean_inner_fold_auroc"
            if task == "ad_vs_cn"
            else "mean_inner_fold_auprc"
            if kind == "classification"
            else "mean_inner_fold_subject_equal_mae"
        ),
        "outer_test_score": float(outer_score),
        "outer_test_score_sign": "negative_mae" if kind == "regression" else "as_reported",
        "outer_test_metric_value": (
            float(-outer_score) if kind == "regression" else float(outer_score)
        ),
        "threshold_inner_oof": threshold,
        "development_rows": len(development),
        "development_participants": development["participant_id"].nunique(),
        "test_rows": len(test),
        "test_participants": test["participant_id"].nunique(),
        "participant_equal_fit_weights": repeated_visits,
        "final_refit_on_complete_outer_development": True,
        "feature_metadata": feature_metadata,
        "inner_history": history.to_dict(orient="records"),
    }
    return summary, predictions, selected_oof
