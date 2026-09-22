from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from neurojepa_adni_audit.nested_cv import NestedFoldSpec
from neurojepa_adni_audit.nested_training import (
    _choose_balanced_accuracy_threshold,
    _classification_score,
    _subject_equal_mae,
    attach_nested_split,
    build_fold_matrices,
    participant_equal_weights,
)


TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True)
class MLPTrainingSpec:
    hidden_units: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    gradient_clip_norm: float


@dataclass(frozen=True)
class MLPFitResult:
    prediction: np.ndarray
    final_training_loss: float
    target_mean: float | None
    target_scale: float | None
    elapsed_seconds: float
    device: str


@dataclass(frozen=True)
class MLPInnerFoldResult:
    weight_decay: float
    inner_fold: int
    score: float
    n_train_rows: int
    n_validation_rows: int
    n_train_participants: int
    n_validation_participants: int
    final_training_loss: float
    elapsed_seconds: float
    device: str


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("The nonlinear probe requires PyTorch.") from exc
    normalized = requested.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no usable CUDA device is available.")
        return "cuda"
    if normalized == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _seed_everything(seed: int, device: str, torch_threads: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif torch_threads > 0:
        torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _weighted_location_scale(target: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(target, weights=weights))
    variance = float(np.average((target - mean) ** 2, weights=weights))
    scale = math.sqrt(max(variance, 0.0))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return mean, scale


def _classification_weights(target: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(target, dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("MLP classification fitting requires both classes.")
    counts = np.bincount(labels, minlength=2).astype(float)
    class_weights = len(labels) / (2.0 * counts)
    combined = np.asarray(base_weights, dtype=float) * class_weights[labels]
    return combined * (len(combined) / combined.sum())


def _fit_predict_mlp(
    *,
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_evaluation: np.ndarray,
    train_weights: np.ndarray,
    kind: TaskKind,
    weight_decay: float,
    spec: MLPTrainingSpec,
    seed: int,
    device: str,
    torch_threads: int,
) -> MLPFitResult:
    import torch
    from torch import nn

    _seed_everything(seed, device, torch_threads)
    started = time.monotonic()
    x_train_array = np.asarray(x_train, dtype=np.float32)
    x_evaluation_array = np.asarray(x_evaluation, dtype=np.float32)
    y = np.asarray(target_train, dtype=np.float32)
    weights = np.asarray(train_weights, dtype=np.float32)
    if not np.isfinite(x_train_array).all() or not np.isfinite(x_evaluation_array).all():
        raise ValueError("MLP inputs must be finite after fold-local preprocessing.")
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("MLP targets and fitting weights must be finite and weights positive.")

    target_mean: float | None = None
    target_scale: float | None = None
    if kind == "regression":
        target_mean, target_scale = _weighted_location_scale(y, weights)
        fitting_target = (y - target_mean) / target_scale
        fitting_weights = weights * (len(weights) / weights.sum())
    else:
        fitting_target = y
        fitting_weights = _classification_weights(y, weights)

    model = nn.Sequential(
        nn.Linear(x_train_array.shape[1], spec.hidden_units),
        nn.ReLU(),
        nn.Dropout(spec.dropout),
        nn.Linear(spec.hidden_units, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=float(weight_decay),
    )
    x_tensor = torch.as_tensor(x_train_array, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(fitting_target, dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(fitting_weights, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_loss = float("nan")
    model.train()
    for _epoch in range(spec.epochs):
        order = torch.randperm(len(x_tensor), generator=generator)
        loss_total = 0.0
        weight_total = 0.0
        for start in range(0, len(order), spec.batch_size):
            indices = order[start : start + spec.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_tensor[indices]).squeeze(1)
            if kind == "classification":
                elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, y_tensor[indices], reduction="none"
                )
            else:
                elementwise = torch.square(output - y_tensor[indices])
            batch_weights = weight_tensor[indices]
            weighted_sum = torch.sum(elementwise * batch_weights)
            # Global weights are normalized to mean one. Dividing by the batch
            # size gives an unbiased minibatch gradient for the locked weighted
            # empirical risk; normalizing separately inside each batch would
            # distort participant-equal weights when batch composition changes.
            loss = weighted_sum / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MLP training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip_norm)
            optimizer.step()
            batch_weight = float(torch.sum(batch_weights).detach().cpu())
            loss_total += float(weighted_sum.detach().cpu())
            weight_total += batch_weight
        final_loss = loss_total / weight_total

    model.eval()
    with torch.no_grad():
        evaluation_tensor = torch.as_tensor(
            x_evaluation_array, dtype=torch.float32, device=device
        )
        raw = model(evaluation_tensor).squeeze(1)
        if kind == "classification":
            prediction = torch.sigmoid(raw).cpu().numpy()
        else:
            prediction = raw.cpu().numpy() * float(target_scale) + float(target_mean)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite MLP predictions.")
    return MLPFitResult(
        prediction=prediction,
        final_training_loss=float(final_loss),
        target_mean=target_mean,
        target_scale=target_scale,
        elapsed_seconds=float(time.monotonic() - started),
        device=device,
    )


def _fit_seed(base_seed: int, outer_fold: int, inner_fold: int | None) -> int:
    inner_component = 99 if inner_fold is None else int(inner_fold)
    return int((base_seed + 1009 * outer_fold + 37 * inner_component) % (2**31 - 1))


def run_nested_mlp_fold(
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
    weight_decays: list[float],
    training_spec: MLPTrainingSpec,
    seed: int,
    repeated_visits: bool,
    device: str,
    torch_threads: int = 1,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not weight_decays or any(float(value) <= 0 for value in weight_decays):
        raise ValueError("weight_decays must be a non-empty positive grid.")
    resolved_device = resolve_device(device)
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
    histories: list[MLPInnerFoldResult] = []
    oof_by_parameter: dict[float, list[pd.DataFrame]] = {}
    for weight_decay in map(float, weight_decays):
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
            base_weights = (
                participant_equal_weights(train_rows)
                if repeated_visits
                else np.ones(len(train_rows), dtype=float)
            )
            fit = _fit_predict_mlp(
                x_train=x_train,
                target_train=train_rows[target].to_numpy(),
                x_evaluation=x_validation,
                train_weights=base_weights,
                kind=kind,
                weight_decay=weight_decay,
                spec=training_spec,
                seed=_fit_seed(seed, outer_fold, inner_fold),
                device=resolved_device,
                torch_threads=torch_threads,
            )
            if kind == "classification":
                score = _classification_score(
                    task,
                    validation_rows[target].to_numpy(dtype=int),
                    fit.prediction,
                )
            else:
                score = -_subject_equal_mae(
                    validation_rows,
                    validation_rows[target].to_numpy(dtype=float),
                    fit.prediction,
                )
            histories.append(
                MLPInnerFoldResult(
                    weight_decay=weight_decay,
                    inner_fold=inner_fold,
                    score=float(score),
                    n_train_rows=len(train_rows),
                    n_validation_rows=len(validation_rows),
                    n_train_participants=train_rows["participant_id"].nunique(),
                    n_validation_participants=validation_rows["participant_id"].nunique(),
                    final_training_loss=fit.final_training_loss,
                    elapsed_seconds=fit.elapsed_seconds,
                    device=fit.device,
                )
            )
            oof_parts.append(
                pd.DataFrame(
                    {
                        "_nested_row_id": validation_rows["_nested_row_id"].to_numpy(),
                        "truth": validation_rows[target].to_numpy(),
                        "prediction": fit.prediction,
                        "participant_id": validation_rows["participant_id"].to_numpy(),
                        "inner_fold": inner_fold,
                    }
                )
            )
        oof_by_parameter[weight_decay] = oof_parts

    history = pd.DataFrame([asdict(item) for item in histories])
    means = history.groupby("weight_decay", as_index=False)["score"].mean()
    selected = means.sort_values(
        ["score", "weight_decay"], ascending=[False, False]
    ).iloc[0]
    best_weight_decay = float(selected["weight_decay"])
    selected_oof = pd.concat(oof_by_parameter[best_weight_decay], ignore_index=True)

    x_development, x_test, feature_metadata = build_fold_matrices(
        train_rows=development,
        evaluation_rows=test,
        full_pooled_matrix=full_pooled_matrix,
        task=task,
        feature_group=feature_group,
        target=target,
    )
    final_weights = (
        participant_equal_weights(development)
        if repeated_visits
        else np.ones(len(development), dtype=float)
    )
    final_fit = _fit_predict_mlp(
        x_train=x_development,
        target_train=development[target].to_numpy(),
        x_evaluation=x_test,
        train_weights=final_weights,
        kind=kind,
        weight_decay=best_weight_decay,
        spec=training_spec,
        seed=_fit_seed(seed, outer_fold, None),
        device=resolved_device,
        torch_threads=torch_threads,
    )
    if kind == "classification":
        threshold = _choose_balanced_accuracy_threshold(
            selected_oof["truth"].to_numpy(dtype=int),
            selected_oof["prediction"].to_numpy(dtype=float),
        )
        outer_score = _classification_score(
            task, test[target].to_numpy(dtype=int), final_fit.prediction
        )
    else:
        threshold = None
        outer_score = -_subject_equal_mae(
            test, test[target].to_numpy(dtype=float), final_fit.prediction
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
    predictions["prediction"] = final_fit.prediction
    if task in {"future_cognition", "future_morphometry"} and "current_target_value" in predictions:
        predictions["predicted_change"] = final_fit.prediction
        predictions["predicted_future_value"] = (
            predictions["current_target_value"].to_numpy(dtype=float)
            + final_fit.prediction
        )
        predictions["carry_forward_future_value"] = predictions["current_target_value"]
    if threshold is not None:
        predictions["threshold_inner_oof"] = threshold
        predictions["predicted_label"] = (final_fit.prediction >= threshold).astype(int)
        predictions["predicted_label_threshold_0_5"] = (
            final_fit.prediction >= 0.5
        ).astype(int)

    summary = {
        "decoder": "mlp",
        "repeat": repeat,
        "outer_fold": outer_fold,
        "task": task,
        "target": target,
        "kind": kind,
        "feature_group": feature_group,
        "selected_weight_decay": best_weight_decay,
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
        "outer_test_metric_value": (
            float(-outer_score) if kind == "regression" else float(outer_score)
        ),
        "threshold_inner_oof": threshold,
        "development_rows": len(development),
        "development_participants": development["participant_id"].nunique(),
        "test_rows": len(test),
        "test_participants": test["participant_id"].nunique(),
        "participant_equal_fit_weights": repeated_visits,
        "class_balanced_fit_weights": kind == "classification",
        "final_refit_on_complete_outer_development": True,
        "feature_metadata": feature_metadata,
        "training_spec": asdict(training_spec),
        "resolved_device": resolved_device,
        "torch_threads": torch_threads,
        "final_fit": {
            "final_training_loss": final_fit.final_training_loss,
            "target_mean": final_fit.target_mean,
            "target_scale": final_fit.target_scale,
            "elapsed_seconds": final_fit.elapsed_seconds,
        },
        "inner_history": history.to_dict(orient="records"),
    }
    return summary, predictions, selected_oof
