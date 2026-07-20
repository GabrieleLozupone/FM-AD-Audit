from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MetricResult:
    split: str
    n: int
    n_positive: int
    auroc: float | None
    auprc: float | None
    accuracy: float | None
    balanced_accuracy: float | None
    matthews_correlation_coefficient: float | None
    sensitivity: float | None
    specificity: float | None
    brier_score: float | None
    log_loss: float | None
    calibration_intercept: float | None
    calibration_slope: float | None
    expected_calibration_error: float | None
    threshold: float


@dataclass
class RegressionMetric:
    split: str
    n: int
    target_mean: float | None
    target_std: float | None
    prediction_mean: float | None
    mae: float | None
    rmse: float | None
    r2: float | None
    pearson: float | None
    spearman: float | None


@dataclass
class GroupedRegressionMetric:
    split: str
    group: str
    n: int
    target_mean: float | None
    target_std: float | None
    prediction_mean: float | None
    mae: float | None
    rmse: float | None
    r2: float | None
    pearson: float | None
    spearman: float | None


def require_binary_metric_deps() -> tuple[Any, Any, Any]:
    try:
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return average_precision_score, balanced_accuracy_score, roc_auc_score


def require_regression_metric_deps() -> tuple[Any, Any, Any]:
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return mean_absolute_error, mean_squared_error, r2_score


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float | None:
    if len(y_true) < 2 or np.nanstd(y_true) == 0.0 or np.nanstd(y_pred) == 0.0:
        return None
    value = pd.Series(y_true).corr(pd.Series(y_pred), method=method)
    if pd.isna(value):
        return None
    return float(value)


def compute_regression_metric(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetric:
    mean_absolute_error, mean_squared_error, r2_score = require_regression_metric_deps()
    if len(y_true) == 0:
        return RegressionMetric(split, 0, None, None, None, None, None, None, None, None)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.nanstd(y_true) > 0.0 else None
    return RegressionMetric(
        split=split,
        n=int(len(y_true)),
        target_mean=float(np.mean(y_true)),
        target_std=float(np.std(y_true)),
        prediction_mean=float(np.mean(y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=r2,
        pearson=safe_corr(y_true, y_pred, "pearson"),
        spearman=safe_corr(y_true, y_pred, "spearman"),
    )


def compute_grouped_regression_metric(split: str, group: str, y_true: np.ndarray, y_pred: np.ndarray) -> GroupedRegressionMetric:
    metric = compute_regression_metric(split, y_true, y_pred)
    return GroupedRegressionMetric(
        split=metric.split,
        group=group,
        n=metric.n,
        target_mean=metric.target_mean,
        target_std=metric.target_std,
        prediction_mean=metric.prediction_mean,
        mae=metric.mae,
        rmse=metric.rmse,
        r2=metric.r2,
        pearson=metric.pearson,
        spearman=metric.spearman,
    )


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    (_average_precision_score, balanced_accuracy_score, _roc_auc_score) = require_binary_metric_deps()
    if len(np.unique(labels)) < 2:
        return 0.5
    candidates = np.unique(np.clip(probabilities, 0.0, 1.0))
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 200))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        score = balanced_accuracy_score(labels, probabilities >= threshold)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def compute_calibration(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | None]:
    if len(labels) == 0:
        return {
            "brier_score": None,
            "log_loss": None,
            "calibration_intercept": None,
            "calibration_slope": None,
            "expected_calibration_error": None,
        }
    labels_float = np.asarray(labels, dtype=float)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((clipped - labels_float) ** 2))
    logloss = float(-np.mean(labels_float * np.log(clipped) + (1.0 - labels_float) * np.log(1.0 - clipped)))

    bin_ids = np.minimum((clipped * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(float(clipped[mask].mean()) - float(labels_float[mask].mean()))

    intercept = None
    slope = None
    logits = np.log(clipped / (1.0 - clipped))
    if len(np.unique(labels)) == 2 and float(np.std(logits)) > 0.0:
        try:
            from scipy.optimize import minimize

            def objective(parameters: np.ndarray) -> float:
                linear = np.clip(parameters[0] + parameters[1] * logits, -30.0, 30.0)
                return float(np.sum(np.logaddexp(0.0, linear) - labels_float * linear))

            fitted = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
            if fitted.success and np.all(np.isfinite(fitted.x)):
                intercept, slope = (float(value) for value in fitted.x)
        except (ImportError, ValueError):
            pass
    return {
        "brier_score": brier,
        "log_loss": logloss,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": float(ece),
    }


def compute_metrics(split: str, labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> MetricResult:
    average_precision_score, balanced_accuracy_score, roc_auc_score = require_binary_metric_deps()
    auroc = None
    auprc = None
    accuracy = None
    balanced_accuracy = None
    matthews_correlation_coefficient = None
    sensitivity = None
    specificity = None
    calibration = compute_calibration(labels, probabilities)
    if len(labels) > 0 and len(np.unique(labels)) == 2:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
        predictions = (probabilities >= threshold).astype(int)
        balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
        accuracy = float(np.mean(predictions == labels))
        true_positive = int(np.sum((labels == 1) & (predictions == 1)))
        true_negative = int(np.sum((labels == 0) & (predictions == 0)))
        false_positive = int(np.sum((labels == 0) & (predictions == 1)))
        false_negative = int(np.sum((labels == 1) & (predictions == 0)))
        positive_count = true_positive + false_negative
        negative_count = true_negative + false_positive
        sensitivity = float(true_positive / positive_count) if positive_count > 0 else None
        specificity = float(true_negative / negative_count) if negative_count > 0 else None
        denominator = np.sqrt(
            float(
                (true_positive + false_positive)
                * (true_positive + false_negative)
                * (true_negative + false_positive)
                * (true_negative + false_negative)
            )
        )
        matthews_correlation_coefficient = (
            float(((true_positive * true_negative) - (false_positive * false_negative)) / denominator)
            if denominator > 0.0
            else 0.0
        )
    return MetricResult(
        split=split,
        n=int(len(labels)),
        n_positive=int(np.sum(labels)),
        auroc=auroc,
        auprc=auprc,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        matthews_correlation_coefficient=matthews_correlation_coefficient,
        sensitivity=sensitivity,
        specificity=specificity,
        brier_score=calibration["brier_score"],
        log_loss=calibration["log_loss"],
        calibration_intercept=calibration["calibration_intercept"],
        calibration_slope=calibration["calibration_slope"],
        expected_calibration_error=calibration["expected_calibration_error"],
        threshold=float(threshold),
    )


def compute_metrics_by_split(
    table: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    metrics = []
    for split in ("train", "val", "test"):
        mask = table["subject_split"].to_numpy() == split
        metrics.append(asdict(compute_metrics(split, labels[mask], probabilities[mask], threshold)))
    return metrics
