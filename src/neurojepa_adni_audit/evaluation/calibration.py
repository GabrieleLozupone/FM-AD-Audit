from __future__ import annotations

import numpy as np
import pandas as pd


def gaussian_smooth_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    points: int = 101,
    probability: bool = False,
) -> pd.DataFrame:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("Calibration truth and predictions must be non-empty and aligned.")
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    if sample_weight is None:
        weights = np.ones(len(truth), dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float)[finite]
    if probability and (np.any(prediction < 0) or np.any(prediction > 1)):
        raise ValueError("Probability calibration predictions must be within [0, 1].")
    if len(truth) < 2 or np.ptp(prediction) == 0:
        return pd.DataFrame(
            {
                "predicted": [float(np.mean(prediction))],
                "observed_smooth": [float(np.average(truth, weights=weights))],
                "effective_weight": [float(weights.sum())],
                "bandwidth": [0.0],
            }
        )
    lower, upper = (0.0, 1.0) if probability else tuple(np.quantile(prediction, [0.01, 0.99]))
    grid = np.linspace(lower, upper, points)
    scale = float(np.std(prediction, ddof=1))
    bandwidth = max(scale * len(prediction) ** (-1.0 / 5.0), np.ptp(prediction) / 100.0, 1e-8)
    rows = []
    for value in grid:
        kernel = np.exp(-0.5 * ((prediction - value) / bandwidth) ** 2) * weights
        total = float(kernel.sum())
        rows.append(
            {
                "predicted": float(value),
                "observed_smooth": float(np.sum(kernel * truth) / total) if total > 0 else None,
                "effective_weight": total,
                "bandwidth": bandwidth,
            }
        )
    return pd.DataFrame(rows)
