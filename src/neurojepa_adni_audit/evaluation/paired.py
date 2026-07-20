from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedBootstrapResult:
    estimate: float
    ci95_low: float
    ci95_high: float
    n_subjects: int
    replicates: int
    seed: int


@dataclass(frozen=True)
class WeightedPairedBootstrapResult:
    estimate: float
    ci95_low: float
    ci95_high: float
    bootstrap_sign_p_value: float
    n_subjects: int
    n_rows: int
    replicates: int
    valid_replicates: int
    seed: int


def subject_equal_mae(frame: pd.DataFrame, prediction: str) -> float:
    losses = (frame["target"].to_numpy(dtype=float) - frame[prediction].to_numpy(dtype=float)) ** 2
    work = pd.DataFrame({"participant_id": frame["participant_id"].astype(str), "loss": np.sqrt(losses)})
    return float(work.groupby("participant_id", sort=False)["loss"].mean().mean())


def paired_subject_bootstrap(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame, str], float],
    combined_prediction: str,
    conventional_prediction: str,
    replicates: int = 10_000,
    seed: int = 20260623,
) -> PairedBootstrapResult:
    if replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates for paper reporting.")
    subjects = frame["participant_id"].astype(str).drop_duplicates().to_numpy()
    if not len(subjects):
        raise ValueError("No participants available for paired bootstrap.")
    indexed = {subject: frame[frame["participant_id"].astype(str) == subject] for subject in subjects}

    def difference(sampled: np.ndarray) -> float:
        draws = []
        for draw_index, subject in enumerate(sampled):
            draw = indexed[subject].copy()
            draw["participant_id"] = f"bootstrap_{draw_index}"
            draws.append(draw)
        sampled_frame = pd.concat(draws, ignore_index=True)
        return metric(sampled_frame, combined_prediction) - metric(sampled_frame, conventional_prediction)

    estimate = difference(subjects)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        values[index] = difference(rng.choice(subjects, size=len(subjects), replace=True))
    low, high = np.quantile(values, [0.025, 0.975])
    return PairedBootstrapResult(estimate, float(low), float(high), len(subjects), replicates, seed)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output.tolist()


def result_dict(result: PairedBootstrapResult) -> dict[str, int | float]:
    return asdict(result)


def paired_subject_bootstrap_weighted(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame, str, np.ndarray], float],
    combined_prediction: str,
    conventional_prediction: str,
    replicates: int = 10_000,
    seed: int = 20260623,
) -> WeightedPairedBootstrapResult:
    if replicates < 1_000:
        raise ValueError("Use at least 1000 bootstrap replicates for paper reporting.")
    if frame.empty:
        raise ValueError("No paired rows available for bootstrap.")
    participant_codes, subjects = pd.factorize(frame["participant_id"].astype(str), sort=False)
    n_subjects = len(subjects)
    if n_subjects == 0:
        raise ValueError("No participants available for paired bootstrap.")
    rows_per_subject = np.bincount(participant_codes, minlength=n_subjects).astype(float)

    def row_weights(multiplicity: np.ndarray) -> np.ndarray:
        return multiplicity[participant_codes] / rows_per_subject[participant_codes]

    unit_multiplicity = np.ones(n_subjects, dtype=float)
    point_weights = row_weights(unit_multiplicity)
    estimate = metric(frame, combined_prediction, point_weights) - metric(
        frame, conventional_prediction, point_weights
    )
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    probabilities = np.full(n_subjects, 1.0 / n_subjects)
    for index in range(replicates):
        multiplicity = rng.multinomial(n_subjects, probabilities).astype(float)
        weights = row_weights(multiplicity)
        values[index] = metric(frame, combined_prediction, weights) - metric(
            frame, conventional_prediction, weights
        )
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < max(1000, int(0.95 * replicates)):
        raise ValueError(
            f"Too few valid bootstrap replicates: {len(finite_values)}/{replicates}. "
            "Check class/event counts."
        )
    low, high = np.quantile(finite_values, [0.025, 0.975])
    non_positive = (np.count_nonzero(finite_values <= 0.0) + 1.0) / (len(finite_values) + 1.0)
    non_negative = (np.count_nonzero(finite_values >= 0.0) + 1.0) / (len(finite_values) + 1.0)
    sign_p = min(1.0, 2.0 * min(non_positive, non_negative))
    return WeightedPairedBootstrapResult(
        estimate=float(estimate),
        ci95_low=float(low),
        ci95_high=float(high),
        bootstrap_sign_p_value=float(sign_p),
        n_subjects=n_subjects,
        n_rows=int(len(frame)),
        replicates=replicates,
        valid_replicates=int(len(finite_values)),
        seed=seed,
    )
