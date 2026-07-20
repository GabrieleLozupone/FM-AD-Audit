from __future__ import annotations

import numpy as np


def censoring_survival_at(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    query_times: np.ndarray,
    *,
    left_limit: bool,
) -> np.ndarray:
    durations = np.asarray(train_durations, dtype=float)
    events = np.asarray(train_events, dtype=int)
    queries = np.asarray(query_times, dtype=float)
    if len(durations) == 0 or len(durations) != len(events):
        raise ValueError("Training durations/events must be non-empty and aligned.")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("Training survival durations must be finite and positive.")
    censor_events = 1 - events
    event_times = np.unique(durations[censor_events == 1])
    survival = np.ones(len(queries), dtype=float)
    for event_time in event_times:
        at_risk = int(np.sum(durations >= event_time))
        censored = int(np.sum((durations == event_time) & (censor_events == 1)))
        factor = 1.0 - censored / at_risk
        affected = queries > event_time if left_limit else queries >= event_time
        survival[affected] *= factor
    return survival


def uno_c_index(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    evaluation_durations: np.ndarray,
    evaluation_events: np.ndarray,
    risk_scores: np.ndarray,
    *,
    tau: float,
) -> float | None:
    durations = np.asarray(evaluation_durations, dtype=float)
    events = np.asarray(evaluation_events, dtype=int)
    risk = np.asarray(risk_scores, dtype=float)
    if not (len(durations) == len(events) == len(risk)):
        raise ValueError("Evaluation survival arrays must be aligned.")
    eligible = (events == 1) & (durations <= float(tau))
    g_before = censoring_survival_at(
        train_durations, train_events, durations, left_limit=True
    )
    numerator = 0.0
    denominator = 0.0
    for index in np.flatnonzero(eligible):
        if g_before[index] <= 0:
            continue
        comparable = durations > durations[index]
        n_comparable = int(np.sum(comparable))
        if n_comparable == 0:
            continue
        weight = 1.0 / (g_before[index] ** 2)
        concordant = np.sum(risk[index] > risk[comparable])
        tied = np.sum(risk[index] == risk[comparable])
        numerator += weight * (float(concordant) + 0.5 * float(tied))
        denominator += weight * n_comparable
    return None if denominator == 0 else float(numerator / denominator)


def ipcw_brier_score(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    evaluation_durations: np.ndarray,
    evaluation_events: np.ndarray,
    survival_probability: np.ndarray,
    *,
    horizon: float,
) -> float | None:
    durations = np.asarray(evaluation_durations, dtype=float)
    events = np.asarray(evaluation_events, dtype=int)
    survival = np.asarray(survival_probability, dtype=float)
    if not (len(durations) == len(events) == len(survival)):
        raise ValueError("Evaluation survival arrays must be aligned.")
    if np.any((survival < 0) | (survival > 1) | ~np.isfinite(survival)):
        raise ValueError("Predicted survival probabilities must be finite and within [0, 1].")
    g_before = censoring_survival_at(
        train_durations, train_events, durations, left_limit=True
    )
    g_horizon = float(
        censoring_survival_at(
            train_durations,
            train_events,
            np.asarray([horizon], dtype=float),
            left_limit=False,
        )[0]
    )
    contributions = np.zeros(len(durations), dtype=float)
    observed_event = (durations <= horizon) & (events == 1) & (g_before > 0)
    contributions[observed_event] = (survival[observed_event] ** 2) / g_before[observed_event]
    event_free = durations > horizon
    if g_horizon > 0:
        contributions[event_free] = ((1.0 - survival[event_free]) ** 2) / g_horizon
    elif event_free.any():
        return None
    return None if len(contributions) == 0 else float(np.mean(contributions))


def cumulative_dynamic_auc(
    train_durations: np.ndarray,
    train_events: np.ndarray,
    evaluation_durations: np.ndarray,
    evaluation_events: np.ndarray,
    risk_probability: np.ndarray,
    *,
    horizon: float,
) -> float | None:
    durations = np.asarray(evaluation_durations, dtype=float)
    events = np.asarray(evaluation_events, dtype=int)
    risk = np.asarray(risk_probability, dtype=float)
    if not (len(durations) == len(events) == len(risk)):
        raise ValueError("Evaluation survival arrays must be aligned.")
    cases = (durations <= horizon) & (events == 1)
    controls = durations > horizon
    if not cases.any() or not controls.any():
        return None
    g_before = censoring_survival_at(
        train_durations, train_events, durations, left_limit=True
    )
    valid_cases = cases & (g_before > 0)
    if not valid_cases.any():
        return None
    numerator = 0.0
    denominator = 0.0
    control_risk = risk[controls]
    for index in np.flatnonzero(valid_cases):
        weight = 1.0 / g_before[index]
        concordant = np.sum(risk[index] > control_risk)
        tied = np.sum(risk[index] == control_risk)
        numerator += weight * (float(concordant) + 0.5 * float(tied))
        denominator += weight * len(control_risk)
    return None if denominator == 0 else float(numerator / denominator)
