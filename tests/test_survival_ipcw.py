import numpy as np

from neurojepa_adni_audit.survival.ipcw import cumulative_dynamic_auc, ipcw_brier_score, uno_c_index


def test_uno_c_is_one_for_perfect_risk_order_and_zero_when_reversed_without_censoring():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.ones(4, dtype=int)
    perfect_risk = -durations
    assert uno_c_index(durations, events, durations, events, perfect_risk, tau=4.0) == 1.0
    assert uno_c_index(durations, events, durations, events, -perfect_risk, tau=4.0) == 0.0


def test_ipcw_brier_reduces_to_standard_brier_without_censoring():
    durations = np.array([1.0, 2.0, 4.0, 5.0])
    events = np.ones(4, dtype=int)
    predicted_survival = np.array([0.1, 0.2, 0.8, 0.9])
    score = ipcw_brier_score(
        durations,
        events,
        durations,
        events,
        predicted_survival,
        horizon=3.0,
    )
    truth_survival = np.array([0.0, 0.0, 1.0, 1.0])
    assert np.isclose(score, np.mean((truth_survival - predicted_survival) ** 2))


def test_ipcw_metrics_handle_censoring_and_return_finite_values():
    train_durations = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    train_events = np.array([1, 0, 1, 0, 1, 0])
    risk = np.array([0.9, 0.7, 0.8, 0.4, 0.6, 0.2])
    survival = 1.0 / (1.0 + np.exp(risk))
    c_index = uno_c_index(
        train_durations,
        train_events,
        train_durations,
        train_events,
        risk,
        tau=5.0,
    )
    brier = ipcw_brier_score(
        train_durations,
        train_events,
        train_durations,
        train_events,
        survival,
        horizon=3.0,
    )
    assert c_index is not None and 0 <= c_index <= 1
    assert brier is not None and np.isfinite(brier)


def test_cumulative_dynamic_auc_is_one_for_perfect_horizon_ranking():
    durations = np.array([1.0, 2.0, 4.0, 5.0])
    events = np.ones(4, dtype=int)
    risk = np.array([0.9, 0.8, 0.2, 0.1])
    assert cumulative_dynamic_auc(
        durations, events, durations, events, risk, horizon=3.0
    ) == 1.0
