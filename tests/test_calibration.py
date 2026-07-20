import numpy as np

from neurojepa_adni_audit.evaluation.calibration import gaussian_smooth_calibration


def test_gaussian_smooth_probability_calibration_is_finite_and_bounded_on_grid():
    prediction = np.linspace(0.01, 0.99, 200)
    truth = (prediction > 0.5).astype(float)
    curve = gaussian_smooth_calibration(truth, prediction, probability=True)
    assert len(curve) == 101
    assert curve["predicted"].between(0, 1).all()
    assert curve["observed_smooth"].between(0, 1).all()


def test_gaussian_smooth_regression_accepts_participant_weights():
    prediction = np.linspace(-2, 2, 50)
    truth = prediction + 0.2
    weights = np.linspace(0.5, 1.5, 50)
    curve = gaussian_smooth_calibration(truth, prediction, sample_weight=weights)
    assert curve["observed_smooth"].notna().all()
    assert (curve["effective_weight"] > 0).all()
