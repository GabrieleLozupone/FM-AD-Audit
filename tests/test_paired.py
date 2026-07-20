import pandas as pd
import pytest

from neurojepa_adni_audit.metrics import compute_calibration
from neurojepa_adni_audit.evaluation.paired import (
    benjamini_hochberg,
    paired_subject_bootstrap,
    paired_subject_bootstrap_weighted,
    subject_equal_mae,
)


def test_negative_mae_difference_favors_combined():
    frame = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b"],
            "target": [0.0, 1.0, 0.0, 1.0],
            "combined": [0.0, 1.0, 0.0, 1.0],
            "conventional": [1.0, 2.0, 1.0, 2.0],
        }
    )
    result = paired_subject_bootstrap(
        frame, subject_equal_mae, "combined", "conventional", replicates=1000, seed=7
    )
    assert result.estimate < 0
    assert result.ci95_high < 0


def test_bh_is_monotone_in_rank():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]


def test_calibration_metrics_for_perfect_probabilities():
    calibration = compute_calibration(
        labels=pd.Series([0, 0, 1, 1]).to_numpy(),
        probabilities=pd.Series([0.0, 0.0, 1.0, 1.0]).to_numpy(),
    )
    assert calibration["brier_score"] == pytest.approx(0.0, abs=1e-12)
    assert calibration["log_loss"] == pytest.approx(0.0, abs=1e-6)
    assert calibration["expected_calibration_error"] == pytest.approx(0.0, abs=1e-6)


def test_weighted_binary_bootstrap_preserves_subject_pairing():
    frame = pd.DataFrame(
        {
            "participant_id": [f"p{index:02d}" for index in range(30)],
            "target": [index % 2 for index in range(30)],
            "combined": [0.1 if index % 2 == 0 else 0.9 for index in range(30)],
            "conventional": [0.5] * 30,
        }
    )

    def weighted_auroc(rows, prediction, weights):
        from sklearn.metrics import roc_auc_score

        return float(
            roc_auc_score(
                rows["target"].to_numpy(), rows[prediction].to_numpy(), sample_weight=weights
            )
        )

    result = paired_subject_bootstrap_weighted(
        frame, weighted_auroc, "combined", "conventional", replicates=1000, seed=11
    )
    assert result.estimate > 0
    assert result.ci95_low > 0
    assert result.valid_replicates == 1000
