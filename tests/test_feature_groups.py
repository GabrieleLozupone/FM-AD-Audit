import pandas as pd
import pytest

from neurojepa_adni_audit.features.groups import (
    FEATURE_GROUPS,
    add_morphometry_ratios,
    feature_columns,
    is_feature_group,
    transition_feature_columns,
    uses_neurojepa,
    validate_no_target_leakage,
)


def test_morphometry_ratios_are_explicit():
    table = pd.DataFrame(
        {
            "vol_total_intracranial": [1000.0],
            "vol_hippocampus_total": [10.0],
            "vol_lateral_ventricles_total": [20.0],
            "vol_left_amygdala": [2.0],
            "vol_right_amygdala": [3.0],
            "vol_left_inferior_lateral_ventricle": [1.0],
            "vol_right_inferior_lateral_ventricle": [1.5],
            "vol_left_cerebral_cortex": [200.0],
            "vol_right_cerebral_cortex": [210.0],
        }
    )
    output = add_morphometry_ratios(table)
    assert output.loc[0, "hippocampus_icv_ratio"] == pytest.approx(0.01)
    assert output.loc[0, "lateral_ventricles_icv_ratio"] == pytest.approx(0.02)
    assert output.loc[0, "amygdala_icv_ratio"] == pytest.approx(0.005)
    assert output.loc[0, "inferior_lateral_ventricles_icv_ratio"] == pytest.approx(0.0025)
    assert output.loc[0, "cerebral_cortex_icv_ratio"] == pytest.approx(0.41)


def test_same_visit_target_is_rejected():
    with pytest.raises(ValueError, match="Target leakage"):
        validate_no_target_leakage("same_visit_cognition", "adas13_at_session", ["adas13_at_session"])


def test_future_current_target_is_allowed():
    validate_no_target_leakage("future_cognition", "adas13_at_session", ["current_target_value"])


def test_conventional_group_contains_morphometry_and_demographics():
    columns = feature_columns("ad_vs_cn", "conventional_full")
    assert "hippocampus_icv_ratio" in columns
    assert "amygdala_icv_ratio" in columns
    assert "field_strength_at_session" not in columns
    assert "age_at_session" in columns


def test_all_paper_groups_resolve_for_every_task():
    tasks = [
        "ad_vs_cn",
        "same_visit_cognition",
        "future_cognition",
        "future_morphometry",
        "mci_conversion_36m",
        "first_mci_survival",
    ]
    for task in tasks:
        for group in FEATURE_GROUPS:
            assert is_feature_group(group)
            assert isinstance(feature_columns(task, group), list)


def test_neurojepa_group_detection_is_explicit():
    assert uses_neurojepa("neurojepa_only")
    assert uses_neurojepa("neurojepa_conventional")
    assert not uses_neurojepa("conventional_full")


def test_future_outcome_columns_are_rejected():
    with pytest.raises(ValueError, match="Target leakage"):
        validate_no_target_leakage(
            "future_cognition", "adas13_at_session", ["future_adas13_at_session"]
        )


def test_transition_groups_use_current_visit_column_names():
    columns = transition_feature_columns("future_morphometry", "conventional_full")
    assert "current_hippocampus_icv_ratio" in columns
    assert "current_age_at_session" in columns
    assert "current_diagnosis_at_session" in columns
    assert not any(column.startswith("next_") for column in columns)
