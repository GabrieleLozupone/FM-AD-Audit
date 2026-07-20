from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MORPHOMETRY_COLUMNS = (
    "hippocampus_icv_ratio",
    "amygdala_icv_ratio",
    "lateral_ventricles_icv_ratio",
    "inferior_lateral_ventricles_icv_ratio",
    "cerebral_cortex_icv_ratio",
)
DEMOGRAPHIC_COLUMNS = (
    "age_at_session",
    "sex_baseline",
    "education_years_baseline",
    "apoe4_status",
)
ACQUISITION_COLUMNS: tuple[str, ...] = ()
FEATURE_GROUPS = (
    "morphometry_only",
    "clinical_demographic",
    "conventional_full",
    "neurojepa_only",
    "neurojepa_morphometry",
    "neurojepa_conventional",
)


@dataclass(frozen=True)
class TaskFeaturePolicy:
    task: str
    clinical_columns: tuple[str, ...] = ()
    allow_current_diagnosis: bool = False
    allow_current_target: bool = False


TASK_POLICIES = {
    "ad_vs_cn": TaskFeaturePolicy("ad_vs_cn"),
    "same_visit_cognition": TaskFeaturePolicy(
        "same_visit_cognition", clinical_columns=("diagnosis_at_session",), allow_current_diagnosis=True
    ),
    "future_cognition": TaskFeaturePolicy(
        "future_cognition",
        clinical_columns=("diagnosis_at_session", "current_target_value"),
        allow_current_diagnosis=True,
        allow_current_target=True,
    ),
    "future_morphometry": TaskFeaturePolicy(
        "future_morphometry",
        clinical_columns=(
            "current_diagnosis_at_session",
            "current_mmse_at_session",
            "current_adas11_at_session",
            "current_adas13_at_session",
        ),
        allow_current_diagnosis=True,
    ),
    "mci_conversion_36m": TaskFeaturePolicy(
        "mci_conversion_36m", clinical_columns=("mmse_at_session", "adas11_at_session", "adas13_at_session")
    ),
    "first_mci_survival": TaskFeaturePolicy(
        "first_mci_survival", clinical_columns=("mmse_at_session", "adas11_at_session", "adas13_at_session")
    ),
}


def add_morphometry_ratios(table: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = table.copy()
    icv = pd.to_numeric(out[f"{prefix}vol_total_intracranial"], errors="coerce")
    totals = {
        "hippocampus": pd.to_numeric(out[f"{prefix}vol_hippocampus_total"], errors="coerce"),
        "amygdala": (
            pd.to_numeric(out[f"{prefix}vol_left_amygdala"], errors="coerce")
            + pd.to_numeric(out[f"{prefix}vol_right_amygdala"], errors="coerce")
        ),
        "lateral_ventricles": pd.to_numeric(
            out[f"{prefix}vol_lateral_ventricles_total"], errors="coerce"
        ),
        "inferior_lateral_ventricles": (
            pd.to_numeric(out[f"{prefix}vol_left_inferior_lateral_ventricle"], errors="coerce")
            + pd.to_numeric(out[f"{prefix}vol_right_inferior_lateral_ventricle"], errors="coerce")
        ),
        "cerebral_cortex": (
            pd.to_numeric(out[f"{prefix}vol_left_cerebral_cortex"], errors="coerce")
            + pd.to_numeric(out[f"{prefix}vol_right_cerebral_cortex"], errors="coerce")
        ),
    }
    valid = icv > 0
    for name, total in totals.items():
        out[f"{prefix}{name}_icv_ratio"] = np.where(valid, total / icv, np.nan)
    return out


def feature_columns(task: str, group: str, include_acquisition: bool = True) -> list[str]:
    if task not in TASK_POLICIES:
        raise ValueError(f"Unknown task policy: {task}")
    policy = TASK_POLICIES[task]
    acquisition = list(ACQUISITION_COLUMNS) if include_acquisition else []
    morphometry = list(MORPHOMETRY_COLUMNS) + acquisition
    clinical = list(DEMOGRAPHIC_COLUMNS) + list(policy.clinical_columns)
    groups = {
        "morphometry_only": morphometry,
        "clinical_demographic": clinical,
        "conventional_full": list(dict.fromkeys(morphometry + clinical)),
        "neurojepa_only": [],
        "neurojepa_morphometry": morphometry,
        "neurojepa_conventional": list(dict.fromkeys(morphometry + clinical)),
    }
    if group not in groups:
        raise ValueError(f"Unknown feature group: {group}")
    return groups[group]


def is_feature_group(group: str) -> bool:
    return group in FEATURE_GROUPS


def transition_feature_columns(task: str, group: str, include_acquisition: bool = True) -> list[str]:
    columns = feature_columns(task, group, include_acquisition=include_acquisition)
    mapping = {
        "hippocampus_icv_ratio": "current_hippocampus_icv_ratio",
        "amygdala_icv_ratio": "current_amygdala_icv_ratio",
        "lateral_ventricles_icv_ratio": "current_lateral_ventricles_icv_ratio",
        "inferior_lateral_ventricles_icv_ratio": "current_inferior_lateral_ventricles_icv_ratio",
        "cerebral_cortex_icv_ratio": "current_cerebral_cortex_icv_ratio",
        "age_at_session": "current_age_at_session",
    }
    return [mapping.get(column, column) for column in columns]


def uses_neurojepa(group: str) -> bool:
    return group.startswith("neurojepa_")


def validate_no_target_leakage(task: str, target: str, columns: Iterable[str]) -> None:
    policy = TASK_POLICIES[task]
    selected = {column.lower() for column in columns}
    target_lower = target.lower()
    forbidden = {target_lower, "label", "future_target_value", "target_change"}
    if policy.allow_current_target:
        forbidden.discard("current_target_value")
    leaked = sorted(column for column in selected if column in forbidden or column.startswith(f"future_{target_lower}"))
    if leaked:
        raise ValueError(f"Target leakage for task={task}, target={target}: {', '.join(leaked)}")
