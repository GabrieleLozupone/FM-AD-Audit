from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from neurojepa_adni_audit.probes.downstream import first_baseline_rows, normalize_diagnosis


def _ids(values: Iterable[object]) -> set[str]:
    return {str(value) for value in values}


def _flow(
    manifest: pd.DataFrame,
    eligible: pd.DataFrame,
    ordered_reasons: list[tuple[str, set[str]]],
) -> dict:
    source = _ids(manifest["participant_id"])
    eligible_ids = _ids(eligible["participant_id"])
    remaining = source - eligible_ids
    reasons = {}
    for label, candidates in ordered_reasons:
        selected = remaining & candidates
        reasons[label] = len(selected)
        remaining -= selected
    reasons["other_prespecified_or_technical_exclusion"] = len(remaining)
    if len(eligible_ids) + sum(reasons.values()) != len(source):
        raise ValueError("Hierarchical participant exclusion flow does not reconcile.")
    return {
        "source_participants": len(source),
        "eligible_participants": len(eligible_ids),
        "excluded_participants": len(source - eligible_ids),
        "hierarchical_exclusion_reasons": reasons,
        "reconciled": True,
    }


def ad_vs_cn_flow(manifest: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    baseline = first_baseline_rows(manifest).copy()
    baseline["diagnosis_norm"] = baseline["diagnosis_at_session"].map(normalize_diagnosis)
    return _flow(
        manifest,
        eligible,
        [
            ("first_mri_diagnosis_mci", _ids(baseline.loc[baseline["diagnosis_norm"] == "MCI", "participant_id"])),
            ("first_mri_diagnosis_missing", _ids(baseline.loc[baseline["diagnosis_norm"].isna(), "participant_id"])),
            (
                "first_mri_diagnosis_other",
                _ids(
                    baseline.loc[
                        baseline["diagnosis_norm"].notna()
                        & ~baseline["diagnosis_norm"].isin(["CN", "AD", "MCI"]),
                        "participant_id",
                    ]
                ),
            ),
        ],
    )


def same_visit_cognition_flow(manifest: pd.DataFrame, eligible: pd.DataFrame, target: str) -> dict:
    available = pd.to_numeric(manifest[target], errors="coerce").notna()
    any_available = _ids(manifest.loc[available, "participant_id"])
    source = _ids(manifest["participant_id"])
    flow = _flow(manifest, eligible, [("no_target_available_at_any_session", source - any_available)])
    flow["source_rows"] = len(manifest)
    flow["eligible_rows"] = len(eligible)
    flow["rows_missing_target"] = int((~available).sum())
    return flow


def future_cognition_flow(manifest: pd.DataFrame, eligible: pd.DataFrame, target: str) -> dict:
    available = pd.to_numeric(manifest[target], errors="coerce").notna()
    any_available = _ids(manifest.loc[available, "participant_id"])
    source = _ids(manifest["participant_id"])
    return _flow(
        manifest,
        eligible,
        [
            ("no_target_available_at_any_session", source - any_available),
            ("no_strictly_later_target_in_horizon_window", any_available - _ids(eligible["participant_id"])),
        ],
    )


def _first_mci_characteristics(manifest: pd.DataFrame) -> pd.DataFrame:
    work = manifest.copy()
    work["month"] = pd.to_numeric(work["session_month"], errors="coerce")
    work["diagnosis_norm"] = work["diagnosis_at_session"].map(normalize_diagnosis)
    work = work[work["month"].notna()].sort_values(["participant_id", "month", "session_id"])
    rows = []
    for participant_id, subject in work.groupby("participant_id", sort=False):
        mci = subject[subject["diagnosis_norm"] == "MCI"]
        if mci.empty:
            continue
        anchor = float(mci.iloc[0]["month"])
        later = subject[subject["month"] > anchor]
        later_ad = later[later["diagnosis_norm"] == "AD"]
        first_ad = float(later_ad["month"].min() - anchor) if not later_ad.empty else np.nan
        rows.append(
            {
                "participant_id": str(participant_id),
                "followup": float(subject["month"].max() - anchor),
                "first_ad": first_ad,
            }
        )
    return pd.DataFrame(rows)


def mci_conversion_flow(manifest: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    characteristics = _first_mci_characteristics(manifest)
    with_mci = _ids(characteristics["participant_id"]) if not characteristics.empty else set()
    source = _ids(manifest["participant_id"])
    late = _ids(
        characteristics.loc[characteristics["first_ad"].notna() & (characteristics["first_ad"] > 36), "participant_id"]
    )
    short = _ids(
        characteristics.loc[characteristics["first_ad"].isna() & (characteristics["followup"] < 36), "participant_id"]
    )
    return _flow(
        manifest,
        eligible,
        [
            ("no_mci_anchor", source - with_mci),
            ("first_ad_after_36_month_window", late),
            ("no_ad_and_less_than_36_month_followup", short),
        ],
    )


def survival_flow(manifest: pd.DataFrame, eligible: pd.DataFrame) -> dict:
    characteristics = _first_mci_characteristics(manifest)
    with_mci = _ids(characteristics["participant_id"]) if not characteristics.empty else set()
    source = _ids(manifest["participant_id"])
    no_positive_followup = _ids(
        characteristics.loc[characteristics["followup"] <= 0, "participant_id"]
    )
    return _flow(
        manifest,
        eligible,
        [("no_mci_anchor", source - with_mci), ("no_positive_followup_after_mci_anchor", no_positive_followup)],
    )


def future_morphometry_flow(
    manifest: pd.DataFrame, candidate_pairs: pd.DataFrame, eligible: pd.DataFrame
) -> dict:
    candidates = _ids(candidate_pairs["participant_id"])
    source = _ids(manifest["participant_id"])
    return _flow(
        manifest,
        eligible,
        [
            ("no_horizon_matched_pair", source - candidates),
            ("pair_missing_valid_current_or_future_target", candidates - _ids(eligible["participant_id"])),
        ],
    )
