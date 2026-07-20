from __future__ import annotations

import pandas as pd


PAIR_MORPHOMETRY_COLUMNS = (
    "vol_total_intracranial",
    "vol_hippocampus_total",
    "vol_left_amygdala",
    "vol_right_amygdala",
    "vol_lateral_ventricles_total",
    "vol_left_inferior_lateral_ventricle",
    "vol_right_inferior_lateral_ventricle",
    "vol_left_cerebral_cortex",
    "vol_right_cerebral_cortex",
)


def _prefixed_manifest(manifest: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"participant_id", "session_id", "image_id", *PAIR_MORPHOMETRY_COLUMNS}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing pair morphometry columns: {', '.join(missing)}")
    selected = manifest[list(required)].copy()
    if selected.duplicated(["participant_id", "session_id", "image_id"]).any():
        raise ValueError("Manifest contains duplicate keys while enriching pair morphometry.")
    return selected.rename(
        columns={
            "session_id": f"{prefix}_session_id",
            "image_id": f"{prefix}_image_id",
            **{column: f"{prefix}_{column}" for column in PAIR_MORPHOMETRY_COLUMNS},
        }
    )


def enrich_horizon_pairs(
    pairs: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    allowed_horizons: tuple[float, ...] = (12.0, 24.0, 36.0),
) -> pd.DataFrame:
    required = {
        "participant_id",
        "current_session_id",
        "current_image_id",
        "next_session_id",
        "next_image_id",
        "requested_horizon_months",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Horizon pair table is missing columns: {', '.join(missing)}")
    work = pairs.copy()
    horizon = pd.to_numeric(work["requested_horizon_months"], errors="coerce")
    work = work[horizon.isin(allowed_horizons)].copy()
    if work.empty:
        raise ValueError("No horizon pairs remain after applying the locked 12/24/36-month policy.")
    for prefix in ("current", "next"):
        columns = [f"{prefix}_{column}" for column in PAIR_MORPHOMETRY_COLUMNS]
        work = work.drop(columns=columns, errors="ignore")
        prepared = _prefixed_manifest(manifest, prefix)
        work = work.merge(
            prepared,
            on=["participant_id", f"{prefix}_session_id", f"{prefix}_image_id"],
            how="left",
            validate="many_to_one",
        )
    return work.reset_index(drop=True)
