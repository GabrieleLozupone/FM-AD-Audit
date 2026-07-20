from __future__ import annotations

import pandas as pd


PREVIOUS_COPY_COLUMNS = sorted(
    {
        "session_id",
        "image_id",
        "session_month",
        "pooled_embedding_path",
        "age_at_session",
        "diagnosis_at_session",
        "mmse_at_session",
        "adas11_at_session",
        "adas13_at_session",
        "cdrsb_at_session",
        "vol_total_intracranial",
        "vol_hippocampus_total",
        "vol_lateral_ventricles_total",
        "vol_left_hippocampus",
        "vol_right_hippocampus",
        "vol_left_lateral_ventricle",
        "vol_right_lateral_ventricle",
        "vol_left_amygdala",
        "vol_right_amygdala",
        "vol_left_inferior_lateral_ventricle",
        "vol_right_inferior_lateral_ventricle",
        "vol_left_cerebral_cortex",
        "vol_right_cerebral_cortex",
    }
)


def strip_prefixed_columns(
    table: pd.DataFrame,
    prefix: str,
    keep_columns: list[str] | None = None,
    *,
    static_columns: list[str] | None = None,
) -> pd.DataFrame:
    static = static_columns if static_columns is not None else ["participant_id"]
    if keep_columns is None:
        prefixed_columns = [column for column in table.columns if column.startswith(f"{prefix}_")]
    else:
        prefixed_columns = [f"{prefix}_{column}" for column in keep_columns if f"{prefix}_{column}" in table.columns]
    available = list(dict.fromkeys([column for column in static if column in table.columns] + prefixed_columns))
    out = table[available].copy()
    rename = {column: column.removeprefix(f"{prefix}_") for column in prefixed_columns if column in out.columns}
    return out.rename(columns=rename)


def build_session_table(table: pd.DataFrame) -> pd.DataFrame:
    current = strip_prefixed_columns(table, "current", PREVIOUS_COPY_COLUMNS)
    next_rows = strip_prefixed_columns(table, "next", PREVIOUS_COPY_COLUMNS)
    sessions = pd.concat([current, next_rows], ignore_index=True, sort=False)
    sessions["session_month_numeric"] = pd.to_numeric(sessions["session_month"], errors="coerce")
    sessions = sessions[sessions["session_month_numeric"].notna()].copy()
    key_columns = ["participant_id", "session_id"]
    sessions = sessions.sort_values(["participant_id", "session_month_numeric", "session_id"]).drop_duplicates(
        key_columns, keep="first"
    )
    return sessions.reset_index(drop=True)
