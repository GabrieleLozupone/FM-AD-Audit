from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.paths import PROJECT_ROOT
from neurojepa_adni_audit.probes.downstream import DEFAULT_EMBEDDING_INDEX_PATH
from neurojepa_adni_audit.features.covariates import build_covariate_transform
from neurojepa_adni_audit.features.progression_aligned import (
    STAGE1_FEATURE_MODES,
    filter_stage1_available_rows,
    is_stage1_feature,
    load_stage1_feature_matrix,
    stage1_feature_info,
)
from neurojepa_adni_audit.tables.sessions import strip_prefixed_columns


DEFAULT_PROGRESSION_PAIRS_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_progression_pairs.csv"
DEFAULT_STATE_TRANSITION_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_state_transition_pairs.csv"
DEFAULT_STATE_TRANSITION_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "neurostar_adni_t1w_state_transition_pairs_summary.json"
)
DEFAULT_STATE_TRANSITION_OUTPUT_DIR = PROJECT_ROOT / "results" / "neurojepa_state_transition_probe"
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
DEFAULT_HORIZONS = [6.0, 12.0, 24.0, 36.0]
PAIR_POLICIES = ["adjacent", "horizon_matched", "all_future"]
DEFAULT_STATE_COVARIATE_COLUMNS = [
    "delta_months",
    "current_age_at_session",
    "sex_baseline",
    "education_years_baseline",
    "apoe4_status",
    "current_diagnosis_at_session",
    "current_mmse_at_session",
    "current_adas11_at_session",
    "current_adas13_at_session",
    "current_cdrsb_at_session",
    "current_vol_total_intracranial",
    "current_vol_hippocampus_total",
    "current_vol_lateral_ventricles_total",
    "current_vol_left_hippocampus",
    "current_vol_right_hippocampus",
    "current_vol_left_lateral_ventricle",
    "current_vol_right_lateral_ventricle",
    "current_vol_left_amygdala",
    "current_vol_right_amygdala",
    "current_vol_left_inferior_lateral_ventricle",
    "current_vol_right_inferior_lateral_ventricle",
    "current_vol_left_cerebral_cortex",
    "current_vol_right_cerebral_cortex",
]
PAIR_METADATA_COLUMNS = {
    "pair_id",
    "gap_months",
    "progression_pair_core_complete",
    "progression_pair_volume_ready",
    "diagnosis_changed",
}
CHANGE_FIELD_MAP = {
    "age": "age_at_session",
    "mmse": "mmse_at_session",
    "moca": "moca_at_session",
    "cdrsb": "cdrsb_at_session",
    "faq": "faq_at_session",
    "adas11": "adas11_at_session",
    "adas13": "adas13_at_session",
    "ravlt_immediate": "ravlt_immediate_at_session",
    "abeta": "abeta_at_session",
    "tau": "tau_at_session",
    "ptau": "ptau_at_session",
}


@dataclass
class VectorMetric:
    split: str
    group: str
    n: int
    future_mse: float | None
    future_cosine: float | None
    delta_mse: float | None
    delta_directional_cosine: float | None


def require_transition_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return SimpleImputer, Ridge, Pipeline, StandardScaler, None


def load_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): load_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [load_jsonable(item) for item in value]
    return value


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def prepare_embedding_index(index: pd.DataFrame, prefix: str) -> pd.DataFrame:
    required = {"participant_id", "session_id", "source_image_path", "pooled_embedding_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Embedding index is missing required columns: {missing}")
    keep = [
        "participant_id",
        "session_id",
        "image_id",
        "source_image_path",
        "pooled_embedding_path",
        "token_embedding_path",
        "pooled_dim",
        "n_tokens",
        "token_dim",
    ]
    keep = [column for column in keep if column in index.columns]
    prepared = index[keep].drop_duplicates(["participant_id", "session_id", "source_image_path"]).copy()
    rename = {column: f"{prefix}_{column}" for column in keep if column != "participant_id"}
    return prepared.rename(columns=rename)


def attach_embeddings(pairs: pd.DataFrame, index: pd.DataFrame, prefix: str) -> pd.DataFrame:
    session_col = f"{prefix}_session_id"
    image_path_col = f"{prefix}_processed_t1_path"
    if session_col not in pairs.columns or image_path_col not in pairs.columns:
        raise ValueError(f"Progression pairs must contain `{session_col}` and `{image_path_col}`.")
    prepared = prepare_embedding_index(index, prefix)
    return pairs.merge(
        prepared,
        left_on=["participant_id", session_col, image_path_col],
        right_on=["participant_id", f"{prefix}_session_id", f"{prefix}_source_image_path"],
        how="left",
        suffixes=("", f"_{prefix}_embedding"),
    )


def is_static_session_column(column: str) -> bool:
    if column.startswith("current_") or column.startswith("next_"):
        return False
    if column.endswith("_change"):
        return False
    if column in PAIR_METADATA_COLUMNS:
        return False
    return True


def extract_session_table(pairs: pd.DataFrame) -> pd.DataFrame:
    static_columns = [column for column in pairs.columns if is_static_session_column(column)]
    current = strip_prefixed_columns(pairs, "current", static_columns=static_columns)
    next_rows = strip_prefixed_columns(pairs, "next", static_columns=static_columns)
    sessions = pd.concat([current, next_rows], ignore_index=True, sort=False)
    if "session_month" not in sessions.columns:
        raise ValueError("Progression pairs cannot be converted to sessions: missing session_month.")
    sessions["session_month_numeric"] = pd.to_numeric(sessions["session_month"], errors="coerce")
    sessions = sessions[sessions["session_month_numeric"].notna()].copy()
    key_columns = ["participant_id", "session_id"]
    if "processed_t1_path" in sessions.columns:
        key_columns.append("processed_t1_path")
    sessions = sessions.sort_values(["participant_id", "session_month_numeric", "session_id"]).drop_duplicates(
        key_columns, keep="first"
    )
    return sessions.reset_index(drop=True)


def numeric_delta(next_row: pd.Series, current_row: pd.Series, field: str) -> float | None:
    if field not in next_row.index or field not in current_row.index:
        return None
    current_value = pd.to_numeric(pd.Series([current_row[field]]), errors="coerce").iloc[0]
    next_value = pd.to_numeric(pd.Series([next_row[field]]), errors="coerce").iloc[0]
    if pd.isna(current_value) or pd.isna(next_value):
        return None
    return float(next_value - current_value)


def bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def build_pair_row(
    current: pd.Series,
    future: pd.Series,
    output_columns: list[str],
    pair_policy: str,
    requested_horizon_months: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    current_month = float(current["session_month_numeric"])
    future_month = float(future["session_month_numeric"])
    delta_months = future_month - current_month
    for column in output_columns:
        if column == "pair_id":
            row[column] = f"{current['participant_id']}__{current['session_id']}__{future['session_id']}"
        elif column == "gap_months":
            row[column] = delta_months
        elif column == "progression_pair_core_complete":
            row[column] = bool_value(current.get("has_all_core_files")) and bool_value(future.get("has_all_core_files"))
        elif column == "progression_pair_volume_ready":
            row[column] = bool_value(current.get("target_volume_error_ready")) and bool_value(
                future.get("target_volume_error_ready")
            )
        elif column == "diagnosis_changed":
            row[column] = current.get("diagnosis_at_session") != future.get("diagnosis_at_session")
        elif column.startswith("current_"):
            row[column] = current.get(column.removeprefix("current_"))
        elif column.startswith("next_"):
            row[column] = future.get(column.removeprefix("next_"))
        elif column.endswith("_change"):
            base = column.removesuffix("_change")
            field = CHANGE_FIELD_MAP.get(base, base)
            row[column] = numeric_delta(future, current, field)
        elif column in current.index:
            row[column] = current.get(column)
        else:
            row[column] = None
    row["pair_policy"] = pair_policy
    row["requested_horizon_months"] = requested_horizon_months
    row["horizon_error_months"] = None if requested_horizon_months is None else delta_months - requested_horizon_months
    return row


def generate_all_future_pairs(
    sessions: pd.DataFrame,
    output_columns: list[str],
    min_delta_months: float,
    max_delta_months: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _participant_id, subject_rows in sessions.groupby("participant_id", sort=False):
        subject_rows = subject_rows.sort_values(["session_month_numeric", "session_id"]).reset_index(drop=True)
        for current_index in range(len(subject_rows)):
            current = subject_rows.iloc[current_index]
            for future_index in range(current_index + 1, len(subject_rows)):
                future = subject_rows.iloc[future_index]
                delta = float(future["session_month_numeric"] - current["session_month_numeric"])
                if delta < min_delta_months:
                    continue
                if max_delta_months is not None and delta > max_delta_months:
                    continue
                rows.append(build_pair_row(current, future, output_columns, pair_policy="all_future"))
    return pd.DataFrame(rows)


def generate_horizon_matched_pairs(
    sessions: pd.DataFrame,
    output_columns: list[str],
    horizons: list[float],
    tolerance_months: float,
    min_delta_months: float,
    max_delta_months: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, float]] = set()
    for _participant_id, subject_rows in sessions.groupby("participant_id", sort=False):
        subject_rows = subject_rows.sort_values(["session_month_numeric", "session_id"]).reset_index(drop=True)
        for current_index in range(len(subject_rows)):
            current = subject_rows.iloc[current_index]
            current_month = float(current["session_month_numeric"])
            future_rows = subject_rows.iloc[current_index + 1 :].copy()
            if future_rows.empty:
                continue
            future_rows["delta_from_current"] = future_rows["session_month_numeric"] - current_month
            for horizon in horizons:
                candidates = future_rows[
                    (future_rows["delta_from_current"] >= min_delta_months)
                    & ((future_rows["delta_from_current"] - float(horizon)).abs() <= tolerance_months)
                ].copy()
                if max_delta_months is not None:
                    candidates = candidates[candidates["delta_from_current"] <= max_delta_months].copy()
                if candidates.empty:
                    continue
                candidates["horizon_abs_error"] = (candidates["delta_from_current"] - float(horizon)).abs()
                future = candidates.sort_values(["horizon_abs_error", "delta_from_current", "session_id"]).iloc[0]
                key = (
                    str(current["participant_id"]),
                    str(current["session_id"]),
                    str(future["session_id"]),
                    float(horizon),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    build_pair_row(
                        current,
                        future,
                        output_columns,
                        pair_policy="horizon_matched",
                        requested_horizon_months=float(horizon),
                    )
                )
    return pd.DataFrame(rows)


def generate_transition_pairs(
    pairs: pd.DataFrame,
    pair_policy: str,
    min_delta_months: float,
    max_delta_months: float | None,
    horizons: list[float],
    horizon_tolerance_months: float,
) -> pd.DataFrame:
    if pair_policy == "adjacent":
        generated = pairs.copy()
        generated["pair_policy"] = "adjacent"
        generated["requested_horizon_months"] = pd.NA
        generated["horizon_error_months"] = pd.NA
        return generated
    sessions = extract_session_table(pairs)
    output_columns = list(pairs.columns)
    if pair_policy == "all_future":
        generated = generate_all_future_pairs(sessions, output_columns, min_delta_months, max_delta_months)
    elif pair_policy == "horizon_matched":
        generated = generate_horizon_matched_pairs(
            sessions=sessions,
            output_columns=output_columns,
            horizons=horizons,
            tolerance_months=horizon_tolerance_months,
            min_delta_months=min_delta_months,
            max_delta_months=max_delta_months,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported pair policy: {pair_policy}")
    if generated.empty:
        raise ValueError(f"No transition pairs were generated for pair policy `{pair_policy}`.")
    return generated.reset_index(drop=True)


def make_horizon_bins(months: pd.Series) -> pd.Series:
    values = pd.to_numeric(months, errors="coerce")
    bins = [-np.inf, 9.0, 18.0, 30.0, 42.0, np.inf]
    labels = ["0-9m", "9-18m", "18-30m", "30-42m", "42m+"]
    return pd.cut(values, bins=bins, labels=labels)


def summarize_transition_table(table: pd.DataFrame) -> dict[str, Any]:
    subject_sets = {
        split: set(table.loc[table["subject_split"] == split, "participant_id"].astype(str))
        for split in ("train", "val", "test")
    }
    overlap = {
        "train_val": sorted(subject_sets["train"] & subject_sets["val"]),
        "train_test": sorted(subject_sets["train"] & subject_sets["test"]),
        "val_test": sorted(subject_sets["val"] & subject_sets["test"]),
    }
    missing_columns = [
        "current_pooled_embedding_path",
        "next_pooled_embedding_path",
        "current_vol_hippocampus_total",
        "next_vol_hippocampus_total",
        "current_vol_lateral_ventricles_total",
        "next_vol_lateral_ventricles_total",
        "current_mmse_at_session",
        "next_mmse_at_session",
        "current_adas11_at_session",
        "next_adas11_at_session",
        "current_adas13_at_session",
        "next_adas13_at_session",
    ]
    available_missing = {
        column: int(table[column].isna().sum()) for column in missing_columns if column in table.columns
    }
    order_ok = (
        pd.to_numeric(table["next_session_month"], errors="coerce")
        > pd.to_numeric(table["current_session_month"], errors="coerce")
    )
    summary = {
        "n_rows": int(len(table)),
        "n_subjects": int(table["participant_id"].nunique()),
        "split_counts": table["subject_split"].value_counts(dropna=False).to_dict(),
        "subject_counts_by_split": table.groupby("subject_split")["participant_id"].nunique().to_dict(),
        "subject_overlap": {key: {"n": len(value), "examples": value[:10]} for key, value in overlap.items()},
        "sequence_order": {
            "n_valid_future_after_current": int(order_ok.sum()),
            "n_invalid_or_missing": int((~order_ok.fillna(False)).sum()),
        },
        "delta_months": {
            "min": float(pd.to_numeric(table["delta_months"], errors="coerce").min()),
            "median": float(pd.to_numeric(table["delta_months"], errors="coerce").median()),
            "max": float(pd.to_numeric(table["delta_months"], errors="coerce").max()),
        },
        "horizon_bin_counts": make_horizon_bins(table["delta_months"]).value_counts(dropna=False).to_dict(),
        "pair_policy_counts": table["pair_policy"].value_counts(dropna=False).to_dict()
        if "pair_policy" in table.columns
        else {},
        "requested_horizon_counts": table["requested_horizon_months"].value_counts(dropna=False).to_dict()
        if "requested_horizon_months" in table.columns
        else {},
        "horizon_error_months": {
            "min": float(pd.to_numeric(table["horizon_error_months"], errors="coerce").min()),
            "median": float(pd.to_numeric(table["horizon_error_months"], errors="coerce").median()),
            "max": float(pd.to_numeric(table["horizon_error_months"], errors="coerce").max()),
        }
        if "horizon_error_months" in table.columns
        and pd.to_numeric(table["horizon_error_months"], errors="coerce").notna().any()
        else {},
        "baseline_diagnosis_counts": table["current_diagnosis_at_session"].value_counts(dropna=False).to_dict()
        if "current_diagnosis_at_session" in table.columns
        else {},
        "missing_counts": available_missing,
        "rows_with_current_and_next_pooled_embeddings": int(
            table["current_pooled_embedding_path"].notna().sum()
            if "current_pooled_embedding_path" in table.columns
            else 0
        ),
    }
    if {"current_pooled_embedding_path", "next_pooled_embedding_path"}.issubset(table.columns):
        summary["rows_with_current_and_next_pooled_embeddings"] = int(
            (table["current_pooled_embedding_path"].notna() & table["next_pooled_embedding_path"].notna()).sum()
        )
    return load_jsonable(summary)


def build_state_transition_table(
    progression_pairs_path: Path,
    embedding_index_path: Path,
    output_path: Path,
    summary_path: Path,
    pair_policy: str,
    min_delta_months: float,
    max_delta_months: float | None,
    horizons: list[float],
    horizon_tolerance_months: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = read_csv(progression_pairs_path)
    index = read_csv(embedding_index_path)
    pairs = generate_transition_pairs(
        pairs=pairs,
        pair_policy=pair_policy,
        min_delta_months=min_delta_months,
        max_delta_months=max_delta_months,
        horizons=horizons,
        horizon_tolerance_months=horizon_tolerance_months,
    )
    table = attach_embeddings(pairs, index, "current")
    table = attach_embeddings(table, index, "next")
    table["delta_months"] = pd.to_numeric(table["gap_months"], errors="coerce")
    if "progression_pair_core_complete" in table.columns:
        table = table[table["progression_pair_core_complete"].astype("boolean").fillna(False)].copy()
    if "subject_split" in table.columns:
        table = table[table["subject_split"].isin(["train", "val", "test"])].copy()
    table = table.reset_index(drop=True)
    summary = summarize_transition_table(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return table, summary


def load_matrix(paths: pd.Series) -> np.ndarray:
    vectors = [np.load(Path(str(path))).astype(np.float32, copy=False).reshape(-1) for path in paths]
    return np.stack(vectors, axis=0)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.full(a.shape[0], np.nan, dtype=np.float32)
    valid = denom > 0
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return out


def safe_nanmean(values: np.ndarray) -> float | None:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def vector_metrics(
    split: str,
    group: str,
    current: np.ndarray,
    future: np.ndarray,
    predicted_future: np.ndarray,
) -> VectorMetric:
    if current.shape[0] == 0:
        return VectorMetric(split, group, 0, None, None, None, None)
    true_delta = future - current
    predicted_delta = predicted_future - current
    return VectorMetric(
        split=split,
        group=group,
        n=int(current.shape[0]),
        future_mse=float(np.mean((future - predicted_future) ** 2)),
        future_cosine=safe_nanmean(rowwise_cosine(future, predicted_future)),
        delta_mse=float(np.mean((true_delta - predicted_delta) ** 2)),
        delta_directional_cosine=safe_nanmean(rowwise_cosine(true_delta, predicted_delta)),
    )


def metrics_by_split_and_group(
    table: pd.DataFrame,
    current: np.ndarray,
    future: np.ndarray,
    predicted_future: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_values = table["subject_split"].astype(str).to_numpy()
    for split in ("train", "val", "test"):
        mask = split_values == split
        rows.append(asdict(vector_metrics(split, "all", current[mask], future[mask], predicted_future[mask])))
        if "current_diagnosis_at_session" in table.columns:
            for diagnosis in sorted(table.loc[mask, "current_diagnosis_at_session"].dropna().astype(str).unique()):
                group_mask = mask & (table["current_diagnosis_at_session"].astype(str).to_numpy() == diagnosis)
                rows.append(
                    asdict(vector_metrics(split, f"diagnosis={diagnosis}", current[group_mask], future[group_mask], predicted_future[group_mask]))
                )
        horizon = make_horizon_bins(table["delta_months"]).astype(str).to_numpy()
        for label in ["0-9m", "9-18m", "18-30m", "30-42m", "42m+"]:
            group_mask = mask & (horizon == label)
            if group_mask.any():
                rows.append(
                    asdict(vector_metrics(split, f"horizon={label}", current[group_mask], future[group_mask], predicted_future[group_mask]))
                )
    return rows


def build_transition_features(
    table: pd.DataFrame,
    feature: str,
    current_embeddings: np.ndarray,
    covariate_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    train_mask = table["subject_split"].to_numpy() == "train"
    cov_info: dict[str, Any] = {"included": False, "numeric_columns": [], "categorical_columns": []}
    cov_all = None
    stage1_covariate_modes = {f"{mode}_covariates_morphometry": mode for mode in STAGE1_FEATURE_MODES}
    stage1_feature = stage1_covariate_modes.get(feature, feature)
    if feature in {"covariates_morphometry", "jepa_covariates_morphometry"} or feature in stage1_covariate_modes:
        _transform, numeric, categorical, cov_all = build_covariate_transform(
            train_rows=table[train_mask],
            all_rows=table,
            covariate_columns=covariate_columns,
        )
        if cov_all is None:
            raise ValueError("No usable state covariates were found.")
        cov_info = {"included": True, "numeric_columns": numeric, "categorical_columns": categorical}
    if feature == "jepa_only":
        return current_embeddings.astype(np.float32), cov_info
    if feature == "covariates_morphometry":
        return cov_all.astype(np.float32), cov_info
    if feature == "jepa_covariates_morphometry":
        return np.concatenate([current_embeddings, cov_all], axis=1).astype(np.float32), cov_info
    if is_stage1_feature(stage1_feature):
        current_stage1 = load_stage1_feature_matrix(table, stage1_feature, image_id_column="current_image_id")
        cov_info["stage1"] = stage1_feature_info(stage1_feature)
        if feature in stage1_covariate_modes:
            return np.concatenate([current_stage1, cov_all], axis=1).astype(np.float32), cov_info
        return current_stage1.astype(np.float32), cov_info
    raise ValueError(f"Unsupported feature mode: {feature}")


def select_alpha(
    x_all: np.ndarray,
    y_delta: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    alphas: list[float],
) -> tuple[float, list[dict[str, float]]]:
    SimpleImputer, Ridge, Pipeline, StandardScaler, _ = require_transition_sklearn()
    history = []
    best_alpha = float(alphas[0])
    best_mse = np.inf
    for alpha in alphas:
        model = Pipeline(
            [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(alpha)))]
        )
        model.fit(x_all[train_mask], y_delta[train_mask])
        pred = model.predict(x_all[val_mask])
        mse = float(np.mean((y_delta[val_mask] - pred) ** 2))
        history.append({"alpha": float(alpha), "val_delta_mse": mse})
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha, history


def group_mean_delta_predictions(
    table: pd.DataFrame,
    current: np.ndarray,
    future: np.ndarray,
    group_column: str = "current_diagnosis_at_session",
) -> np.ndarray:
    train_mask = table["subject_split"].to_numpy() == "train"
    deltas = future - current
    global_delta = deltas[train_mask].mean(axis=0)
    predicted_delta = np.repeat(global_delta[None, :], len(table), axis=0)
    if group_column not in table.columns:
        return current + predicted_delta
    groups = table[group_column].astype(str).fillna("missing")
    train_groups = groups[train_mask]
    for group in sorted(train_groups.unique()):
        group_train_mask = train_mask & (groups.to_numpy() == group)
        if group_train_mask.any():
            predicted_delta[groups.to_numpy() == group] = deltas[group_train_mask].mean(axis=0)
    return current + predicted_delta


def train_state_transition_probe(
    table_path: Path,
    output_dir: Path,
    feature: str,
    covariate_columns: list[str],
    alphas: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    table = read_csv(table_path)
    required = {"subject_split", "current_pooled_embedding_path", "next_pooled_embedding_path"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"State transition table is missing required columns: {missing}")
    table = table[
        table["subject_split"].isin(["train", "val", "test"])
        & table["current_pooled_embedding_path"].notna()
        & table["next_pooled_embedding_path"].notna()
    ].copy()
    table = table[
        table["current_pooled_embedding_path"].map(lambda value: Path(str(value)).exists())
        & table["next_pooled_embedding_path"].map(lambda value: Path(str(value)).exists())
    ].reset_index(drop=True)
    stage1_covariate_modes = {f"{mode}_covariates_morphometry": mode for mode in STAGE1_FEATURE_MODES}
    if is_stage1_feature(feature) or feature in stage1_covariate_modes:
        table = filter_stage1_available_rows(table, image_id_column="current_image_id")
    if table.empty:
        raise ValueError("No rows have usable current and next pooled Neuro-JEPA embeddings.")
    split_values = table["subject_split"].to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    if not train_mask.any() or not val_mask.any() or not (split_values == "test").any():
        raise ValueError("Train, val, and test must all contain at least one usable row.")

    current = load_matrix(table["current_pooled_embedding_path"])
    future = load_matrix(table["next_pooled_embedding_path"])
    y_delta = future - current
    x_all, cov_info = build_transition_features(table, feature, current, covariate_columns)

    best_alpha, alpha_history = select_alpha(x_all, y_delta, train_mask, val_mask, alphas)
    SimpleImputer, Ridge, Pipeline, StandardScaler, _ = require_transition_sklearn()
    model = Pipeline(
        [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))]
    )
    model.fit(x_all[train_mask], y_delta[train_mask])
    predicted_delta = model.predict(x_all)
    predicted_future = current + predicted_delta
    identity_future = current.copy()
    group_future = group_mean_delta_predictions(table, current, future)

    summary = {
        "task": "neurojepa_state_transition",
        "target": "future_pooled_neurojepa_embedding",
        "model": "ridge_residual_delta",
        "feature": feature,
        "covariates": cov_info,
        "n_rows": int(len(table)),
        "n_input_features": int(x_all.shape[1]),
        "embedding_dim": int(current.shape[1]),
        "selected_alpha": float(best_alpha),
        "alpha_selection_metric": "validation_delta_mse",
        "alpha_history": alpha_history,
        "metrics": metrics_by_split_and_group(table, current, future, predicted_future),
        "identity_baseline_metrics": metrics_by_split_and_group(table, current, future, identity_future),
        "diagnosis_mean_delta_baseline_metrics": metrics_by_split_and_group(table, current, future, group_future),
    }

    prediction_columns = [
        "pair_id",
        "participant_id",
        "current_session_id",
        "next_session_id",
        "subject_split",
        "delta_months",
        "current_diagnosis_at_session",
        "next_diagnosis_at_session",
    ]
    prediction_columns = [column for column in prediction_columns if column in table.columns]
    predictions = table[prediction_columns].copy()
    predictions["model_delta_mse"] = np.mean((y_delta - predicted_delta) ** 2, axis=1)
    predictions["identity_delta_mse"] = np.mean(y_delta**2, axis=1)
    predictions["diagnosis_mean_delta_mse"] = np.mean((future - group_future) ** 2, axis=1)
    predictions["model_future_cosine"] = rowwise_cosine(future, predicted_future)
    predictions["identity_future_cosine"] = rowwise_cosine(future, identity_future)
    predictions["diagnosis_mean_future_cosine"] = rowwise_cosine(future, group_future)
    predictions["model_delta_directional_cosine"] = rowwise_cosine(y_delta, predicted_delta)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(load_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    table.to_csv(output_dir / "task_table.csv", index=False)
    return load_jsonable(summary), predictions


def build_run(args: Any) -> None:
    table, summary = build_state_transition_table(
        progression_pairs_path=args.progression_pairs_path,
        embedding_index_path=args.embedding_index_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        pair_policy=args.pair_policy,
        min_delta_months=args.min_delta_months,
        max_delta_months=args.max_delta_months,
        horizons=args.horizons,
        horizon_tolerance_months=args.horizon_tolerance_months,
    )
    print(f"Wrote {len(table)} rows to {args.output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def train_run(args: Any) -> None:
    features = args.feature or ["jepa_only"]
    comparison_rows = []
    for feature in features:
        run_dir = args.output_dir / feature
        summary, _predictions = train_state_transition_probe(
            table_path=args.table_path,
            output_dir=run_dir,
            feature=feature,
            covariate_columns=args.covariate_columns,
            alphas=args.alpha,
        )
        test_model = next(row for row in summary["metrics"] if row["split"] == "test" and row["group"] == "all")
        test_identity = next(
            row for row in summary["identity_baseline_metrics"] if row["split"] == "test" and row["group"] == "all"
        )
        comparison_rows.append(
            {
                "feature": feature,
                "run_dir": str(run_dir),
                "test_future_mse": test_model["future_mse"],
                "test_delta_mse": test_model["delta_mse"],
                "test_delta_directional_cosine": test_model["delta_directional_cosine"],
                "identity_test_future_mse": test_identity["future_mse"],
                "identity_test_delta_mse": test_identity["delta_mse"],
                "delta_mse_improvement_vs_identity": test_identity["delta_mse"] - test_model["delta_mse"],
            }
        )
        print(f"Wrote Task E run: {run_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "comparison.csv", index=False)
    print(f"Wrote comparison: {args.output_dir / 'comparison.csv'}")
