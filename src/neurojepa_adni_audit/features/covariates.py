from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


FORBIDDEN_COVARIATE_TOKENS = [
    "diagnosis",
    "diagnosis_norm",
    "label",
    "subject_split",
    "dxsum",
    "dxmci",
    "dxad",
    "dxnorm",
    "adnimerge_dx",
    "bids_diagnosis",
]


def require_covariate_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return ColumnTransformer, SimpleImputer, Pipeline, OneHotEncoder, StandardScaler


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .str.replace(r"(?i)\s*T$", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def infer_numeric_and_categorical(table: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        if column not in table.columns:
            continue
        converted = coerce_numeric_series(table[column])
        non_null_original = table[column].notna().sum()
        if non_null_original > 0 and converted.notna().sum() >= max(1, int(0.8 * non_null_original)):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def prepare_covariate_frame(rows: pd.DataFrame, columns: list[str], numeric_columns: list[str]) -> pd.DataFrame:
    frame = rows[columns].copy()
    for column in numeric_columns:
        frame[column] = coerce_numeric_series(frame[column])
    return frame


def build_covariate_transform(
    train_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
    covariate_columns: list[str],
) -> tuple[Any | None, list[str], list[str], np.ndarray | None]:
    ColumnTransformer, SimpleImputer, Pipeline, OneHotEncoder, StandardScaler = require_covariate_sklearn()
    available = [column for column in covariate_columns if column in all_rows.columns]
    if not available:
        return None, [], [], None
    numeric, categorical = infer_numeric_and_categorical(train_rows, available)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        return None, [], [], None
    transform = ColumnTransformer(transformers)
    train_covariates = prepare_covariate_frame(train_rows, available, numeric)
    all_covariates = prepare_covariate_frame(all_rows, available, numeric)
    transform.fit(train_covariates)
    covariates = transform.transform(all_covariates).astype(np.float32)
    return transform, numeric, categorical, covariates


def validate_covariate_columns(covariate_columns: list[str]) -> None:
    forbidden = []
    for column in covariate_columns:
        lowered = column.lower()
        if any(token in lowered for token in FORBIDDEN_COVARIATE_TOKENS):
            forbidden.append(column)
    if forbidden:
        raise ValueError(
            "Refusing label-like covariates for downstream probing: "
            + ", ".join(forbidden)
            + ". Use imaging, demographic, acquisition, or explicitly justified non-diagnostic covariates instead."
        )
