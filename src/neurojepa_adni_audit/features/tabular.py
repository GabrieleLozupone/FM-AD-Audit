from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from neurojepa_adni_audit.features.covariates import build_covariate_transform
from neurojepa_adni_audit.features.progression_aligned import (
    is_stage1_feature,
    load_stage1_feature_matrix,
    stage1_feature_info,
)
from neurojepa_adni_audit.sklearn_utils import require_tabular_sklearn


def fit_feature_transform(train_mask: np.ndarray, x_all: np.ndarray) -> tuple[Any, np.ndarray]:
    _ColumnTransformer, SimpleImputer, Pipeline, _OneHotEncoder, StandardScaler, _ = require_tabular_sklearn()
    transform = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    transform.fit(x_all[train_mask])
    return transform, transform.transform(x_all).astype(np.float32)


def build_pooled_or_covariate_feature_matrix(
    table: pd.DataFrame,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
    load_pooled_matrix: Callable[[pd.DataFrame], np.ndarray],
    *,
    standardize_pooled: bool,
    covariate_only_error: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_mask = table["subject_split"].to_numpy() == "train"
    cov_info: dict[str, Any] = {"included": False, "numeric_columns": [], "categorical_columns": []}
    if feature == "covariates":
        _transform, numeric, categorical, cov_all = build_covariate_transform(
            train_rows=table[train_mask],
            all_rows=table,
            covariate_columns=covariate_columns,
        )
        if cov_all is None:
            raise ValueError(covariate_only_error)
        cov_info = {"included": True, "numeric_columns": numeric, "categorical_columns": categorical}
        return cov_all.astype(np.float32), cov_info

    if feature != "pooled" and not is_stage1_feature(feature):
        raise ValueError(f"Unsupported feature mode: {feature}")
    x_all = load_stage1_feature_matrix(table, feature) if is_stage1_feature(feature) else load_pooled_matrix(table)
    if standardize_pooled:
        _pooled_transform, x_all = fit_feature_transform(train_mask, x_all)
    if is_stage1_feature(feature):
        cov_info["stage1"] = stage1_feature_info(feature)
    if include_covariates:
        _transform, numeric, categorical, cov_all = build_covariate_transform(
            train_rows=table[train_mask],
            all_rows=table,
            covariate_columns=covariate_columns,
        )
        if cov_all is not None:
            x_all = np.concatenate([x_all, cov_all], axis=1)
            cov_info = {"included": True, "numeric_columns": numeric, "categorical_columns": categorical}
    return x_all.astype(np.float32), cov_info
