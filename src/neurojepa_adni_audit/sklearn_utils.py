from __future__ import annotations

from typing import Any


def require_tabular_sklearn() -> tuple[Any, Any, Any, Any, Any, None]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return ColumnTransformer, SimpleImputer, Pipeline, OneHotEncoder, StandardScaler, None


def require_regression_sklearn() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return SimpleImputer, Ridge, mean_absolute_error, mean_squared_error, r2_score, Pipeline, StandardScaler


def require_ridge_logistic_sklearn() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return SimpleImputer, LogisticRegression, Ridge, Pipeline, StandardScaler
