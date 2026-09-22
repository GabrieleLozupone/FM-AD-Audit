from __future__ import annotations

import warnings
from collections.abc import Callable
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from neurojepa_adni_audit.nested_cv import NestedFoldSpec
from neurojepa_adni_audit.nested_training import attach_nested_split, build_fold_matrices
from neurojepa_adni_audit.survival.ipcw import (
    censoring_survival_at,
    cumulative_dynamic_auc,
    ipcw_brier_score,
    uno_c_index,
)


def _require_lifelines() -> tuple[Any, Any, Any]:
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.statistics import proportional_hazard_test
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `lifelines`.") from exc
    return CoxPHFitter, KaplanMeierFitter, proportional_hazard_test


def _convergence_warning_messages(caught: list[Any]) -> list[str]:
    try:
        from lifelines.exceptions import ConvergenceWarning
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency `lifelines`.") from exc
    return [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]


def _cox_frame(features: np.ndarray, rows: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(features, columns=[f"x{index:04d}" for index in range(features.shape[1])])
    frame["duration_months"] = rows["duration_months"].to_numpy(dtype=float)
    frame["event"] = rows["event"].to_numpy(dtype=int)
    return frame


def _fit_final_cox(
    *,
    fitter_class: Any,
    development_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    emit: Callable[[str], None],
    show_fit_progress: bool,
) -> tuple[Any, pd.Series, list[Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        penalizer = float(candidate["penalizer"])
        l1_ratio = float(candidate["l1_ratio"])
        model = fitter_class(penalizer=penalizer, l1_ratio=l1_ratio)
        emit(
            f"COX FINAL START candidate_rank={rank}/{len(candidates)} "
            f"penalizer={penalizer:g} l1_ratio={l1_ratio:g} "
            f"n={len(development_frame)} features={development_frame.shape[1] - 2}"
        )
        started = monotonic()
        caught: list[Any] = []
        error: str | None = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(
                    development_frame,
                    duration_col="duration_months",
                    event_col="event",
                    show_progress=show_fit_progress,
                    fit_options={"max_steps": 500},
                )
            convergence_warnings = _convergence_warning_messages(caught)
            if convergence_warnings:
                error = " | ".join(convergence_warnings)
        except Exception as exc:
            error = str(exc)
        attempts.append(
            {
                "candidate_rank": rank,
                "penalizer": penalizer,
                "l1_ratio": l1_ratio,
                "mean_inner_uno_c_36m": float(candidate["mean_inner_uno_c_36m"]),
                "status": "ok" if error is None else "failed",
                "error": error,
                "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
            }
        )
        elapsed = monotonic() - started
        if error is None:
            emit(
                f"COX FINAL DONE candidate_rank={rank}/{len(candidates)} "
                f"elapsed_seconds={elapsed:.1f}"
            )
            return model, candidate, caught, attempts
        emit(
            f"COX FINAL FAILED candidate_rank={rank}/{len(candidates)} "
            f"elapsed_seconds={elapsed:.1f} error={error}"
        )
    errors = " || ".join(
        f"rank {item['candidate_rank']} (penalizer={item['penalizer']:g}, "
        f"l1_ratio={item['l1_ratio']:g}): {item['error']}"
        for item in attempts
    )
    raise RuntimeError(f"All final Cox refit candidates failed to converge: {errors}")


def _calibration_rows(
    rows: pd.DataFrame,
    survival_probabilities: np.ndarray,
    horizons: list[float],
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    _CoxPHFitter, KaplanMeierFitter, _ph_test = _require_lifelines()
    output = []
    predicted_risk_at_last = 1.0 - survival_probabilities[-1]
    ranks = pd.Series(predicted_risk_at_last).rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, len(rows)), labels=False, duplicates="drop")
    for bin_index in sorted(pd.unique(bins)):
        mask = np.asarray(bins == bin_index)
        km = KaplanMeierFitter().fit(
            rows.loc[mask, "duration_months"], event_observed=rows.loc[mask, "event"]
        )
        for horizon_index, horizon in enumerate(horizons):
            observed_risk = 1.0 - float(km.predict(horizon))
            output.append(
                {
                    "horizon_months": float(horizon),
                    "risk_bin": int(bin_index),
                    "n": int(mask.sum()),
                    "events": int(rows.loc[mask, "event"].sum()),
                    "mean_predicted_risk": float(
                        np.mean(1.0 - survival_probabilities[horizon_index, mask])
                    ),
                    "kaplan_meier_observed_risk": observed_risk,
                }
            )
    return output


def run_nested_survival_fold(
    *,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    full_pooled_matrix: np.ndarray | None,
    repeat: str,
    outer_fold: int,
    feature_group: str,
    penalizers: list[float],
    l1_ratios: list[float],
    horizons: list[float],
    tau: float = 36.0,
    log: Callable[[str], None] | None = None,
    show_fit_progress: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CoxPHFitter, _KaplanMeierFitter, proportional_hazard_test = _require_lifelines()
    outer = attach_nested_split(table, assignments, NestedFoldSpec(repeat, outer_fold))
    development = outer[outer["subject_split"] == "train"].copy()
    test = outer[outer["subject_split"] == "test"].copy()
    if min(int(development["event"].sum()), int(test["event"].sum())) == 0:
        raise ValueError("Outer development and test must both contain survival events.")
    n_inner = int(
        assignments.loc[
            (assignments["repeat"] == repeat)
            & (assignments["selected_outer_fold"] == outer_fold),
            "inner_folds",
        ].iloc[0]
    )

    emit = log if log is not None else lambda _message: None
    total_inner_fits = len(penalizers) * len(l1_ratios) * n_inner
    fit_index = 0
    histories = []
    for penalizer in map(float, penalizers):
        for l1_ratio in map(float, l1_ratios):
            for inner_fold in range(n_inner):
                fit_index += 1
                fit_started = monotonic()
                fit_label = (
                    f"inner_fit={fit_index}/{total_inner_fits} inner_fold={inner_fold} "
                    f"penalizer={penalizer:g} l1_ratio={l1_ratio:g}"
                )
                emit(f"COX START {fit_label}")
                inner = attach_nested_split(
                    table, assignments, NestedFoldSpec(repeat, outer_fold, inner_fold)
                )
                train = inner[inner["subject_split"] == "train"].copy()
                validation = inner[inner["subject_split"] == "val"].copy()
                record = {
                    "penalizer": penalizer,
                    "l1_ratio": l1_ratio,
                    "inner_fold": inner_fold,
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "train_events": int(train["event"].sum()),
                    "validation_events": int(validation["event"].sum()),
                    "uno_c_36m": None,
                    "status": "failed",
                    "error": None,
                    "warnings": [],
                }
                caught: list[Any] = []
                try:
                    if record["train_events"] == 0 or record["validation_events"] == 0:
                        raise ValueError("Inner train/validation fold has no events.")
                    x_train, x_validation, _metadata = build_fold_matrices(
                        train_rows=train,
                        evaluation_rows=validation,
                        full_pooled_matrix=full_pooled_matrix,
                        task="first_mci_survival",
                        feature_group=feature_group,
                        target="event",
                    )
                    train_frame = _cox_frame(x_train, train)
                    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(
                            train_frame,
                            duration_col="duration_months",
                            event_col="event",
                            show_progress=show_fit_progress,
                            fit_options={"max_steps": 500},
                        )
                        risk = model.predict_log_partial_hazard(
                            pd.DataFrame(x_validation, columns=train_frame.columns[:-2])
                        ).to_numpy()
                    record["warnings"] = [
                        f"{item.category.__name__}: {item.message}" for item in caught
                    ]
                    convergence_warnings = _convergence_warning_messages(caught)
                    if convergence_warnings:
                        raise RuntimeError(
                            "Cox fit did not converge: " + " | ".join(convergence_warnings)
                        )
                    score = uno_c_index(
                        train["duration_months"].to_numpy(),
                        train["event"].to_numpy(),
                        validation["duration_months"].to_numpy(),
                        validation["event"].to_numpy(),
                        risk,
                        tau=tau,
                    )
                    if score is None:
                        raise ValueError("Uno C is undefined for this inner fold.")
                    record.update({"uno_c_36m": float(score), "status": "ok"})
                except Exception as exc:
                    record["error"] = str(exc)
                    record["warnings"] = [
                        f"{item.category.__name__}: {item.message}" for item in caught
                    ]
                histories.append(record)
                elapsed = monotonic() - fit_started
                if record["status"] == "ok":
                    emit(
                        f"COX DONE {fit_label} status=ok "
                        f"uno_c_36m={record['uno_c_36m']:.6f} elapsed_seconds={elapsed:.1f}"
                    )
                else:
                    emit(
                        f"COX DONE {fit_label} status=failed elapsed_seconds={elapsed:.1f} "
                        f"error={record['error']}"
                    )

    history = pd.DataFrame(histories)
    successful = history[history["status"] == "ok"]
    grouped = successful.groupby(["penalizer", "l1_ratio"], as_index=False).agg(
        mean_inner_uno_c_36m=("uno_c_36m", "mean"), successful_folds=("inner_fold", "nunique")
    )
    grouped = grouped[grouped["successful_folds"] == n_inner]
    if grouped.empty:
        emit(
            f"COX GRID FAILED successful_fits={len(successful)}/{len(history)} "
            "complete_hyperparameter_settings=0"
        )
        raise RuntimeError("No Cox hyperparameter setting converged in every inner fold.")
    ranked = grouped.sort_values(
        ["mean_inner_uno_c_36m", "penalizer", "l1_ratio"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    inner_cv_selected = ranked.iloc[0]
    emit(
        f"COX GRID DONE successful_fits={len(successful)}/{len(history)} "
        f"selected_penalizer={float(inner_cv_selected['penalizer']):g} "
        f"selected_l1_ratio={float(inner_cv_selected['l1_ratio']):g} "
        f"mean_inner_uno_c_36m={float(inner_cv_selected['mean_inner_uno_c_36m']):.6f}"
    )

    x_development, x_test, feature_metadata = build_fold_matrices(
        train_rows=development,
        evaluation_rows=test,
        full_pooled_matrix=full_pooled_matrix,
        task="first_mci_survival",
        feature_group=feature_group,
        target="event",
    )
    development_frame = _cox_frame(x_development, development)
    feature_names = list(development_frame.columns[:-2])
    model, selected, final_caught, final_refit_attempts = _fit_final_cox(
        fitter_class=CoxPHFitter,
        development_frame=development_frame,
        candidates=ranked,
        emit=emit,
        show_fit_progress=show_fit_progress,
    )
    test_features = pd.DataFrame(x_test, columns=feature_names)
    risk = model.predict_log_partial_hazard(test_features).to_numpy()
    survival = model.predict_survival_function(test_features, times=horizons).to_numpy()
    integration_times = np.linspace(12.0, 36.0, 25)
    integration_survival = model.predict_survival_function(
        test_features, times=integration_times
    ).to_numpy()
    outer_uno = uno_c_index(
        development["duration_months"].to_numpy(),
        development["event"].to_numpy(),
        test["duration_months"].to_numpy(),
        test["event"].to_numpy(),
        risk,
        tau=tau,
    )
    brier = {
        f"{horizon:g}": ipcw_brier_score(
            development["duration_months"].to_numpy(),
            development["event"].to_numpy(),
            test["duration_months"].to_numpy(),
            test["event"].to_numpy(),
            survival[index],
            horizon=horizon,
        )
        for index, horizon in enumerate(horizons)
    }
    dynamic_auc = {
        f"{horizon:g}": cumulative_dynamic_auc(
            development["duration_months"].to_numpy(),
            development["event"].to_numpy(),
            test["duration_months"].to_numpy(),
            test["event"].to_numpy(),
            1.0 - survival[index],
            horizon=horizon,
        )
        for index, horizon in enumerate(horizons)
    }
    integration_brier = np.asarray(
        [
            ipcw_brier_score(
                development["duration_months"].to_numpy(),
                development["event"].to_numpy(),
                test["duration_months"].to_numpy(),
                test["event"].to_numpy(),
                integration_survival[index],
                horizon=float(horizon),
            )
            for index, horizon in enumerate(integration_times)
        ],
        dtype=float,
    )
    integrated_brier = (
        float(
            trapezoid(integration_brier, integration_times)
            / (integration_times[-1] - integration_times[0])
        )
        if np.isfinite(integration_brier).all()
        else None
    )

    emit("COX DIAGNOSTICS START proportional_hazards_and_scaled_schoenfeld")
    diagnostics_started = monotonic()
    scaled_schoenfeld_raw = model.compute_residuals(
        development_frame, kind="scaled_schoenfeld"
    )
    ph_result = proportional_hazard_test(
        model,
        development_frame,
        time_transform="rank",
        precomputed_residuals=scaled_schoenfeld_raw,
    )
    ph_table = ph_result.summary.reset_index().rename(columns={"index": "feature"})
    try:
        from scipy.stats import combine_pvalues

        global_statistic, global_p_value = combine_pvalues(
            ph_table["p"].to_numpy(dtype=float), method="fisher"
        )
    except Exception:
        global_statistic, global_p_value = None, None
    scaled_schoenfeld = scaled_schoenfeld_raw.reset_index(names="development_row_index")
    emit(f"COX DIAGNOSTICS DONE elapsed_seconds={monotonic() - diagnostics_started:.1f}")
    calibration = pd.DataFrame(_calibration_rows(test, survival, horizons))
    predictions = test[
        [
            column
            for column in (
                "participant_id",
                "session_id",
                "image_id",
                "duration_months",
                "event",
                "survival_status",
            )
            if column in test
        ]
    ].copy()
    predictions["repeat"] = repeat
    predictions["outer_fold"] = outer_fold
    predictions["risk_score"] = risk
    for index, horizon in enumerate(horizons):
        predictions[f"survival_probability_{horizon:g}m"] = survival[index]
    if 36.0 not in [float(value) for value in horizons]:
        raise ValueError("Survival reporting requires the locked 36-month horizon.")
    horizon_36_index = [float(value) for value in horizons].index(36.0)
    predictions["truth"] = predictions["event"]
    predictions["prediction"] = 1.0 - survival[horizon_36_index]
    predictions["ipcw_censoring_survival_before_duration"] = censoring_survival_at(
        development["duration_months"].to_numpy(),
        development["event"].to_numpy(),
        test["duration_months"].to_numpy(),
        left_limit=True,
    )

    summary = {
        "repeat": repeat,
        "outer_fold": outer_fold,
        "task": "first_mci_survival",
        "feature_group": feature_group,
        "selected_penalizer": float(selected["penalizer"]),
        "selected_l1_ratio": float(selected["l1_ratio"]),
        "mean_inner_uno_c_36m": float(selected["mean_inner_uno_c_36m"]),
        "inner_cv_selected_penalizer": float(inner_cv_selected["penalizer"]),
        "inner_cv_selected_l1_ratio": float(inner_cv_selected["l1_ratio"]),
        "inner_cv_selected_mean_uno_c_36m": float(
            inner_cv_selected["mean_inner_uno_c_36m"]
        ),
        "final_refit_used_ranked_fallback": bool(
            float(selected["penalizer"]) != float(inner_cv_selected["penalizer"])
            or float(selected["l1_ratio"]) != float(inner_cv_selected["l1_ratio"])
        ),
        "final_refit_attempts": final_refit_attempts,
        "outer_test_uno_c_36m": outer_uno,
        "outer_test_ipcw_brier": brier,
        "outer_test_cumulative_dynamic_auc": dynamic_auc,
        "outer_test_integrated_brier_12_to_36m": integrated_brier,
        "development_participants": development["participant_id"].nunique(),
        "development_events": int(development["event"].sum()),
        "test_participants": test["participant_id"].nunique(),
        "test_events": int(test["event"].sum()),
        "final_refit_on_complete_outer_development": True,
        "feature_metadata": feature_metadata,
        "failed_inner_fits": int((history["status"] == "failed").sum()),
        "inner_fit_warning_count": int(history["warnings"].map(len).sum()),
        "final_refit_warnings": [
            f"{item.category.__name__}: {item.message}" for item in final_caught
        ],
        "ph_feature_tests": ph_table.to_dict(orient="records"),
        "ph_global_test": {
            "method": "fisher_combination_of_featurewise_rank_time_tests",
            "statistic": None if global_statistic is None else float(global_statistic),
            "p_value": None if global_p_value is None else float(global_p_value),
            "interpretation": "omnibus diagnostic; featurewise scaled-Schoenfeld tests remain primary",
        },
    }
    return summary, predictions, history, calibration, scaled_schoenfeld
