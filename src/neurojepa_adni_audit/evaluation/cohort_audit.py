from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd

def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return [float(center - half_width), float(center + half_width)]


def _partition_summary(
    rows: pd.DataFrame,
    *,
    outcome_column: str | None,
    missing_columns: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(rows)),
        "participants": int(rows["participant_id"].nunique()),
        "missing": {
            column: {
                "count": int(rows[column].isna().sum()),
                "percent": float(100 * rows[column].isna().mean()) if len(rows) else None,
            }
            for column in missing_columns
            if column in rows
        },
    }
    if outcome_column is not None:
        outcome = pd.to_numeric(rows[outcome_column], errors="coerce")
        successes = int((outcome == 1).sum())
        total = int(outcome.notna().sum())
        summary["outcome"] = {
            "column": outcome_column,
            "available": total,
            "positive_or_event": successes,
            "negative_or_censored": total - successes,
            "proportion": float(successes / total) if total else None,
            "wilson_95": wilson_interval(successes, total),
        }
    if "duration_months" in rows:
        duration = pd.to_numeric(rows["duration_months"], errors="coerce")
        summary["followup_months"] = {
            "median": float(duration.median()) if duration.notna().any() else None,
            "minimum": float(duration.min()) if duration.notna().any() else None,
            "maximum": float(duration.max()) if duration.notna().any() else None,
        }
    if "horizon_error_months" in rows:
        error = pd.to_numeric(rows["horizon_error_months"], errors="coerce")
        summary["horizon_error_months"] = {
            "mean": float(error.mean()) if error.notna().any() else None,
            "mean_absolute": float(error.abs().mean()) if error.notna().any() else None,
            "maximum_absolute": float(error.abs().max()) if error.notna().any() else None,
        }
    return summary


def summarize_nested_cohort(
    *,
    table: pd.DataFrame,
    assignments: pd.DataFrame,
    task_name: str,
    outcome_column: str | None,
    missing_columns: list[str],
) -> dict[str, Any]:
    repeats = list(dict.fromkeys(assignments["repeat"].astype(str)))
    outer_folds = sorted(assignments["selected_outer_fold"].astype(int).unique())
    folds = []
    work = table.copy()
    work["participant_id"] = work["participant_id"].astype(str)
    for repeat in repeats:
        repeat_assignments = assignments[assignments["repeat"].astype(str) == repeat][
            ["participant_id", "outer_fold"]
        ].copy()
        repeat_assignments["participant_id"] = repeat_assignments["participant_id"].astype(str)
        if (repeat_assignments.groupby("participant_id")["outer_fold"].nunique() != 1).any():
            raise ValueError(f"Inconsistent outer-fold assignments in repeat {repeat}.")
        repeat_assignments = repeat_assignments.drop_duplicates("participant_id")
        fold_lookup = repeat_assignments.set_index("participant_id")["outer_fold"]
        assigned_outer_fold = work["participant_id"].map(fold_lookup)
        if assigned_outer_fold.isna().any():
            raise ValueError(
                f"{int(assigned_outer_fold.isna().sum())} task rows have no assignment in {repeat}."
            )
        for outer_fold in outer_folds:
            test_mask = assigned_outer_fold.to_numpy(dtype=int) == outer_fold
            development = work.loc[~test_mask]
            test = work.loc[test_mask]
            development_ids = set(development["participant_id"])
            test_ids = set(test["participant_id"])
            if development_ids & test_ids:
                raise ValueError(f"Participant overlap in cohort audit {repeat}/{outer_fold}.")
            folds.append(
                {
                    "repeat": repeat,
                    "outer_fold": int(outer_fold),
                    "development": _partition_summary(
                        development,
                        outcome_column=outcome_column,
                        missing_columns=missing_columns,
                    ),
                    "test": _partition_summary(
                        test,
                        outcome_column=outcome_column,
                        missing_columns=missing_columns,
                    ),
                }
            )
    return {
        "task": task_name,
        "overall": _partition_summary(
            table, outcome_column=outcome_column, missing_columns=missing_columns
        ),
        "outer_folds": folds,
        "comparative_performance_computed": False,
    }
