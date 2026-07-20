from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from neurojepa_adni_audit.evaluation.paired import benjamini_hochberg
from neurojepa_adni_audit.evaluation.reporting import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METRICS,
    SUPPORTED_METRICS,
)
from neurojepa_adni_audit.protocol import Protocol


def _weighted_uno_c_36m(frame: pd.DataFrame, prediction: str, weights: np.ndarray) -> float:
    durations = frame["duration_months"].to_numpy(dtype=float)
    events = frame["target"].to_numpy(dtype=int)
    risk = frame[prediction].to_numpy(dtype=float)
    censoring = frame["ipcw_censoring_survival_before_duration"].to_numpy(dtype=float)
    numerator = 0.0
    denominator = 0.0
    for index in np.flatnonzero((events == 1) & (durations <= 36.0) & (censoring > 0) & (weights > 0)):
        comparable = (durations > durations[index]) & (weights > 0)
        if not comparable.any():
            continue
        pair_weights = (weights[index] * weights[comparable]) / (censoring[index] ** 2)
        concordance = (risk[index] > risk[comparable]).astype(float)
        concordance += 0.5 * (risk[index] == risk[comparable])
        numerator += float(np.sum(pair_weights * concordance))
        denominator += float(np.sum(pair_weights))
    return float("nan") if denominator == 0 else numerator / denominator


NESTED_METRICS = {**METRICS, "uno_c_36m": _weighted_uno_c_36m}
NESTED_HIGHER_IS_BETTER = HIGHER_IS_BETTER | {"uno_c_36m"}
NESTED_SUPPORTED_METRICS = SUPPORTED_METRICS | {"uno_c_36m"}


@dataclass(frozen=True)
class RepeatedPairedBootstrapResult:
    estimate: float
    ci95_low: float
    ci95_high: float
    bootstrap_sign_p_value: float
    n_subjects: int
    n_rows: int
    repeats: int
    replicates: int
    seed: int


def _prediction_files(root: Path, task: str, target: str, feature: str) -> list[Path]:
    directory = root / task / target / feature
    files = sorted(directory.glob("*/outer_*/outer_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No nested predictions found under {directory}.")
    return files


def load_feature_oof_predictions(
    *, root: Path, task: str, target: str, feature: str
) -> pd.DataFrame:
    frames = []
    for path in _prediction_files(root, task, target, feature):
        frame = pd.read_csv(path, low_memory=False)
        required = {"participant_id", "repeat", "outer_fold", "truth", "prediction"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing nested prediction columns: {', '.join(missing)}")
        if frame["repeat"].nunique() != 1 or frame["outer_fold"].nunique() != 1:
            raise ValueError(f"{path} mixes repeat or outer-fold values.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _pair_keys(combined: pd.DataFrame, conventional: pd.DataFrame) -> list[str]:
    keys = ["repeat", "outer_fold", "participant_id"]
    for candidate in ("pair_id", "image_id", "current_image_id", "session_id", "current_session_id"):
        if candidate in combined and candidate in conventional:
            keys.append(candidate)
            break
    return keys


def pair_nested_predictions(combined: pd.DataFrame, conventional: pd.DataFrame) -> pd.DataFrame:
    keys = _pair_keys(combined, conventional)
    for label, frame in (("combined", combined), ("conventional", conventional)):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} nested predictions contain duplicate paired keys {keys}.")
    auxiliary = [
        column
        for column in ("duration_months", "ipcw_censoring_survival_before_duration")
        if column in combined and column in conventional
    ]
    left = combined[keys + ["truth", "prediction", *auxiliary]].rename(
        columns={"truth": "target_combined", "prediction": "combined"}
    )
    right = conventional[keys + ["truth", "prediction", *auxiliary]].rename(
        columns={
            "truth": "target_conventional",
            "prediction": "conventional",
            **{column: f"{column}_conventional" for column in auxiliary},
        }
    )
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(
            "Nested paired cohorts differ: "
            f"combined={len(left)}, conventional={len(right)}, paired={len(paired)}."
        )
    if not np.allclose(
        paired["target_combined"], paired["target_conventional"], equal_nan=True
    ):
        raise ValueError("Nested paired targets differ between feature groups.")
    for column in auxiliary:
        if not np.allclose(
            paired[column], paired[f"{column}_conventional"], equal_nan=True
        ):
            raise ValueError(f"Nested paired `{column}` values differ between feature groups.")
    paired["target"] = paired["target_combined"]
    return paired[[*keys, "target", "combined", "conventional", *auxiliary]]


def _metric_difference(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame, str, np.ndarray], float],
    weights: np.ndarray,
) -> float:
    return metric(frame, "combined", weights) - metric(frame, "conventional", weights)


def repeated_paired_subject_bootstrap(
    frame: pd.DataFrame,
    *,
    metric: Callable[[pd.DataFrame, str, np.ndarray], float],
    replicates: int,
    seed: int,
    stratify_binary: bool,
) -> tuple[RepeatedPairedBootstrapResult, dict[str, float]]:
    if replicates < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates for paper reporting.")
    repeats = list(dict.fromkeys(frame["repeat"].astype(str)))
    subjects = frame["participant_id"].astype(str).drop_duplicates().to_numpy()
    subject_position = {subject: index for index, subject in enumerate(subjects)}
    subject_codes = frame["participant_id"].astype(str).map(subject_position).to_numpy(dtype=int)
    repeat_masks = {repeat: frame["repeat"].astype(str).to_numpy() == repeat for repeat in repeats}

    def effects(multiplicity: np.ndarray) -> dict[str, float]:
        output = {}
        for repeat, mask in repeat_masks.items():
            repeat_codes = subject_codes[mask]
            counts = np.bincount(repeat_codes, minlength=len(subjects)).astype(float)
            row_weights = multiplicity[repeat_codes] / counts[repeat_codes]
            output[repeat] = _metric_difference(frame.loc[mask], metric, row_weights)
        return output

    point_effects = effects(np.ones(len(subjects), dtype=float))
    estimate = float(np.mean(list(point_effects.values())))
    strata: list[np.ndarray]
    if stratify_binary:
        target_counts = frame.groupby(frame["participant_id"].astype(str))["target"].nunique()
        if (target_counts != 1).any():
            raise ValueError("Binary bootstrap participants must have one stable target label.")
        subject_targets = frame.groupby(frame["participant_id"].astype(str))["target"].first()
        if not subject_targets.isin([0, 1]).all():
            raise ValueError("Binary stratified bootstrap requires participant-level 0/1 targets.")
        strata = [
            np.asarray([subject_position[subject] for subject in subject_targets.index[subject_targets == value]])
            for value in (0, 1)
        ]
    else:
        strata = [np.arange(len(subjects))]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=float)
    for bootstrap_index in range(replicates):
        multiplicity = np.zeros(len(subjects), dtype=float)
        for stratum in strata:
            sampled = rng.choice(stratum, size=len(stratum), replace=True)
            multiplicity += np.bincount(sampled, minlength=len(subjects))
        values[bootstrap_index] = float(np.mean(list(effects(multiplicity).values())))
    finite = values[np.isfinite(values)]
    if len(finite) < max(1000, int(0.95 * replicates)):
        raise ValueError(f"Too few valid repeated bootstrap draws: {len(finite)}/{replicates}.")
    low, high = np.quantile(finite, [0.025, 0.975])
    non_positive = (np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1)
    non_negative = (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)
    result = RepeatedPairedBootstrapResult(
        estimate=estimate,
        ci95_low=float(low),
        ci95_high=float(high),
        bootstrap_sign_p_value=float(min(1.0, 2 * min(non_positive, non_negative))),
        n_subjects=len(subjects),
        n_rows=len(frame),
        repeats=len(repeats),
        replicates=replicates,
        seed=seed,
    )
    return result, point_effects


def run_nested_paired_report(
    *, spec_path: Path, protocol: Protocol, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("Nested report specification requires comparisons.")
    effect_rows = []
    dispersion_rows = []
    for comparison in comparisons:
        metric_name = str(comparison["metric"])
        if metric_name not in NESTED_SUPPORTED_METRICS:
            raise ValueError(f"Unsupported nested paired metric `{metric_name}`.")
        root = Path(comparison["predictions_root"])
        combined = load_feature_oof_predictions(
            root=root,
            task=str(comparison["task"]),
            target=str(comparison["target"]),
            feature=str(comparison.get("combined_feature", "neurojepa_conventional")),
        )
        conventional = load_feature_oof_predictions(
            root=root,
            task=str(comparison["task"]),
            target=str(comparison["target"]),
            feature=str(comparison.get("conventional_feature", "conventional_full")),
        )
        paired = pair_nested_predictions(combined, conventional)
        metric = NESTED_METRICS[metric_name]
        for (repeat, outer_fold), fold in paired.groupby(["repeat", "outer_fold"], sort=True):
            rows_per_subject = fold.groupby("participant_id")["participant_id"].transform("size")
            weights = 1.0 / rows_per_subject.to_numpy(dtype=float)
            dispersion_rows.append(
                {
                    "id": comparison["id"],
                    "repeat": repeat,
                    "outer_fold": int(outer_fold),
                    "effect": _metric_difference(fold, metric, weights),
                }
            )
        result, repeat_effects = repeated_paired_subject_bootstrap(
            paired,
            metric=metric,
            replicates=int(protocol.payload["bootstrap"]["replicates"]),
            seed=int(protocol.payload["bootstrap"]["seed"]),
            stratify_binary=metric_name in NESTED_HIGHER_IS_BETTER,
        )
        values = np.asarray(list(repeat_effects.values()), dtype=float)
        effect_rows.append(
            {
                "id": comparison["id"],
                "family": comparison.get("family", "global_core_endpoints"),
                "metric": metric_name,
                "difference_definition": "combined-minus-conventional",
                "favorable_direction": "negative" if metric_name in LOWER_IS_BETTER else "positive",
                **asdict(result),
                "repeat_effects": json.dumps(repeat_effects, sort_keys=True),
                "repeat_effect_mean": float(values.mean()),
                "repeat_effect_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "repeat_effect_min": float(values.min()),
                "repeat_effect_max": float(values.max()),
            }
        )
    effects = pd.DataFrame(effect_rows)
    effects["fdr_adjusted_p_value"] = np.nan
    for _family, indices in effects.groupby("family", sort=False).groups.items():
        positions = list(indices)
        effects.loc[positions, "fdr_adjusted_p_value"] = benjamini_hochberg(
            effects.loc[positions, "bootstrap_sign_p_value"].tolist()
        )
    dispersion = pd.DataFrame(dispersion_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_dir / "nested_paired_effects.csv", index=False)
    dispersion.to_csv(output_dir / "outer_fold_effects.csv", index=False)
    (output_dir / "nested_paired_effects.json").write_text(
        json.dumps(
            {
                "protocol_sha256": protocol.sha256,
                "effects": effects.to_dict(orient="records"),
                "fold_dispersion": dispersion.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return effects, dispersion
