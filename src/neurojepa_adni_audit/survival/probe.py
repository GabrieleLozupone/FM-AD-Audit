from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from neurojepa_adni_audit.probes.downstream import (
    DEFAULT_COVARIATE_COLUMNS,
    DEFAULT_EMBEDDING_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    load_pooled_matrix,
    normalize_diagnosis,
    require_sklearn,
    split_rows,
    validate_feature_paths,
)
from neurojepa_adni_audit.features.covariates import validate_covariate_columns
from neurojepa_adni_audit.features.progression_aligned import STAGE1_FEATURE_MODES
from neurojepa_adni_audit.features.tabular import build_pooled_or_covariate_feature_matrix
from neurojepa_adni_audit.features.groups import (
    add_morphometry_ratios,
    feature_columns,
    is_feature_group,
    uses_neurojepa,
    validate_no_target_leakage,
)
from neurojepa_adni_audit.paper_run import bind_paper_run, require_protocol_grid


DEFAULT_SURVIVAL_OUTPUT_DIR = DEFAULT_OUTPUT_DIR.parent / "neurojepa_survival_probe"
DEFAULT_PENALIZERS = [0.01, 0.1, 1.0, 10.0]
DEFAULT_L1_RATIOS = [0.0, 0.5, 1.0]
DEFAULT_SURVIVAL_HORIZONS = [24.0, 36.0, 60.0]


@dataclass
class SurvivalMetric:
    split: str
    n: int
    n_events: int
    n_censored: int
    c_index: float | None
    mean_duration_months: float | None
    median_duration_months: float | None


def require_lifelines() -> tuple[Any, Any]:
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `lifelines`. Run `pip install -r requirements.txt`.") from exc
    return CoxPHFitter, concordance_index


def require_pycox() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torchtuples as tt
        from pycox.evaluation import EvalSurv
        from pycox.models import DeepHitSingle, LogisticHazard
        from torchtuples.practical import MLPVanilla
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `pycox`. Run `pip install -r requirements.txt`.") from exc
    return tt, EvalSurv, LogisticHazard, DeepHitSingle, MLPVanilla


def ensure_scipy_simps_compat() -> None:
    try:
        import scipy.integrate as integrate
    except ModuleNotFoundError:
        return
    if hasattr(integrate, "simps") or not hasattr(integrate, "simpson"):
        return

    def simps(y: Any, x: Any = None, dx: float = 1.0, axis: int = -1, even: str | None = None) -> Any:
        _ = even
        return integrate.simpson(y, x=x, dx=dx, axis=axis)

    integrate.simps = simps


def build_mci_time_to_ad_table(
    manifest: pd.DataFrame,
    mci_anchor_policy: str,
) -> pd.DataFrame:
    sortable = manifest.copy()
    sortable["session_month_numeric"] = pd.to_numeric(sortable["session_month"], errors="coerce")
    sortable["session_month_sort"] = sortable["session_month_numeric"].fillna(np.inf)
    sortable["diagnosis_norm"] = sortable["diagnosis_at_session"].map(normalize_diagnosis)
    sortable = sortable.sort_values(["participant_id", "session_month_sort", "session_id"]).reset_index(drop=True)

    anchor_rows = sortable[
        (sortable["diagnosis_norm"] == "MCI") & sortable["session_month_numeric"].notna()
    ].copy()
    if mci_anchor_policy == "first":
        anchor_rows = anchor_rows.groupby("participant_id", as_index=False).head(1).copy()
    elif mci_anchor_policy != "all":  # pragma: no cover
        raise ValueError(f"Unsupported MCI anchor policy: {mci_anchor_policy}")

    targets: list[dict[str, Any]] = []
    for row in anchor_rows.itertuples(index=False):
        participant_id = getattr(row, "participant_id")
        session_id = getattr(row, "session_id")
        image_id = getattr(row, "image_id")
        anchor_month = float(getattr(row, "session_month_numeric"))

        subject_rows = sortable[sortable["participant_id"] == participant_id].copy()
        subject_rows = subject_rows[subject_rows["session_month_numeric"].notna()].copy()
        followup_rows = subject_rows[subject_rows["session_month_numeric"] >= anchor_month].copy()
        if followup_rows.empty:
            continue

        max_followup_months = float(followup_rows["session_month_numeric"].max() - anchor_month)
        future_ad_rows = followup_rows[
            (followup_rows["diagnosis_norm"] == "AD")
            & (followup_rows["session_month_numeric"] > anchor_month)
        ].copy()
        months_to_first_ad = None
        event = 0
        duration_months = max_followup_months
        if not future_ad_rows.empty:
            months_to_first_ad = float(future_ad_rows["session_month_numeric"].min() - anchor_month)
            event = 1
            duration_months = months_to_first_ad
        if not np.isfinite(duration_months) or duration_months <= 0:
            continue

        targets.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "image_id": image_id,
                "mci_anchor_id": f"{participant_id}_{session_id}",
                "anchor_month": anchor_month,
                "baseline_month": anchor_month,
                "duration_months": float(duration_months),
                "event": int(event),
                "months_to_first_ad": months_to_first_ad,
                "followup_months_observed": max_followup_months,
                "survival_status": "converted_to_ad" if event else "censored_no_ad_observed",
            }
        )

    if not targets:
        return anchor_rows.iloc[0:0].copy()
    target_table = pd.DataFrame(targets)
    return anchor_rows.merge(target_table, on=["participant_id", "session_id", "image_id"], how="inner")


def load_survival_task_table(
    manifest_path: Path,
    embedding_index_path: Path,
    mci_anchor_policy: str,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not embedding_index_path.exists():
        raise FileNotFoundError(
            f"Embedding index not found: {embedding_index_path}. Finish Neuro-JEPA extraction first."
        )
    manifest = pd.read_csv(manifest_path, low_memory=False)
    index = pd.read_csv(embedding_index_path, low_memory=False)
    if "image_id" not in manifest.columns or "image_id" not in index.columns:
        raise ValueError("Both manifest and embedding index must contain `image_id`.")

    survival_table = build_mci_time_to_ad_table(manifest, mci_anchor_policy=mci_anchor_policy)
    table = survival_table.merge(
        index.drop_duplicates("image_id"),
        on=["participant_id", "session_id", "image_id"],
        how="inner",
        suffixes=("", "_embedding"),
    )
    if "subject_split" not in table.columns:
        raise ValueError("Merged survival task table does not contain `subject_split`.")
    table = table[table["subject_split"].isin(["train", "val", "test"])].copy()
    return table.reset_index(drop=True)


def build_feature_matrix(
    table: pd.DataFrame,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    return build_pooled_or_covariate_feature_matrix(
        table,
        feature,
        include_covariates,
        covariate_columns,
        load_pooled_matrix,
        standardize_pooled=True,
        covariate_only_error="No usable covariate columns were found for survival `--feature covariates`.",
    )


def make_cox_frame(x: np.ndarray, durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(x, columns=[f"x{i:04d}" for i in range(x.shape[1])])
    frame["duration_months"] = durations.astype(float)
    frame["event"] = events.astype(int)
    return frame


def c_index_for_scores(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> float | None:
    (_CoxPHFitter, concordance_index) = require_lifelines()
    if len(durations) == 0 or int(np.sum(events)) == 0:
        return None
    return float(concordance_index(durations, -risk_scores, events))


def compute_survival_metrics(table: pd.DataFrame, risk_scores: np.ndarray) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    durations = table["duration_months"].to_numpy(dtype=float)
    events = table["event"].to_numpy(dtype=int)
    for split in ("train", "val", "test"):
        mask = table["subject_split"].to_numpy() == split
        split_durations = durations[mask]
        split_events = events[mask]
        metric = SurvivalMetric(
            split=split,
            n=int(np.sum(mask)),
            n_events=int(np.sum(split_events)),
            n_censored=int(np.sum(split_events == 0)),
            c_index=c_index_for_scores(split_durations, split_events, risk_scores[mask]),
            mean_duration_months=float(np.mean(split_durations)) if len(split_durations) else None,
            median_duration_months=float(np.median(split_durations)) if len(split_durations) else None,
        )
        metrics.append(asdict(metric))
    return metrics


def nearest_survival_values(surv: pd.DataFrame, horizons: list[float]) -> dict[float, np.ndarray]:
    values: dict[float, np.ndarray] = {}
    duration_index = surv.index.to_numpy(dtype=float)
    for horizon in horizons:
        if duration_index.size == 0:
            values[horizon] = np.full(surv.shape[1], np.nan, dtype=float)
            continue
        idx = int(np.searchsorted(duration_index, horizon, side="right") - 1)
        idx = max(0, min(idx, len(duration_index) - 1))
        values[horizon] = surv.iloc[idx].to_numpy(dtype=float)
    return values


def valid_metric_times(train_durations: np.ndarray, test_durations: np.ndarray, horizons: list[float]) -> np.ndarray:
    if len(train_durations) == 0 or len(test_durations) == 0:
        return np.array([], dtype=float)
    lower = max(float(np.min(train_durations)), float(np.min(test_durations)))
    upper = min(float(np.max(train_durations)), float(np.max(test_durations)))
    return np.array(sorted({float(h) for h in horizons if lower < float(h) < upper}), dtype=float)


def compute_neural_survival_metrics(
    table: pd.DataFrame,
    risk_scores: np.ndarray,
    surv: pd.DataFrame,
    horizons: list[float],
) -> list[dict[str, Any]]:
    ensure_scipy_simps_compat()
    (_tt, EvalSurv, _LogisticHazard, _DeepHitSingle, _MLPVanilla) = require_pycox()
    metrics = compute_survival_metrics(table, risk_scores)
    durations = table["duration_months"].to_numpy(dtype=float)
    events = table["event"].to_numpy(dtype=int).astype(bool)
    split_values = table["subject_split"].to_numpy()
    train_mask = split_values == "train"

    for metric in metrics:
        split = metric["split"]
        mask = split_values == split
        split_surv = surv.iloc[:, mask]
        try:
            ev = EvalSurv(split_surv, durations[mask], events[mask], censor_surv="km")
            metric["concordance_td"] = float(ev.concordance_td("antolini"))
        except Exception as exc:
            metric["concordance_td"] = None
            metric["concordance_td_error"] = str(exc)

        times = valid_metric_times(durations[train_mask], durations[mask], horizons)
        if len(times):
            try:
                ev = EvalSurv(split_surv, durations[mask], events[mask], censor_surv="km")
                metric["integrated_brier_score"] = float(ev.integrated_brier_score(times))
                metric["brier_score_times_months"] = [float(value) for value in times]
            except Exception as exc:
                metric["integrated_brier_score"] = None
                metric["integrated_brier_score_error"] = str(exc)
        else:
            metric["integrated_brier_score"] = None
            metric["integrated_brier_score_error"] = "No configured horizon is inside train/test observed-duration support."
    return metrics


def fit_cox_elastic_net(
    table: pd.DataFrame,
    x_all: np.ndarray,
    penalizers: list[float],
    l1_ratios: list[float],
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    CoxPHFitter, _concordance_index = require_lifelines()
    train_mask = table["subject_split"].to_numpy() == "train"
    val_mask = table["subject_split"].to_numpy() == "val"
    durations = table["duration_months"].to_numpy(dtype=float)
    events = table["event"].to_numpy(dtype=int)

    train_frame = make_cox_frame(x_all[train_mask], durations[train_mask], events[train_mask])
    val_frame = make_cox_frame(x_all[val_mask], durations[val_mask], events[val_mask])
    rng = np.random.default_rng(seed)
    candidates = [(float(p), float(l1)) for p in penalizers for l1 in l1_ratios]
    rng.shuffle(candidates)

    best_model = None
    best_score = -np.inf
    best_params: dict[str, Any] | None = None
    tuning_rows: list[dict[str, Any]] = []
    progress = tqdm(candidates, desc="cox_elastic_net", unit="fit", dynamic_ncols=True)
    for penalizer, l1_ratio in progress:
        progress.set_postfix(penalizer=penalizer, l1_ratio=l1_ratio, best_val_c_index=best_score if np.isfinite(best_score) else None)
        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        try:
            model.fit(train_frame, duration_col="duration_months", event_col="event", show_progress=False)
            val_risk = model.predict_log_partial_hazard(val_frame).to_numpy(dtype=float).reshape(-1)
            val_c_index = c_index_for_scores(durations[val_mask], events[val_mask], val_risk)
        except Exception as exc:  # lifelines can reject ill-conditioned settings.
            tuning_rows.append(
                {
                    "penalizer": penalizer,
                    "l1_ratio": l1_ratio,
                    "val_c_index": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue
        tuning_rows.append(
            {
                "penalizer": penalizer,
                "l1_ratio": l1_ratio,
                "val_c_index": val_c_index,
                "status": "ok",
                "error": None,
            }
        )
        if val_c_index is not None and val_c_index > best_score:
            best_score = val_c_index
            best_model = model
            best_params = {"penalizer": penalizer, "l1_ratio": l1_ratio, "val_c_index": val_c_index}
        progress.set_postfix(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            val_c_index=val_c_index,
            best_val_c_index=best_score if np.isfinite(best_score) else None,
        )
    progress.close()

    if best_params is None or best_model is None:
        raise RuntimeError("All Cox elastic-net hyperparameter settings failed.")

    all_frame = make_cox_frame(x_all, durations, events)
    risk_scores = best_model.predict_log_partial_hazard(all_frame).to_numpy(dtype=float).reshape(-1)
    tuning = {
        "selected": best_params,
        "candidates": sorted(
            tuning_rows,
            key=lambda row: (-np.inf if row["val_c_index"] is None else row["val_c_index"]),
            reverse=True,
        ),
    }
    return best_model, risk_scores, tuning


def fit_pycox_discrete_model(
    table: pd.DataFrame,
    x_all: np.ndarray,
    model_name: str,
    num_durations: int,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    deephit_alpha: float,
    deephit_sigma: float,
    seed: int,
) -> tuple[Any, np.ndarray, pd.DataFrame, dict[str, Any]]:
    import torch

    tt, _EvalSurv, LogisticHazard, DeepHitSingle, MLPVanilla = require_pycox()
    if num_durations < 2:
        raise ValueError("`--num-durations` must be at least 2.")
    if not hidden_dims:
        raise ValueError("`--hidden-dims` must contain at least one layer size.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != "cpu":
        if not device.startswith("cuda"):
            raise ValueError("`--device` must be `cpu` or a CUDA device such as `cuda:0`.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with `--device`, but torch reports that CUDA is unavailable.")

    split_values = table["subject_split"].to_numpy()
    train_mask = split_values == "train"
    val_mask = split_values == "val"
    durations = table["duration_months"].to_numpy(dtype=np.float32)
    events = table["event"].to_numpy(dtype=np.int64)
    x_all = x_all.astype(np.float32)

    ModelClass = LogisticHazard if model_name == "logistic_hazard" else DeepHitSingle
    labtrans = ModelClass.label_transform(num_durations)
    y_train = labtrans.fit_transform(durations[train_mask], events[train_mask])
    y_val = labtrans.transform(durations[val_mask], events[val_mask])

    net = MLPVanilla(
        in_features=x_all.shape[1],
        num_nodes=hidden_dims,
        out_features=labtrans.out_features,
        batch_norm=True,
        dropout=dropout,
        output_bias=False,
    )
    optimizer = tt.optim.Adam(lr=lr)
    if model_name == "logistic_hazard":
        model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
    elif model_name == "deephit_single":
        model = DeepHitSingle(
            net,
            optimizer,
            alpha=deephit_alpha,
            sigma=deephit_sigma,
            duration_index=labtrans.cuts,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported pycox model: {model_name}")
    if device != "cpu":
        if hasattr(model, "set_device"):
            model.set_device(device)
        elif hasattr(model, "to_device"):
            model.to_device(device)
        else:
            model.net.to(device)

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(
        x_all[train_mask],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        val_data=(x_all[val_mask], y_val),
        verbose=False,
    )
    surv = model.predict_surv_df(x_all)
    risk_scores = (1.0 - surv.iloc[-1].to_numpy(dtype=float)).reshape(-1)
    log_frame = log.to_pandas()
    tuning = {
        "selected": {
            "model": model_name,
            "num_durations": int(num_durations),
            "duration_cuts_months": [float(value) for value in labtrans.cuts],
            "hidden_dims": [int(value) for value in hidden_dims],
            "dropout": float(dropout),
            "lr": float(lr),
            "epochs_requested": int(epochs),
            "epochs_trained": int(len(log_frame)),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "device": device,
            "deephit_alpha": float(deephit_alpha) if model_name == "deephit_single" else None,
            "deephit_sigma": float(deephit_sigma) if model_name == "deephit_single" else None,
        },
        "candidates": log_frame.reset_index().to_dict(orient="records"),
    }
    return model, risk_scores, surv, tuning


def build_prediction_table(
    table: pd.DataFrame,
    risk_scores: np.ndarray,
    surv: pd.DataFrame | None = None,
    horizons: list[float] | None = None,
) -> pd.DataFrame:
    columns = [
        "participant_id",
        "session_id",
        "image_id",
        "subject_split",
        "diagnosis_at_session",
        "diagnosis_norm",
        "mci_anchor_id",
        "anchor_month",
        "duration_months",
        "event",
        "survival_status",
        "months_to_first_ad",
        "followup_months_observed",
    ]
    columns = [column for column in columns if column in table.columns]
    predictions = table[columns].copy()
    predictions["risk_score"] = risk_scores
    if surv is not None and horizons is not None:
        for horizon, survival_values in nearest_survival_values(surv, horizons).items():
            suffix = f"{int(horizon)}m" if float(horizon).is_integer() else f"{horizon:g}m"
            predictions[f"survival_probability_{suffix}"] = survival_values
            predictions[f"risk_probability_{suffix}"] = 1.0 - survival_values
    return predictions


def summarize_task_table(table: pd.DataFrame) -> dict[str, Any]:
    by_split = {}
    for split, group in table.groupby("subject_split"):
        anchors_per_subject = group.groupby("participant_id").size()
        by_split[split] = {
            "n": int(len(group)),
            "n_subjects": int(group["participant_id"].nunique()),
            "n_images": int(group["image_id"].nunique()) if "image_id" in group.columns else None,
            "n_events": int(group["event"].sum()),
            "n_censored": int((group["event"] == 0).sum()),
            "mean_duration_months": float(group["duration_months"].mean()) if len(group) else None,
            "median_duration_months": float(group["duration_months"].median()) if len(group) else None,
            "mean_anchors_per_subject": float(anchors_per_subject.mean()) if not anchors_per_subject.empty else 0.0,
            "max_anchors_per_subject": int(anchors_per_subject.max()) if not anchors_per_subject.empty else 0,
        }
    return {
        "n_rows": int(len(table)),
        "n_subjects": int(table["participant_id"].nunique()),
        "n_images": int(table["image_id"].nunique()) if "image_id" in table.columns else None,
        "n_events": int(table["event"].sum()),
        "n_censored": int((table["event"] == 0).sum()),
        "n_subjects_with_multiple_anchors": int((table.groupby("participant_id").size() > 1).sum()),
        "by_split": by_split,
    }


def write_run_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    task_table: pd.DataFrame,
    tuning: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    task_table.to_csv(output_dir / "task_table.csv", index=False)
    tuning_frame = pd.DataFrame(tuning["candidates"])
    tuning_frame.to_csv(output_dir / "model_training_log.csv", index=False)
    if "penalizer" in tuning_frame.columns and "l1_ratio" in tuning_frame.columns:
        tuning_frame.to_csv(output_dir / "cox_hyperparameter_tuning.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def train_survival_probe(
    table: pd.DataFrame,
    model_name: str,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
    penalizers: list[float],
    l1_ratios: list[float],
    num_durations: int,
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    deephit_alpha: float,
    deephit_sigma: float,
    survival_horizons: list[float],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")
    for split, rows in splits.items():
        if int(rows["event"].sum()) == 0:
            raise ValueError(f"Split `{split}` has no observed AD conversion events; cannot score survival C-index.")

    x_all, cov_info = build_feature_matrix(
        table=table,
        feature=feature,
        include_covariates=include_covariates,
        covariate_columns=covariate_columns,
    )
    if model_name == "cox_elastic_net":
        _model, risk_scores, tuning = fit_cox_elastic_net(
            table=table,
            x_all=x_all,
            penalizers=penalizers,
            l1_ratios=l1_ratios,
            seed=seed,
        )
        surv = None
        metrics = compute_survival_metrics(table, risk_scores)
    elif model_name in {"logistic_hazard", "deephit_single"}:
        _model, risk_scores, surv, tuning = fit_pycox_discrete_model(
            table=table,
            x_all=x_all,
            model_name=model_name,
            num_durations=num_durations,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            deephit_alpha=deephit_alpha,
            deephit_sigma=deephit_sigma,
            seed=seed,
        )
        metrics = compute_neural_survival_metrics(table, risk_scores, surv, survival_horizons)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported model: {model_name}")

    predictions = build_prediction_table(table, risk_scores, surv=surv, horizons=survival_horizons)
    summary = {
        "model": model_name,
        "feature": feature,
        "include_covariates": include_covariates,
        "covariates": cov_info,
        "n_input_features": int(x_all.shape[1]),
        "hyperparameter_tuning": tuning,
        "survival_horizons_months": [float(value) for value in survival_horizons],
        "metrics": metrics,
    }
    return summary, predictions, tuning


def run(args: Any) -> None:
    paper_context = None
    if args.protocol_path is not None:
        if not args.replication:
            raise ValueError("`--replication` is required with `--protocol-path`.")
        paper_context = bind_paper_run(
            protocol_path=args.protocol_path,
            replication_name=args.replication,
            manifest_path=args.manifest_path,
            embedding_index_path=args.embedding_index_path,
            allow_draft_protocol=args.allow_draft_protocol,
        )
        require_protocol_grid(
            args.penalizer, paper_context.protocol.payload["cox_penalizers"], "Cox penalizer grid"
        )
        require_protocol_grid(args.l1_ratio, paper_context.protocol.payload["cox_l1_ratios"], "Cox L1 grid")
        args.seed = paper_context.replication.split_seed
    features = args.feature or ["pooled"]
    models = args.model or ["cox_elastic_net"]
    stage1_covariate_modes = {f"{mode}_covariates": mode for mode in STAGE1_FEATURE_MODES}
    if args.include_covariates or "covariates" in features or any(feature in stage1_covariate_modes for feature in features):
        validate_covariate_columns(args.covariate_columns)
    if any(value < 0 for value in args.penalizer):
        raise ValueError("All `--penalizer` values must be non-negative.")
    if any(value < 0 or value > 1 for value in args.l1_ratio):
        raise ValueError("All `--l1-ratio` values must be in [0, 1].")
    if args.num_durations < 2:
        raise ValueError("`--num-durations` must be at least 2.")
    if not args.hidden_dims or any(value <= 0 for value in args.hidden_dims):
        raise ValueError("`--hidden-dims` must contain positive integers.")
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError("`--dropout` must be in [0, 1).")
    if args.lr <= 0:
        raise ValueError("`--lr` must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("`--epochs`, `--batch-size`, and `--patience` must be positive.")
    if any(value <= 0 for value in args.survival_horizons):
        raise ValueError("All `--survival-horizons` values must be positive months.")

    base_table = load_survival_task_table(
        manifest_path=args.manifest_path,
        embedding_index_path=args.embedding_index_path,
        mci_anchor_policy=args.mci_anchor_policy,
    )
    if base_table.empty:
        raise ValueError("No MCI survival rows available after merging manifest and embeddings.")
    base_table = add_morphometry_ratios(base_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_run_summaries = []
    for model_name in models:
        for feature in features:
            canonical_group = is_feature_group(feature)
            if canonical_group:
                selected_covariates = feature_columns("first_mci_survival", feature)
                validate_no_target_leakage("first_mci_survival", "event", selected_covariates)
                model_feature = "pooled" if uses_neurojepa(feature) else "covariates"
                include_covariates = uses_neurojepa(feature) and bool(selected_covariates)
            else:
                selected_covariates = args.covariate_columns
                model_feature = stage1_covariate_modes.get(feature, feature)
                include_covariates = args.include_covariates or feature in stage1_covariate_modes
            table = validate_feature_paths(base_table, model_feature)
            run_name = f"mci_time_to_ad_{args.mci_anchor_policy}_{feature}_{model_name}"
            if model_name == "cox_elastic_net":
                run_name = f"mci_time_to_ad_{args.mci_anchor_policy}_{feature}"
            if include_covariates and not canonical_group and feature != "covariates" and feature not in stage1_covariate_modes:
                run_name += "_covariates"
            run_dir = args.output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            summary, predictions, tuning = train_survival_probe(
                table=table,
                model_name=model_name,
                feature=model_feature,
                include_covariates=include_covariates,
                covariate_columns=selected_covariates,
                penalizers=args.penalizer,
                l1_ratios=args.l1_ratio,
                num_durations=args.num_durations,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                device=args.device,
                deephit_alpha=args.deephit_alpha,
                deephit_sigma=args.deephit_sigma,
                survival_horizons=args.survival_horizons,
                seed=args.seed,
            )
            summary = {
                "task": "mci_time_to_ad",
                "manifest_path": str(args.manifest_path),
                "embedding_index_path": str(args.embedding_index_path),
                "output_dir": str(run_dir),
                "seed": args.seed,
                "paper_run": paper_context.metadata() if paper_context is not None else None,
                "label_definition": {
                    "input": "MCI anchor visit",
                    "mci_anchor_policy": args.mci_anchor_policy,
                    "event": "first future AD diagnosis after the MCI anchor",
                    "duration_for_event": "months from MCI anchor to first future AD diagnosis",
                    "duration_for_censoring": "months from MCI anchor to last observed follow-up without AD",
                    "late_converters": "Kept as observed events at their actual conversion time",
                },
                "task_table": summarize_task_table(table),
                **summary,
            }
            summary["feature"] = feature
            summary["feature_group"] = feature if canonical_group else None
            summary["source_covariate_columns"] = selected_covariates
            write_run_outputs(run_dir, summary, predictions, table, tuning)
            all_run_summaries.append(summary)
            compact_metrics = {item["split"]: item for item in summary["metrics"]}
            print(f"Wrote {model_name} {feature} survival run to {run_dir}")
            print(json.dumps(compact_metrics, indent=2))

    index_summary = {
        "task": "mci_time_to_ad",
        "mci_anchor_policy": args.mci_anchor_policy,
        "models": models,
        "features": features,
        "include_covariates": args.include_covariates,
        "runs": [
            {
                "model": summary["model"],
                "feature": summary["feature"],
                "output_dir": summary["output_dir"],
                "metrics": summary["metrics"],
                "selected_hyperparameters": summary["hyperparameter_tuning"]["selected"],
            }
            for summary in all_run_summaries
        ],
    }
    (args.output_dir / f"mci_time_to_ad_{args.mci_anchor_policy}_latest_summary.json").write_text(
        json.dumps(index_summary, indent=2),
        encoding="utf-8",
    )
