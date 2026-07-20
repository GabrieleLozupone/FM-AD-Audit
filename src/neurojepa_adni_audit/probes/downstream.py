from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from neurojepa_adni_audit.paths import PROJECT_ROOT
from neurojepa_adni_audit.data.torch_datasets import TokenDataset
from neurojepa_adni_audit.features.covariates import (
    FORBIDDEN_COVARIATE_TOKENS,
    build_covariate_transform,
    coerce_numeric_series,
    infer_numeric_and_categorical,
    prepare_covariate_frame,
    validate_covariate_columns,
)
from neurojepa_adni_audit.features.progression_aligned import (
    STAGE1_FEATURE_MODES,
    filter_stage1_available_rows,
    is_stage1_feature,
    load_stage1_feature_matrix,
    stage1_feature_info,
)
from neurojepa_adni_audit.features.groups import (
    add_morphometry_ratios,
    feature_columns,
    is_feature_group,
    uses_neurojepa,
    validate_no_target_leakage,
)
from neurojepa_adni_audit.metrics import MetricResult, choose_threshold, compute_metrics, compute_metrics_by_split
from neurojepa_adni_audit.models.attention import AttentiveHead
from neurojepa_adni_audit.runtime import list_gpus_pci_order, print_gpus, resolve_device_name, set_gpu_visibility
from neurojepa_adni_audit.paper_run import bind_paper_run, require_protocol_grid


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_manifest.csv"
DEFAULT_EMBEDDING_INDEX_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_neurojepa_embeddings.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "neurojepa_downstream_probe"

DEFAULT_COVARIATE_COLUMNS = [
    "age_at_session",
    "sex_baseline",
    "apoe4_status",
    "education_years_baseline",
    "vol_total_intracranial",
    "vol_hippocampus_total",
    "vol_lateral_ventricles_total",
    "vol_left_amygdala",
    "vol_right_amygdala",
    "vol_left_inferior_lateral_ventricle",
    "vol_right_inferior_lateral_ventricle",
    "vol_left_cerebral_cortex",
    "vol_right_cerebral_cortex",
]
DEFAULT_LOGISTIC_C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

def require_sklearn() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency `scikit-learn`. Run `pip install -r requirements.txt`.") from exc
    return (
        ColumnTransformer,
        SimpleImputer,
        LogisticRegression,
        average_precision_score,
        balanced_accuracy_score,
        roc_auc_score,
        Pipeline,
        OneHotEncoder,
        StandardScaler,
    )


def resolve_device(device_arg: str | None) -> str:
    return resolve_device_name(device_arg)


def normalize_diagnosis(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"CN", "NL", "NORMAL", "COGNITIVELY NORMAL"}:
        return "CN"
    if text in {"AD", "DEMENTIA"}:
        return "AD"
    if text in {"MCI", "EMCI", "LMCI", "SMC", "MEMORY IMPAIRMENT"}:
        return "MCI"
    return text


def first_baseline_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    sortable = manifest.copy()
    sortable["session_month_sort"] = pd.to_numeric(sortable["session_month"], errors="coerce")
    sortable["session_month_sort"] = sortable["session_month_sort"].fillna(np.inf)
    sortable = sortable.sort_values(["participant_id", "session_month_sort", "session_id"]).reset_index(drop=True)
    return sortable.groupby("participant_id", as_index=False).head(1).reset_index(drop=True)


def build_mci_conversion_table(
    manifest: pd.DataFrame,
    conversion_window_months: float,
    stable_min_followup_months: float,
    allow_short_stable_followup: bool,
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

    labels: list[dict[str, Any]] = []
    for row in anchor_rows.itertuples(index=False):
        participant_id = getattr(row, "participant_id")
        session_id = getattr(row, "session_id")
        image_id = getattr(row, "image_id")
        anchor_month = getattr(row, "session_month_numeric")
        if pd.isna(anchor_month):
            continue
        subject_rows = sortable[sortable["participant_id"] == participant_id].copy()
        subject_rows = subject_rows[subject_rows["session_month_numeric"].notna()].copy()
        followup_rows = subject_rows[subject_rows["session_month_numeric"] >= float(anchor_month)].copy()
        max_followup_months = float(followup_rows["session_month_numeric"].max() - float(anchor_month))
        future_ad_rows = followup_rows[
            (followup_rows["diagnosis_norm"] == "AD")
            & (followup_rows["session_month_numeric"] > float(anchor_month))
        ].copy()
        first_ad_delta = None
        if not future_ad_rows.empty:
            first_ad_delta = float(future_ad_rows["session_month_numeric"].min() - float(anchor_month))

        converted_within_window = first_ad_delta is not None and first_ad_delta <= conversion_window_months
        if converted_within_window:
            status = "converter_within_window"
        elif first_ad_delta is not None:
            continue
        elif allow_short_stable_followup or max_followup_months >= stable_min_followup_months:
            status = "stable_never_ad_observed"
        else:
            continue

        labels.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "image_id": image_id,
                "mci_anchor_id": f"{participant_id}_{session_id}",
                "mci_conversion_label_status": status,
                "label": int(converted_within_window),
                "anchor_month": float(anchor_month),
                "baseline_month": float(anchor_month),
                "followup_months_observed": max_followup_months,
                "months_to_first_ad": first_ad_delta,
                "months_to_first_ad_within_window": first_ad_delta if converted_within_window else None,
            }
        )

    if not labels:
        return anchor_rows.iloc[0:0].copy()
    label_table = pd.DataFrame(labels)
    return anchor_rows.merge(label_table, on=["participant_id", "session_id", "image_id"], how="inner")


def load_task_table(
    manifest_path: Path,
    embedding_index_path: Path,
    task: str,
    conversion_window_months: float = 36.0,
    stable_min_followup_months: float = 36.0,
    allow_short_stable_followup: bool = False,
    mci_anchor_policy: str = "all",
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

    if task == "ad_vs_cn":
        baseline_table = first_baseline_rows(manifest)
        baseline_table["diagnosis_norm"] = baseline_table["diagnosis_at_session"].map(normalize_diagnosis)
        baseline_table = baseline_table[baseline_table["diagnosis_norm"].isin(["CN", "AD"])].copy()
        baseline_table["label"] = (baseline_table["diagnosis_norm"] == "AD").astype(int)
    elif task == "mci_conversion_36m":
        baseline_table = build_mci_conversion_table(
            manifest=manifest,
            conversion_window_months=conversion_window_months,
            stable_min_followup_months=stable_min_followup_months,
            allow_short_stable_followup=allow_short_stable_followup,
            mci_anchor_policy=mci_anchor_policy,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported task: {task}")

    table = baseline_table.merge(
        index.drop_duplicates("image_id"),
        on=["participant_id", "session_id", "image_id"],
        how="inner",
        suffixes=("", "_embedding"),
    )

    if "subject_split" not in table.columns:
        raise ValueError("Merged task table does not contain `subject_split`.")
    table = table[table["subject_split"].isin(["train", "val", "test"])].copy()
    return table.reset_index(drop=True)


def validate_feature_paths(table: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature == "covariates":
        return table.reset_index(drop=True)
    if is_stage1_feature(feature):
        filtered = filter_stage1_available_rows(table, image_id_column="image_id")
        if filtered.empty:
            raise ValueError(f"No task rows have available Stage 1 latents for feature `{feature}`.")
        return filtered
    column = "pooled_embedding_path" if feature == "pooled" else "token_embedding_path"
    if column not in table.columns:
        raise ValueError(f"Embedding index is missing `{column}`.")
    filtered = table[table[column].notna()].copy()
    filtered = filtered[filtered[column].map(lambda value: Path(str(value)).exists())].copy()
    if filtered.empty:
        raise ValueError(f"No usable `{column}` files found for feature `{feature}`.")
    return filtered.reset_index(drop=True)


def split_rows(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {split: table[table["subject_split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}


def load_pooled_matrix(rows: pd.DataFrame) -> np.ndarray:
    vectors = [np.load(Path(str(path))).astype(np.float32, copy=False).reshape(-1) for path in rows["pooled_embedding_path"]]
    return np.stack(vectors, axis=0)


def build_prediction_table(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    val_threshold: float,
    fixed_threshold: float = 0.5,
) -> pd.DataFrame:
    columns = ["participant_id", "session_id", "image_id", "subject_split", "diagnosis_at_session", "label"]
    optional_columns = [
        "diagnosis_norm",
        "mci_anchor_id",
        "mci_conversion_label_status",
        "anchor_month",
        "baseline_month",
        "followup_months_observed",
        "months_to_first_ad",
        "months_to_first_ad_within_window",
    ]
    columns.extend([column for column in optional_columns if column in table.columns])
    predictions = table[columns].copy()
    predictions["probability_positive"] = probabilities
    predictions["predicted_label"] = (predictions["probability_positive"].to_numpy() >= val_threshold).astype(int)
    predictions["predicted_label_val_threshold"] = predictions["predicted_label"]
    predictions["predicted_label_threshold_0_5"] = (
        predictions["probability_positive"].to_numpy() >= fixed_threshold
    ).astype(int)
    return predictions


def build_logistic_model(seed: int, c_value: float = 1.0) -> Any:
    (
        _ColumnTransformer,
        SimpleImputer,
        LogisticRegression,
        _average_precision_score,
        _balanced_accuracy_score,
        _roc_auc_score,
        Pipeline,
        _OneHotEncoder,
        StandardScaler,
    ) = require_sklearn()
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def select_logistic_c(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[float, list[dict[str, float | None]]]:
    (*_unused, average_precision_score, _balanced_accuracy, _auroc, _pipeline, _onehot, _scaler) = require_sklearn()
    history: list[dict[str, float | None]] = []
    best_c = float(c_values[0])
    best_score = -np.inf
    for c_value in c_values:
        model = build_logistic_model(seed, float(c_value))
        model.fit(x_all[train_mask], y_all[train_mask])
        probabilities = model.predict_proba(x_all[val_mask])[:, 1]
        score = (
            float(average_precision_score(y_all[val_mask], probabilities))
            if len(np.unique(y_all[val_mask])) == 2
            else None
        )
        history.append({"c": float(c_value), "val_auprc": score})
        comparable = -np.inf if score is None else score
        if comparable > best_score:
            best_score = comparable
            best_c = float(c_value)
    return best_c, history


def train_covariates_probe(
    table: pd.DataFrame,
    covariate_columns: list[str],
    seed: int,
    c_values: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    (_ColumnTransformer, _SimpleImputer, LogisticRegression, *_rest) = require_sklearn()
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")

    train_mask = table["subject_split"].to_numpy() == "train"
    _transform, numeric, categorical, x_all = build_covariate_transform(
        train_rows=table[train_mask],
        all_rows=table,
        covariate_columns=covariate_columns,
    )
    if x_all is None:
        raise ValueError("No usable covariate columns were found for `--feature covariates`.")

    y_all = table["label"].to_numpy(dtype=int)
    val_mask = table["subject_split"].to_numpy() == "val"
    best_c, c_history = select_logistic_c(x_all, y_all, train_mask, val_mask, c_values, seed)
    model = LogisticRegression(
        C=best_c, max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed
    )
    model.fit(x_all[train_mask], y_all[train_mask])
    probabilities = model.predict_proba(x_all)[:, 1]
    threshold = choose_threshold(y_all[val_mask], probabilities[val_mask])

    predictions = build_prediction_table(table, probabilities, threshold)
    summary = {
        "model": "logistic_regression_l2",
        "feature": "covariates",
        "include_covariates": True,
        "covariates": {"included": True, "numeric_columns": numeric, "categorical_columns": categorical},
        "n_input_features": int(x_all.shape[1]),
        "selected_c": best_c,
        "regularization_selection_metric": "validation_auprc",
        "c_selection_history": c_history,
        "threshold_selected_on_val": float(threshold),
        "fixed_threshold": 0.5,
        "metrics": compute_metrics_by_split(table, y_all, probabilities, threshold),
        "metrics_at_threshold_0_5": compute_metrics_by_split(table, y_all, probabilities, 0.5),
    }
    return summary, predictions


def train_pooled_probe(
    table: pd.DataFrame,
    feature: str,
    include_covariates: bool,
    covariate_columns: list[str],
    seed: int,
    c_values: list[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")

    x_all = load_stage1_feature_matrix(table, feature) if is_stage1_feature(feature) else load_pooled_matrix(table)
    train_mask = table["subject_split"].to_numpy() == "train"
    cov_info: dict[str, Any] = {"included": False, "numeric_columns": [], "categorical_columns": []}
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

    y_all = table["label"].to_numpy(dtype=int)
    val_mask = table["subject_split"].to_numpy() == "val"
    best_c, c_history = select_logistic_c(x_all, y_all, train_mask, val_mask, c_values, seed)
    model = build_logistic_model(seed, best_c)
    model.fit(x_all[train_mask], y_all[train_mask])
    probabilities = model.predict_proba(x_all)[:, 1]
    threshold = choose_threshold(y_all[val_mask], probabilities[val_mask])

    predictions = build_prediction_table(table, probabilities, threshold)

    summary = {
        "model": "logistic_regression_l2",
        "feature": feature,
        "include_covariates": include_covariates,
        "covariates": cov_info,
        "n_input_features": int(x_all.shape[1]),
        "selected_c": best_c,
        "regularization_selection_metric": "validation_auprc",
        "c_selection_history": c_history,
        "threshold_selected_on_val": float(threshold),
        "fixed_threshold": 0.5,
        "metrics": compute_metrics_by_split(table, y_all, probabilities, threshold),
        "metrics_at_threshold_0_5": compute_metrics_by_split(table, y_all, probabilities, 0.5),
    }
    return summary, predictions


def make_token_loader(
    rows: pd.DataFrame,
    covariates: np.ndarray | None,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TokenDataset(rows, "token_embedding_path", covariates, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: AttentiveHead,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    probs: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["label"].to(device).view(-1, 1)
        covariates = batch.get("covariates")
        if covariates is not None:
            covariates = covariates.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens, covariates)
        loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.size(0)
        total_n += labels.size(0)
        probs.append(torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1))
        labels_out.append(labels.detach().cpu().numpy().reshape(-1))
    return total_loss / max(1, total_n), np.concatenate(probs), np.concatenate(labels_out)


def train_tokens_attn_probe(
    table: pd.DataFrame,
    include_covariates: bool,
    covariate_columns: list[str],
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_heads: int,
    attn_dropout: float,
    head_dropout: float,
    gradient_clip_norm: float,
    early_stop_metric: str,
    lr_scheduler: str,
    min_delta: float,
    num_workers: int,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = split_rows(table)
    if splits["train"].empty or splits["val"].empty or splits["test"].empty:
        raise ValueError("Train, val, and test must all contain at least one row.")

    cov_all = None
    cov_info: dict[str, Any] = {"included": False, "numeric_columns": [], "categorical_columns": []}
    if include_covariates:
        train_mask = table["subject_split"].to_numpy() == "train"
        _transform, numeric, categorical, cov_all = build_covariate_transform(
            train_rows=table[train_mask],
            all_rows=table,
            covariate_columns=covariate_columns,
        )
        if cov_all is not None:
            cov_info = {"included": True, "numeric_columns": numeric, "categorical_columns": categorical}

    split_indices = {split: table.index[table["subject_split"] == split].to_numpy() for split in ("train", "val", "test")}
    labels = table["label"].to_numpy(dtype=np.float32)
    loaders = {}
    for split, indices in split_indices.items():
        split_cov = cov_all[indices] if cov_all is not None else None
        loaders[split] = make_token_loader(
            rows=table.iloc[indices].reset_index(drop=True),
            covariates=split_cov,
            labels=labels[indices],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )

    pos = float(np.sum(labels[split_indices["train"]] == 1))
    neg = float(np.sum(labels[split_indices["train"]] == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    model = AttentiveHead(
        n_covariates=0 if cov_all is None else cov_all.shape[1],
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        head_dropout=head_dropout,
    ).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler: Any | None = None
    if lr_scheduler == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if early_stop_metric == "loss" else "max",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=lr * 0.01,
        )
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, max_epochs),
            eta_min=lr * 0.01,
        )

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int | None]] = []

    progress = tqdm(range(1, max_epochs + 1), desc="tokens_attn", unit="epoch", dynamic_ncols=True)
    for epoch in progress:
        train_loss, _train_prob, _train_labels = run_epoch(
            model,
            loaders["train"],
            loss_fn,
            device,
            optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_loss, val_prob, val_labels = run_epoch(model, loaders["val"], loss_fn, device)
        val_threshold = choose_threshold(val_labels.astype(int), val_prob)
        val_metric = compute_metrics("val", val_labels.astype(int), val_prob, val_threshold)
        metric_values = {
            "auroc": val_metric.auroc,
            "auprc": val_metric.auprc,
            "balanced_accuracy": val_metric.balanced_accuracy,
            "loss": -val_loss,
        }
        val_score = metric_values[early_stop_metric]
        if val_score is None:
            val_score = -np.inf
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_metric.auroc,
                "val_auprc": val_metric.auprc,
                "val_balanced_accuracy": val_metric.balanced_accuracy,
                "lr": current_lr,
            }
        )
        progress.set_postfix(val_score=val_score, val_loss=val_loss, lr=current_lr)
        if scheduler is not None and lr_scheduler == "reduce_on_plateau":
            scheduler.step(val_loss if early_stop_metric == "loss" else val_score)
        elif scheduler is not None:
            scheduler.step()
        if val_score > best_score + min_delta:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    progress.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = output_dir / "attentive_head.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "num_heads": num_heads,
            "attn_dropout": attn_dropout,
            "head_dropout": head_dropout,
            "gradient_clip_norm": gradient_clip_norm,
            "early_stop_metric": early_stop_metric,
            "lr_scheduler": lr_scheduler,
            "min_delta": min_delta,
            "include_covariates": include_covariates,
            "covariates": cov_info,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    split_probs: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split, loader in loaders.items():
        _loss, prob, label_values = run_epoch(model, loader, loss_fn, device)
        split_probs[split] = prob
        split_labels[split] = label_values.astype(int)
    threshold = choose_threshold(split_labels["val"], split_probs["val"])
    fixed_threshold = 0.5
    metrics = [
        asdict(compute_metrics(split, split_labels[split], split_probs[split], threshold))
        for split in ("train", "val", "test")
    ]
    metrics_at_threshold_0_5 = [
        asdict(compute_metrics(split, split_labels[split], split_probs[split], fixed_threshold))
        for split in ("train", "val", "test")
    ]

    prediction_frames = []
    for split, indices in split_indices.items():
        columns = ["participant_id", "session_id", "image_id", "subject_split", "diagnosis_at_session", "label"]
        optional_columns = [
            "diagnosis_norm",
            "mci_anchor_id",
            "mci_conversion_label_status",
            "anchor_month",
            "baseline_month",
            "followup_months_observed",
            "months_to_first_ad",
            "months_to_first_ad_within_window",
        ]
        columns.extend([column for column in optional_columns if column in table.columns])
        frame = table.iloc[indices][columns].copy()
        frame["probability_positive"] = split_probs[split]
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["predicted_label"] = (predictions["probability_positive"].to_numpy() >= threshold).astype(int)
    predictions["predicted_label_val_threshold"] = predictions["predicted_label"]
    predictions["predicted_label_threshold_0_5"] = (
        predictions["probability_positive"].to_numpy() >= fixed_threshold
    ).astype(int)

    summary = {
        "model": "cross_attention_pooling_linear",
        "feature": "tokens_attn",
        "include_covariates": include_covariates,
        "covariates": cov_info,
        "device": device,
        "max_epochs": max_epochs,
        "patience": patience,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "num_heads": num_heads,
        "attn_dropout": attn_dropout,
        "head_dropout": head_dropout,
        "gradient_clip_norm": gradient_clip_norm,
        "early_stop_metric": early_stop_metric,
        "lr_scheduler": lr_scheduler,
        "min_delta": min_delta,
        "checkpoint_path": str(checkpoint_path),
        "threshold_selected_on_val": float(threshold),
        "fixed_threshold": fixed_threshold,
        "metrics": metrics,
        "metrics_at_threshold_0_5": metrics_at_threshold_0_5,
        "history": history,
    }
    return summary, predictions


def summarize_task_table(table: pd.DataFrame) -> dict[str, Any]:
    by_split = {}
    for split, group in table.groupby("subject_split"):
        anchors_per_subject = group.groupby("participant_id").size()
        by_split[split] = {
            "n": int(len(group)),
            "n_subjects": int(group["participant_id"].nunique()),
            "n_images": int(group["image_id"].nunique()) if "image_id" in group.columns else None,
            "mean_anchors_per_subject": float(anchors_per_subject.mean()) if not anchors_per_subject.empty else 0.0,
            "max_anchors_per_subject": int(anchors_per_subject.max()) if not anchors_per_subject.empty else 0,
            "diagnoses": {str(k): int(v) for k, v in group["diagnosis_norm"].value_counts().to_dict().items()}
            if "diagnosis_norm" in group.columns
            else {},
            "labels": {str(k): int(v) for k, v in group["label"].value_counts().to_dict().items()},
            "mci_conversion_status": {
                str(k): int(v) for k, v in group.get("mci_conversion_label_status", pd.Series(dtype=object)).value_counts().to_dict().items()
            },
        }
    return {
        "n_rows": int(len(table)),
        "n_subjects": int(table["participant_id"].nunique()),
        "n_images": int(table["image_id"].nunique()) if "image_id" in table.columns else None,
        "n_subjects_with_multiple_anchors": int((table.groupby("participant_id").size() > 1).sum()),
        "by_split": by_split,
    }


def write_run_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    task_table: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    task_table.to_csv(output_dir / "task_table.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(args: Any) -> None:
    if args.list_gpus:
        print_gpus()
        return
    features = args.feature or ["pooled"]
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
            args.logistic_c,
            paper_context.protocol.payload["logistic_c_values"],
            "Logistic C grid",
        )
        args.seed = paper_context.replication.split_seed
    if not args.logistic_c or any(float(value) <= 0 for value in args.logistic_c):
        raise ValueError("`--logistic-c` must contain at least one positive value.")
    stage1_covariate_modes = {f"{mode}_covariates": mode for mode in STAGE1_FEATURE_MODES}
    if args.include_covariates or "covariates" in features or any(feature in stage1_covariate_modes for feature in features):
        validate_covariate_columns(args.covariate_columns)
    set_gpu_visibility(args.gpu_index)
    device = resolve_device(args.device)
    base_table = load_task_table(
        args.manifest_path,
        args.embedding_index_path,
        args.task,
        conversion_window_months=args.conversion_window_months,
        stable_min_followup_months=args.stable_min_followup_months,
        allow_short_stable_followup=args.allow_short_stable_followup,
        mci_anchor_policy=args.mci_anchor_policy,
    )
    if base_table.empty:
        raise ValueError(f"No rows available for task `{args.task}` after merging manifest and embeddings.")
    base_table = add_morphometry_ratios(base_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_run_summaries = []
    for feature in features:
        canonical_group = is_feature_group(feature)
        if canonical_group:
            selected_covariates = feature_columns(args.task, feature)
            validate_no_target_leakage(args.task, "label", selected_covariates)
            model_feature = "pooled" if uses_neurojepa(feature) else "covariates"
            include_covariates = uses_neurojepa(feature) and bool(selected_covariates)
        else:
            selected_covariates = args.covariate_columns
            model_feature = stage1_covariate_modes.get(feature, feature)
            include_covariates = args.include_covariates or feature in stage1_covariate_modes
        table = validate_feature_paths(base_table, model_feature)
        run_name = f"{args.task}_{feature}"
        if (
            include_covariates
            and not canonical_group
            and feature != "covariates"
            and feature not in stage1_covariate_modes
        ):
            run_name += "_covariates"
        run_dir = args.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if model_feature == "covariates":
            summary, predictions = train_covariates_probe(
                table=table,
                covariate_columns=selected_covariates,
                seed=args.seed,
                c_values=args.logistic_c,
            )
        elif model_feature == "pooled" or is_stage1_feature(model_feature):
            summary, predictions = train_pooled_probe(
                table=table,
                feature=model_feature,
                include_covariates=include_covariates,
                covariate_columns=selected_covariates,
                seed=args.seed,
                c_values=args.logistic_c,
            )
        else:
            summary, predictions = train_tokens_attn_probe(
                table=table,
                include_covariates=include_covariates,
                covariate_columns=selected_covariates,
                seed=args.seed,
                device=device,
                max_epochs=args.max_epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_heads=args.num_heads,
                attn_dropout=args.attn_dropout,
                head_dropout=args.head_dropout,
                gradient_clip_norm=args.gradient_clip_norm,
                early_stop_metric=args.early_stop_metric,
                lr_scheduler=args.lr_scheduler,
                min_delta=args.min_delta,
                num_workers=args.num_workers,
                output_dir=run_dir,
            )

        summary = {
            "task": args.task,
            "feature": feature,
            "manifest_path": str(args.manifest_path),
            "embedding_index_path": str(args.embedding_index_path),
            "output_dir": str(run_dir),
            "seed": args.seed,
            "paper_run": paper_context.metadata() if paper_context is not None else None,
            "label_definition": {
                "positive_label": "AD" if args.task == "ad_vs_cn" else "MCI converted to AD within window",
                "negative_label": "CN" if args.task == "ad_vs_cn" else "MCI anchor from a never-AD observed subject",
                "conversion_window_months": args.conversion_window_months
                if args.task == "mci_conversion_36m"
                else None,
                "stable_min_followup_months": args.stable_min_followup_months
                if args.task == "mci_conversion_36m"
                else None,
                "allow_short_stable_followup": args.allow_short_stable_followup
                if args.task == "mci_conversion_36m"
                else None,
                "mci_anchor_policy": args.mci_anchor_policy if args.task == "mci_conversion_36m" else None,
                "negative_rule": (
                    "MCI anchor has no AD conversion anywhere in observed follow-up and has enough observed follow-up"
                    if args.task == "mci_conversion_36m"
                    else None
                ),
                "late_converters": (
                    "Excluded from the negative class if first AD occurs after the conversion window"
                    if args.task == "mci_conversion_36m"
                    else None
                ),
            },
            "task_table": summarize_task_table(table),
            **summary,
        }
        summary["feature"] = feature
        summary["feature_group"] = feature if canonical_group else None
        summary["source_covariate_columns"] = selected_covariates
        write_run_outputs(run_dir, summary, predictions, table)
        all_run_summaries.append(summary)

        compact_metrics = {item["split"]: item for item in summary["metrics"]}
        compact_metrics_0_5 = {item["split"]: item for item in summary["metrics_at_threshold_0_5"]}
        print(f"Wrote {feature} run to {run_dir}")
        print("Validation-selected threshold metrics:")
        print(json.dumps(compact_metrics, indent=2))
        print("Fixed 0.5 threshold metrics:")
        print(json.dumps(compact_metrics_0_5, indent=2))

    index_summary = {
        "task": args.task,
        "features": features,
        "include_covariates": args.include_covariates,
        "runs": [
            {
                "feature": summary["feature"],
                "output_dir": summary["output_dir"],
                "metrics": summary["metrics"],
                "metrics_at_threshold_0_5": summary["metrics_at_threshold_0_5"],
            }
            for summary in all_run_summaries
        ],
    }
    (args.output_dir / f"{args.task}_latest_summary.json").write_text(
        json.dumps(index_summary, indent=2),
        encoding="utf-8",
    )
