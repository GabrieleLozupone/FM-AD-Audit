#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "neurojepa_t1w_embeddings"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_neurojepa_embeddings.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "neurostar_adni_t1w_neurojepa_embeddings_summary.json"
DEFAULT_MODEL_NAME_OR_PATH = "NYUMedML/Neuro-JEPA"
DEFAULT_IMAGE_PATH_COLUMN = "processed_t1_path"
DEFAULT_FEATURE_KIND = "pooled"
DEFAULT_DTYPE = "float32"
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 2
DEFAULT_HF_TOKEN_FILE = PROJECT_ROOT / ".secrets" / "huggingface_token"


def ensure_neurojepa_on_path() -> None:
    """Validate that the separately installed, pinned Neuro-JEPA package is importable."""
    try:
        import neurojepa  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Neuro-JEPA is not installed. Install the pinned upstream checkout described in README.md."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reusable Neuro-JEPA embeddings for the NeuroSTAR ADNI T1w manifest."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--image-path-column",
        type=str,
        default=DEFAULT_IMAGE_PATH_COLUMN,
        choices=["processed_t1_path", "uniformres_t1_path", "raw_t1_path"],
    )
    parser.add_argument(
        "--feature-kind",
        type=str,
        default=DEFAULT_FEATURE_KIND,
        choices=["pooled", "tokens", "both"],
        help="`pooled` saves mean-pooled [768] vectors. `tokens` saves [576, 768] patch tokens.",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float16", "float32"],
        help="Storage dtype for saved numpy arrays.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scans to process.")
    parser.add_argument("--device", type=str, default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument(
        "--model-name-or-path",
        "--hf-model-id",
        dest="model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="Hugging Face repo ID or local directory/file supported by `load_backbone_from_hf`.",
    )
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None, help="Explicit Hugging Face token.")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=DEFAULT_HF_TOKEN_FILE,
        help="Local file containing a Hugging Face token. Default: .secrets/huggingface_token.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the existing index.")
    return parser.parse_args()


def require_dependency(module_name: str, install_hint: str) -> Any:
    try:
        return __import__(module_name, fromlist=["dummy"])
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency `{module_name}`. Install project requirements first: `{install_hint}`."
        ) from exc


def resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_token_file(token_path: Path | None) -> str | None:
    if token_path is None or not token_path.exists():
        return None
    token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def resolve_hf_token(cli_value: str | None, token_file: Path | None) -> str | bool | None:
    if cli_value is not None:
        return cli_value
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(env_name)
        if value:
            return value
    file_value = read_token_file(token_file)
    if file_value:
        return file_value
    return None


def build_preprocess_transform():
    monai_transforms = require_dependency("monai.transforms", "pip install -r requirements.txt")
    ensure_neurojepa_on_path()
    from neurojepa.data.transforms import remove_nan

    return monai_transforms.Compose(
        [
            monai_transforms.LoadImaged(keys=["image"], image_only=False),
            monai_transforms.EnsureChannelFirstd(keys=["image"]),
            monai_transforms.Lambdad(("image",), remove_nan),
            monai_transforms.Orientationd(keys=["image"], axcodes="RAS"),
            monai_transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=[5]),
            monai_transforms.CropForegroundd(
                keys=["image"],
                source_key="image",
                select_fn=lambda x: x > 0.0,
                margin=4,
                allow_smaller=True,
            ),
            monai_transforms.ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=[180, 216, 180],
                mode="edge",
            ),
            monai_transforms.Resized(keys=["image"], spatial_size=[100, 120, 100]),
            monai_transforms.CenterSpatialCropd(keys=["image"], roi_size=[96, 108, 96]),
            monai_transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            monai_transforms.CastToTyped(keys=["image"], dtype=np.float32),
            monai_transforms.ToTensord(keys=["image"]),
        ]
    )


def load_backbone(
    model_name_or_path: str,
    device: str,
    hf_revision: str | None,
    hf_cache_dir: Path | None,
    hf_token: str | bool | None,
    checkpoint_filename: str | None,
):
    ensure_neurojepa_on_path()
    require_dependency("safetensors", "pip install -r requirements.txt")
    require_dependency("torchmetrics", "pip install -r requirements.txt")
    from neurojepa.utils.init_utils import load_backbone_from_hf

    backbone = load_backbone_from_hf(
        model_name_or_path,
        device=device,
        revision=hf_revision,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
        token=hf_token,
        checkpoint_filename=checkpoint_filename,
    )
    backbone.eval()
    return backbone


def expected_output_paths(
    output_dir: Path,
    participant_id: str,
    session_id: str,
    image_id: str,
) -> tuple[Path, Path]:
    session_dir = output_dir / participant_id / session_id
    pooled_path = session_dir / f"{image_id}_neurojepa_pooled.npy"
    tokens_path = session_dir / f"{image_id}_neurojepa_tokens.npy"
    return pooled_path, tokens_path


def prepare_manifest_rows(
    manifest: pd.DataFrame,
    image_path_column: str,
    output_dir: Path,
    feature_kind: str,
    existing_index: pd.DataFrame | None,
) -> pd.DataFrame:
    if image_path_column not in manifest.columns:
        raise ValueError(f"Column `{image_path_column}` not found in manifest.")

    rows = manifest.copy()
    rows = rows[rows[image_path_column].notna()].copy()
    rows["source_image_path"] = rows[image_path_column].astype(str)
    rows["source_image_exists"] = rows["source_image_path"].map(lambda value: Path(value).exists())
    rows = rows[rows["source_image_exists"]].copy()

    pooled_paths: list[str] = []
    token_paths: list[str] = []
    for item in rows[["participant_id", "session_id", "image_id"]].itertuples(index=False):
        pooled_path, tokens_path = expected_output_paths(output_dir, item.participant_id, item.session_id, item.image_id)
        pooled_paths.append(str(pooled_path))
        token_paths.append(str(tokens_path))
    rows["pooled_embedding_path"] = pooled_paths
    rows["token_embedding_path"] = token_paths

    if feature_kind == "pooled":
        missing = ~rows["pooled_embedding_path"].map(lambda value: Path(value).exists())
    elif feature_kind == "tokens":
        missing = ~rows["token_embedding_path"].map(lambda value: Path(value).exists())
    else:
        missing = ~(
            rows["pooled_embedding_path"].map(lambda value: Path(value).exists())
            & rows["token_embedding_path"].map(lambda value: Path(value).exists())
        )
    return rows[missing].reset_index(drop=True)


def merge_embedding_index(existing_index: pd.DataFrame | None, new_index: pd.DataFrame) -> pd.DataFrame:
    if existing_index is None or existing_index.empty:
        return new_index
    if new_index.empty:
        return existing_index

    merged = existing_index.copy()
    for _, new_row in new_index.iterrows():
        image_id = str(new_row["image_id"])
        matches = merged["image_id"].astype(str) == image_id
        if not matches.any():
            merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
            continue

        row_idx = merged.index[matches][0]
        for column, value in new_row.items():
            if column not in merged.columns:
                merged[column] = pd.NA
            if pd.isna(value):
                continue
            merged.at[row_idx, column] = value
    return merged


class NeurostarEmbeddingDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, transform: Any):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        sample = self.transform({"image": row["source_image_path"]})
        return {
            "image": sample["image"],
            "participant_id": row["participant_id"],
            "session_id": row["session_id"],
            "image_id": row["image_id"],
            "source_image_path": row["source_image_path"],
            "pooled_embedding_path": row["pooled_embedding_path"],
            "token_embedding_path": row["token_embedding_path"],
        }


@dataclass
class ExtractionResult:
    index_rows: list[dict[str, Any]]
    failures: list[dict[str, Any]]


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def extract_embeddings(
    rows: pd.DataFrame,
    backbone: torch.nn.Module,
    transform: Any,
    device: str,
    feature_kind: str,
    save_dtype: str,
    batch_size: int,
    num_workers: int,
    model_name_or_path: str,
    checkpoint_filename: str | None,
    image_path_column: str,
) -> ExtractionResult:
    dataset = NeurostarEmbeddingDataset(rows=rows, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    np_dtype = np.float16 if save_dtype == "float16" else np.float32
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    progress = tqdm(
        total=len(dataset),
        desc=f"Neuro-JEPA {feature_kind}",
        unit="scan",
        dynamic_ncols=True,
    )

    for batch in loader:
        batch_size_actual = len(batch["image_id"])
        images = batch["image"].to(device)
        try:
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        token_features, _ = backbone(images)
                else:
                    token_features, _ = backbone(images)
            pooled_features = token_features.mean(dim=1)
        except Exception as exc:  # pragma: no cover
            for i in range(batch_size_actual):
                failures.append(
                    {
                        "image_id": batch["image_id"][i],
                        "participant_id": batch["participant_id"][i],
                        "session_id": batch["session_id"][i],
                        "source_image_path": batch["source_image_path"][i],
                        "error": str(exc),
                    }
                )
            progress.update(batch_size_actual)
            progress.set_postfix(ok=len(index_rows), fail=len(failures), refresh=False)
            continue

        pooled_np = pooled_features.detach().cpu().numpy().astype(np_dtype, copy=False)
        tokens_np = token_features.detach().cpu().numpy().astype(np_dtype, copy=False)

        for i in range(batch_size_actual):
            pooled_path = Path(batch["pooled_embedding_path"][i])
            token_path = Path(batch["token_embedding_path"][i])

            try:
                if feature_kind in {"pooled", "both"}:
                    save_array(pooled_path, pooled_np[i])
                if feature_kind in {"tokens", "both"}:
                    save_array(token_path, tokens_np[i])
            except Exception as exc:  # pragma: no cover
                failures.append(
                    {
                        "image_id": batch["image_id"][i],
                        "participant_id": batch["participant_id"][i],
                        "session_id": batch["session_id"][i],
                        "source_image_path": batch["source_image_path"][i],
                        "error": str(exc),
                    }
                )
                continue

            index_rows.append(
                {
                    "participant_id": batch["participant_id"][i],
                    "session_id": batch["session_id"][i],
                    "image_id": batch["image_id"][i],
                    "source_image_path": batch["source_image_path"][i],
                    "image_path_column": image_path_column,
                    "model_name_or_path": model_name_or_path,
                    "checkpoint_filename": checkpoint_filename or "model.safetensors",
                    "feature_kind": feature_kind,
                    "save_dtype": save_dtype,
                    "pooled_embedding_path": str(pooled_path) if feature_kind in {"pooled", "both"} else pd.NA,
                    "token_embedding_path": str(token_path) if feature_kind in {"tokens", "both"} else pd.NA,
                    "pooled_dim": int(pooled_np.shape[1]),
                    "n_tokens": int(tokens_np.shape[1]),
                    "token_dim": int(tokens_np.shape[2]),
                }
            )
        progress.update(batch_size_actual)
        progress.set_postfix(ok=len(index_rows), fail=len(failures), refresh=False)

    progress.close()

    return ExtractionResult(index_rows=index_rows, failures=failures)


def build_summary(
    manifest_rows: pd.DataFrame,
    index: pd.DataFrame,
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    return {
        "date": pd.Timestamp.utcnow().isoformat(),
        "manifest_path": str(args.manifest_path),
        "image_path_column": args.image_path_column,
        "model_name_or_path": args.model_name_or_path,
        "hf_revision": args.hf_revision,
        "checkpoint_filename": args.checkpoint_filename or "model.safetensors",
        "feature_kind": args.feature_kind,
        "save_dtype": args.save_dtype,
        "device": device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "n_manifest_rows_considered": int(len(manifest_rows)),
        "n_embeddings_written": int(len(index)),
        "n_unique_participants": int(index["participant_id"].nunique()) if not index.empty else 0,
        "n_failures": int(len(failures)),
        "output_dir": str(args.output_dir),
        "index_path": str(args.index_path),
        "summary_path": str(args.summary_path),
        "failures": failures[:20],
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_path, low_memory=False)
    existing_index = None
    if args.resume and args.index_path.exists():
        existing_index = pd.read_csv(args.index_path, low_memory=False)

    rows = prepare_manifest_rows(
        manifest=manifest,
        image_path_column=args.image_path_column,
        output_dir=args.output_dir,
        feature_kind=args.feature_kind,
        existing_index=existing_index,
    )
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    print(
        f"Prepared {len(rows)} scans for Neuro-JEPA extraction "
        f"(feature_kind={args.feature_kind}, device={device}, batch_size={args.batch_size})."
    )

    if rows.empty:
        final_index = existing_index if existing_index is not None else pd.DataFrame()
        summary = build_summary(manifest_rows=rows, index=final_index, failures=[], args=args, device=device)
        args.summary_path.write_text(json.dumps(summary, indent=2))
        print("No new rows to process.")
        return

    transform = build_preprocess_transform()
    backbone = load_backbone(
        model_name_or_path=args.model_name_or_path,
        device=device,
        hf_revision=args.hf_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=resolve_hf_token(args.hf_token, args.hf_token_file),
        checkpoint_filename=args.checkpoint_filename,
    )

    result = extract_embeddings(
        rows=rows,
        backbone=backbone,
        transform=transform,
        device=device,
        feature_kind=args.feature_kind,
        save_dtype=args.save_dtype,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_name_or_path=args.model_name_or_path,
        checkpoint_filename=args.checkpoint_filename,
        image_path_column=args.image_path_column,
    )

    new_index = pd.DataFrame(result.index_rows)
    if new_index.empty and (existing_index is None or existing_index.empty):
        final_index = pd.DataFrame(
            columns=[
                "participant_id",
                "session_id",
                "image_id",
                "source_image_path",
                "image_path_column",
                "model_name_or_path",
                "checkpoint_filename",
                "feature_kind",
                "save_dtype",
                "pooled_embedding_path",
                "token_embedding_path",
                "pooled_dim",
                "n_tokens",
                "token_dim",
            ]
        )
    elif existing_index is not None and not existing_index.empty:
        final_index = merge_embedding_index(existing_index, new_index)
    else:
        final_index = new_index
    if not final_index.empty:
        final_index = final_index.sort_values(["participant_id", "session_id", "image_id"]).reset_index(drop=True)
    final_index.to_csv(args.index_path, index=False)

    summary = build_summary(
        manifest_rows=rows,
        index=final_index,
        failures=result.failures,
        args=args,
        device=device,
    )
    args.summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote embedding index to {args.index_path}")
    print(f"Wrote embedding summary to {args.summary_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}, indent=2))


if __name__ == "__main__":
    main()
