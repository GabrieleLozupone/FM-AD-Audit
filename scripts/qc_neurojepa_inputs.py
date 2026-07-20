#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extract_neurojepa_embeddings import (  # noqa: E402
    DEFAULT_HF_TOKEN_FILE,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_NAME_OR_PATH,
    build_preprocess_transform,
    load_backbone,
    resolve_device,
    resolve_hf_token,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "neurojepa_input_qc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC Neuro-JEPA inputs and small-sample embedding similarity before full extraction."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-path-column", type=str, default="processed_t1_path")
    parser.add_argument("--n-subjects", type=int, default=4)
    parser.add_argument("--sessions-per-subject", type=int, default=3)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--subject-id", action="append", default=None, help="Optional subject id. Repeatable.")
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--hf-revision", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--hf-token-file", type=Path, default=DEFAULT_HF_TOKEN_FILE)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    return parser.parse_args()


def choose_rows(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest_path, low_memory=False)
    required = ["participant_id", "session_id", "image_id", args.image_path_column]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    rows = manifest[manifest[args.image_path_column].notna()].copy()
    rows["source_image_path"] = rows[args.image_path_column].astype(str)
    rows = rows[rows["source_image_path"].map(lambda value: Path(value).exists())].copy()

    sort_columns = [column for column in ["participant_id", "age_at_session", "session_id"] if column in rows.columns]
    rows = rows.sort_values(sort_columns).reset_index(drop=True)

    selected_parts: list[pd.DataFrame] = []
    if args.subject_id:
        for subject_id in args.subject_id:
            subject_rows = rows[rows["participant_id"].astype(str) == subject_id].head(args.sessions_per_subject)
            if len(subject_rows) < args.min_sessions:
                raise ValueError(f"Subject {subject_id} has fewer than {args.min_sessions} available sessions.")
            selected_parts.append(subject_rows)
    else:
        counts = rows.groupby("participant_id").size()
        candidate_ids = counts[counts >= args.min_sessions].sort_values(ascending=False).head(args.n_subjects).index
        for subject_id in candidate_ids:
            selected_parts.append(rows[rows["participant_id"] == subject_id].head(args.sessions_per_subject))

    if not selected_parts:
        raise ValueError("No QC rows selected.")
    return pd.concat(selected_parts, ignore_index=True)


def as_numpy_volume(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 4:
        array = array[0]
    return np.asarray(array, dtype=np.float32)


def volume_stats(volume: np.ndarray) -> dict[str, float | list[int]]:
    nonzero = volume[np.abs(volume) > 1e-8]
    return {
        "shape": [int(v) for v in volume.shape],
        "min": float(np.min(volume)),
        "p01": float(np.percentile(volume, 1)),
        "p50": float(np.percentile(volume, 50)),
        "p99": float(np.percentile(volume, 99)),
        "max": float(np.max(volume)),
        "mean": float(np.mean(volume)),
        "std": float(np.std(volume)),
        "nonzero_fraction": float(nonzero.size / volume.size),
    }


def save_input_preview(volume: np.ndarray, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = [dim // 2 for dim in volume.shape]
    slices = [
        ("sagittal", np.rot90(volume[x, :, :])),
        ("coronal", np.rot90(volume[:, y, :])),
        ("axial", np.rot90(volume[:, :, z])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for axis, (name, slc) in zip(axes, slices):
        axis.imshow(slc, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    rows = choose_rows(args)
    transform = build_preprocess_transform()
    backbone = load_backbone(
        model_name_or_path=args.model_name_or_path,
        device=device,
        hf_revision=args.hf_revision,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=resolve_hf_token(args.hf_token, args.hf_token_file),
        checkpoint_filename=args.checkpoint_filename,
    )

    embedding_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    pooled_vectors: dict[str, np.ndarray] = {}

    for row in rows.itertuples(index=False):
        sample = transform({"image": row.source_image_path})
        image = sample["image"]
        volume = as_numpy_volume(image)

        participant_id = str(row.participant_id)
        session_id = str(row.session_id)
        image_id = str(row.image_id)
        case_dir = args.output_dir / participant_id / session_id
        preview_path = case_dir / f"{image_id}_neurojepa_input_qc.png"
        tokens_path = case_dir / f"{image_id}_neurojepa_tokens_qc.npy"
        pooled_path = case_dir / f"{image_id}_neurojepa_pooled_qc.npy"

        save_input_preview(volume, preview_path, f"{participant_id} {session_id} input to Neuro-JEPA")
        stats = volume_stats(volume)
        stats_rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "image_id": image_id,
                "source_image_path": row.source_image_path,
                "preview_path": str(preview_path),
                **stats,
            }
        )

        with torch.inference_mode():
            batch = image.unsqueeze(0).to(device)
            if device == "cuda":
                with torch.autocast(device_type="cuda"):
                    token_features, _ = backbone(batch)
            else:
                token_features, _ = backbone(batch)
            pooled_features = token_features.mean(dim=1)

        tokens_np = token_features.squeeze(0).detach().cpu().numpy().astype(save_dtype, copy=False)
        pooled_np = pooled_features.squeeze(0).detach().cpu().numpy().astype(save_dtype, copy=False)
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tokens_path, tokens_np)
        np.save(pooled_path, pooled_np)
        pooled_vectors[image_id] = pooled_np.astype(np.float32, copy=False)
        embedding_rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "image_id": image_id,
                "source_image_path": row.source_image_path,
                "preview_path": str(preview_path),
                "tokens_path": str(tokens_path),
                "pooled_path": str(pooled_path),
                "n_tokens": int(tokens_np.shape[0]),
                "token_dim": int(tokens_np.shape[1]),
                "pooled_dim": int(pooled_np.shape[0]),
            }
        )

    similarity_rows: list[dict[str, Any]] = []
    for i, left in enumerate(embedding_rows):
        for right in embedding_rows[i + 1 :]:
            similarity_rows.append(
                {
                    "left_participant_id": left["participant_id"],
                    "left_session_id": left["session_id"],
                    "left_image_id": left["image_id"],
                    "right_participant_id": right["participant_id"],
                    "right_session_id": right["session_id"],
                    "right_image_id": right["image_id"],
                    "same_subject": left["participant_id"] == right["participant_id"],
                    "cosine_similarity": cosine(pooled_vectors[left["image_id"]], pooled_vectors[right["image_id"]]),
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    embeddings_df = pd.DataFrame(embedding_rows)
    similarities_df = pd.DataFrame(similarity_rows)
    stats_path = args.output_dir / "neurojepa_input_qc_image_stats.csv"
    embeddings_path = args.output_dir / "neurojepa_input_qc_embeddings.csv"
    similarities_path = args.output_dir / "neurojepa_input_qc_similarities.csv"
    summary_path = args.output_dir / "neurojepa_input_qc_summary.json"
    stats_df.to_csv(stats_path, index=False)
    embeddings_df.to_csv(embeddings_path, index=False)
    similarities_df.to_csv(similarities_path, index=False)

    same = similarities_df[similarities_df["same_subject"]]
    different = similarities_df[~similarities_df["same_subject"]]
    summary = {
        "manifest_path": str(args.manifest_path),
        "image_path_column": args.image_path_column,
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_filename": args.checkpoint_filename or "model.safetensors",
        "device": device,
        "save_dtype": args.save_dtype,
        "n_subjects": int(embeddings_df["participant_id"].nunique()),
        "n_images": int(len(embeddings_df)),
        "input_shape_counts": stats_df["shape"].astype(str).value_counts().to_dict(),
        "mean_nonzero_fraction": float(stats_df["nonzero_fraction"].mean()),
        "mean_same_subject_cosine": float(same["cosine_similarity"].mean()) if not same.empty else None,
        "mean_different_subject_cosine": float(different["cosine_similarity"].mean()) if not different.empty else None,
        "stats_path": str(stats_path),
        "embeddings_path": str(embeddings_path),
        "similarities_path": str(similarities_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
