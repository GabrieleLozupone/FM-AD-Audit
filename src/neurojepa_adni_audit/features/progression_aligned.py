from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.paths import PROJECT_ROOT


DEFAULT_STAGE1_LATENTS_PATH = PROJECT_ROOT / "data" / "progression_aligned_embeddings" / "stage1_visit_latents.npz"
DEFAULT_STAGE1_INDEX_PATH = PROJECT_ROOT / "data" / "progression_aligned_embeddings" / "stage1_visit_latents_index.csv"

STAGE1_BLOCK_KEYS = {
    "stage1_s": ["identity"],
    "stage1_a": ["aging"],
    "stage1_p": ["pathology"],
    "stage1_e": ["residual"],
    "stage1_ap": ["aging", "pathology"],
    "stage1_z": ["z"],
}
STAGE1_FEATURE_MODES = tuple(STAGE1_BLOCK_KEYS)


def is_stage1_feature(feature: str) -> bool:
    return feature in STAGE1_BLOCK_KEYS


def resolve_stage1_latents_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("STAGE1_LATENTS_PATH", DEFAULT_STAGE1_LATENTS_PATH))


def resolve_stage1_index_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("STAGE1_INDEX_PATH", DEFAULT_STAGE1_INDEX_PATH))


def stage1_index_image_ids(index_path: Path | None = None) -> set[str]:
    index_path = resolve_stage1_index_path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Stage 1 latent index not found: {index_path}")
    index = pd.read_csv(index_path, low_memory=False)
    if "image_id" not in index.columns:
        raise ValueError(f"Stage 1 latent index is missing `image_id`: {index_path}")
    return set(index["image_id"].astype(str))


def filter_stage1_available_rows(
    rows: pd.DataFrame,
    image_id_column: str = "image_id",
    index_path: Path | None = None,
) -> pd.DataFrame:
    if image_id_column not in rows.columns:
        raise ValueError(f"Rows are missing image id column `{image_id_column}`.")
    available = stage1_index_image_ids(index_path)
    mask = rows[image_id_column].astype(str).isin(available)
    return rows[mask].reset_index(drop=True)


def load_stage1_feature_matrix(
    rows: pd.DataFrame,
    feature: str,
    latents_path: Path | None = None,
    index_path: Path | None = None,
    image_id_column: str = "image_id",
) -> np.ndarray:
    if feature not in STAGE1_BLOCK_KEYS:
        raise ValueError(f"Unsupported Stage 1 feature mode: {feature}")
    if image_id_column not in rows.columns:
        raise ValueError(f"Rows are missing image id column `{image_id_column}`.")
    latents_path = resolve_stage1_latents_path(latents_path)
    index_path = resolve_stage1_index_path(index_path)
    if not latents_path.exists():
        raise FileNotFoundError(f"Stage 1 latents not found: {latents_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Stage 1 latent index not found: {index_path}")

    index = pd.read_csv(index_path, low_memory=False)
    if "image_id" not in index.columns:
        raise ValueError(f"Stage 1 latent index is missing `image_id`: {index_path}")
    position_by_image = {str(image_id): int(pos) for pos, image_id in enumerate(index["image_id"].astype(str))}
    positions = []
    missing = []
    for image_id in rows[image_id_column].astype(str):
        position = position_by_image.get(image_id)
        if position is None:
            missing.append(image_id)
        else:
            positions.append(position)
    if missing:
        examples = ", ".join(missing[:5])
        raise ValueError(
            f"Stage 1 latents are missing {len(missing)} image_id value(s) for column `{image_id_column}`. "
            f"Examples: {examples}"
        )

    archive = np.load(latents_path)
    parts = []
    for key in STAGE1_BLOCK_KEYS[feature]:
        if key not in archive.files:
            raise ValueError(f"Stage 1 latents file is missing `{key}` array: {latents_path}")
        parts.append(archive[key][positions].astype(np.float32, copy=False))
    matrix = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if matrix.shape[1] == 0:
        raise ValueError(
            f"Stage 1 feature `{feature}` has zero columns in {latents_path}. "
            "This usually means the selected Stage 1 run disabled that component."
        )
    return matrix


def stage1_feature_info(feature: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "feature": feature,
        "blocks": STAGE1_BLOCK_KEYS[feature],
        "latents_path": str(resolve_stage1_latents_path()),
        "index_path": str(resolve_stage1_index_path()),
    }
