from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.replications import sha256_file


KEY_COLUMNS = ("participant_id", "session_id", "image_id")
POOLED_DIMENSION = 768


def configure_gpu(gpu: int | None) -> None:
    """Select one GPU in PCI bus order (the nvidia-smi index); call before any CUDA use."""
    if gpu is None:
        return
    if gpu < 0:
        raise ValueError("--gpu must be a non-negative nvidia-smi index.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def set_full_precision() -> None:
    """Disable TF32 so frozen features do not depend on GPU generation defaults."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_image_list(path: Path, image_path_column: str = "source_image_path") -> pd.DataFrame:
    """Read the locked image list: unique keys and one existing source image per row."""
    table = pd.read_csv(path, dtype={column: "string" for column in KEY_COLUMNS}, low_memory=False)
    missing_columns = [column for column in (*KEY_COLUMNS, image_path_column) if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Image list {path} is missing columns: {missing_columns}")
    rows = table[[*KEY_COLUMNS, image_path_column]].rename(columns={image_path_column: "source_image_path"})
    if rows.isna().any().any():
        raise ValueError("Image list contains missing keys or image paths.")
    if rows["image_id"].duplicated().any():
        raise ValueError("Image list contains duplicated image_id values.")
    if rows.duplicated(["participant_id", "session_id"]).any():
        raise ValueError("Image list contains more than one image per participant session.")
    return rows.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def output_path(output_dir: Path, participant_id: str, session_id: str, image_id: str, suffix: str) -> Path:
    return output_dir / participant_id / session_id / f"{image_id}_{suffix}.npy"


def save_array_atomic(path: Path, array: np.ndarray) -> None:
    """Write float32 .npy via a temporary file so interrupted runs never leave partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Refusing to save non-finite values to {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def outputs_complete(paths: list[Path]) -> bool:
    return all(path.is_file() for path in paths)


def build_index(
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    pooled_suffix: str,
    encoder: str,
    representation: str,
    extra_suffixes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Embedding index in the schema expected by build_embedding_integrity_manifest.py."""
    index = rows[[*KEY_COLUMNS, "source_image_path"]].copy()
    index["encoder"] = encoder
    index["representation"] = representation
    index["pooled_embedding_path"] = [
        str(output_path(output_dir, *keys, pooled_suffix))
        for keys in index[list(KEY_COLUMNS)].itertuples(index=False)
    ]
    for column, suffix in (extra_suffixes or {}).items():
        index[column] = [
            str(output_path(output_dir, *keys, suffix))
            for keys in index[list(KEY_COLUMNS)].itertuples(index=False)
        ]
    index["pooled_dim"] = POOLED_DIMENSION
    return index


def runtime_metadata() -> dict[str, Any]:
    import torch

    packages: dict[str, str | None] = {"torch": torch.__version__, "numpy": np.__version__}
    for name in ("monai", "transformers", "nibabel"):
        try:
            packages[name] = __import__(name).__version__
        except ImportError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_encoder_spec(protocol_path: Path, encoder: str) -> dict[str, Any]:
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    try:
        return payload["additional_encoders"][encoder]
    except KeyError as exc:
        raise ValueError(f"Protocol {protocol_path} defines no additional encoder `{encoder}`.") from exc


def require_sha256(path: Path, expected: str | None, label: str) -> str:
    if not expected:
        raise ValueError(f"The protocol does not pin a SHA-256 for {label}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: observed={observed}, expected={expected}")
    return observed


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
