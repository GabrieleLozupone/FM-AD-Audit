from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurojepa_adni_audit.replications import sha256_file


KEY_COLUMNS = ("participant_id", "session_id", "image_id")
POOLED_PATH_COLUMN = "pooled_embedding_path"


def _read_keys(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    table = pd.read_csv(path, dtype={column: "string" for column in KEY_COLUMNS}, low_memory=False)
    missing = sorted(set(KEY_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")
    if table[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError(f"{label} contains null participant/session/image keys.")
    if table.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{label} contains duplicate participant/session/image keys.")
    if table["image_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate image_id values.")
    return table


def validate_index_keys(index: pd.DataFrame, source_manifest: pd.DataFrame) -> None:
    index_keys = index.loc[:, KEY_COLUMNS].astype("string")
    source_keys = source_manifest.loc[:, KEY_COLUMNS].astype("string")
    comparison = index_keys.merge(source_keys, on=list(KEY_COLUMNS), how="outer", indicator=True)
    counts = comparison["_merge"].value_counts()
    index_only = int(counts.get("left_only", 0))
    source_only = int(counts.get("right_only", 0))
    if index_only or source_only:
        raise ValueError(
            "Embedding index and source manifest keys differ: "
            f"{index_only} index-only and {source_only} manifest-only rows."
        )


def inspect_pooled_embedding(path: Path, *, expected_dimension: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pooled embedding not found: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not load pooled embedding {path}: {exc}") from exc
    if array.shape != (expected_dimension,):
        raise ValueError(
            f"Pooled embedding {path} has shape {array.shape}; expected ({expected_dimension},)."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Pooled embedding {path} has non-numeric dtype {array.dtype}.")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"Pooled embedding {path} contains non-finite values.")
    return {
        "pooled_embedding_path": str(path.resolve()),
        "pooled_embedding_sha256": sha256_file(path),
        "pooled_embedding_bytes": int(path.stat().st_size),
        "pooled_embedding_dtype": str(array.dtype),
        "pooled_embedding_shape": json.dumps(list(array.shape), separators=(",", ":")),
        "pooled_embedding_finite": True,
    }


def build_embedding_integrity_manifest(
    *,
    embedding_index_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
    expected_dimension: int = 768,
) -> Path:
    """Validate and hash the immutable pooled embedding collection.

    The output is created atomically and never overwrites an existing directory.
    Any invalid key, absent file, wrong shape, or non-finite value aborts creation.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite integrity manifest directory: {output_dir}")
    index = _read_keys(embedding_index_path, label="Embedding index")
    source = _read_keys(source_manifest_path, label="Source manifest")
    if POOLED_PATH_COLUMN not in index.columns:
        raise ValueError(f"Embedding index is missing `{POOLED_PATH_COLUMN}`.")
    if index[POOLED_PATH_COLUMN].isna().any():
        raise ValueError("Embedding index contains missing pooled embedding paths.")
    validate_index_keys(index, source)

    records: list[dict[str, Any]] = []
    for row in index.sort_values(list(KEY_COLUMNS)).itertuples(index=False):
        path = Path(str(getattr(row, POOLED_PATH_COLUMN)))
        record = {column: str(getattr(row, column)) for column in KEY_COLUMNS}
        record.update(inspect_pooled_embedding(path, expected_dimension=expected_dimension))
        records.append(record)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    try:
        records_path = temporary / "pooled_embeddings.csv"
        pd.DataFrame(records).to_csv(records_path, index=False)
        summary = {
            "schema_version": "1.0",
            "status": "valid",
            "expected_pooled_shape": [int(expected_dimension)],
            "rows": len(records),
            "participants": int(index["participant_id"].nunique()),
            "sessions": int(index[["participant_id", "session_id"]].drop_duplicates().shape[0]),
            "embedding_index": {
                "path": str(embedding_index_path.resolve()),
                "sha256": sha256_file(embedding_index_path),
            },
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": sha256_file(source_manifest_path),
            },
            "pooled_file_manifest": {
                "filename": records_path.name,
                "sha256": sha256_file(records_path),
            },
            "checks": {
                "keys_complete_unique_and_exactly_matched": True,
                "all_files_present": True,
                "all_shapes_valid": True,
                "all_values_finite": True,
                "all_files_hashed_sha256": True,
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir
