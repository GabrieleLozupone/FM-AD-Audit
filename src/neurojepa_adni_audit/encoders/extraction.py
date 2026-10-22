"""Shared resumable extraction loop for the protocol 0.6.0 encoders."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from neurojepa_adni_audit.encoders.common import KEY_COLUMNS, output_path, outputs_complete, save_array_atomic


class _ImageDataset(Dataset):
    """Loads one model input per row; load errors are returned, not raised, so they are recorded."""

    def __init__(self, rows: pd.DataFrame, load_fn: Callable[[str], torch.Tensor]):
        self.rows = rows.reset_index(drop=True)
        self.load_fn = load_fn

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = str(self.rows.at[index, "source_image_path"])
        try:
            return {"index": index, "image": self.load_fn(path), "error": None}
        except Exception as exc:  # noqa: BLE001 - every failure is recorded with its message
            return {"index": index, "image": None, "error": f"load: {type(exc).__name__}: {exc}"}


def _collate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


def expected_outputs(keys: tuple[str, str, str], output_dir: Path, suffixes: dict[str, str]) -> dict[str, Path]:
    """Output paths for one (participant_id, session_id, image_id)."""
    return {name: output_path(output_dir, *keys, suffix) for name, suffix in suffixes.items()}


def pending_rows(rows: pd.DataFrame, output_dir: Path, suffixes: dict[str, str]) -> pd.DataFrame:
    complete = [
        outputs_complete(list(expected_outputs(tuple(map(str, keys)), output_dir, suffixes).values()))
        for keys in rows[list(KEY_COLUMNS)].itertuples(index=False)
    ]
    return rows[[not done for done in complete]].reset_index(drop=True)


def run_extraction(
    *,
    rows: pd.DataFrame,
    output_dir: Path,
    suffixes: dict[str, str],
    load_fn: Callable[[str], torch.Tensor],
    encode_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    batch_size: int,
    num_workers: int,
    progress_label: str,
) -> list[dict[str, str]]:
    """Encode `rows` and save one float32 .npy per output name; returns the failures.

    `encode_fn` receives a stacked batch on the CPU and returns, for every name in
    `suffixes`, a tensor whose first dimension is the batch.
    """
    from tqdm import tqdm

    loader = DataLoader(
        _ImageDataset(rows, load_fn),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        persistent_workers=False,
    )
    failures: list[dict[str, str]] = []

    def record(index: int, error: str) -> None:
        row = rows.iloc[index]
        failures.append({**{column: str(row[column]) for column in KEY_COLUMNS}, "error": error})

    def save(index: int, outputs: dict[str, torch.Tensor], position: int) -> None:
        keys = tuple(str(rows.at[index, column]) for column in KEY_COLUMNS)
        paths = expected_outputs(keys, output_dir, suffixes)
        for name, path in paths.items():
            save_array_atomic(path, outputs[name][position].detach().cpu().numpy())

    progress = tqdm(total=len(rows), desc=progress_label, unit="image", dynamic_ncols=True)
    for batch in loader:
        loaded = []
        for item in batch:
            if item["error"] is not None:
                record(item["index"], item["error"])
            else:
                loaded.append(item)
        if loaded:
            try:
                outputs = encode_fn(torch.stack([item["image"] for item in loaded]))
                for position, item in enumerate(loaded):
                    save(item["index"], outputs, position)
            except Exception:  # noqa: BLE001 - isolate the failing image(s)
                for item in loaded:
                    try:
                        save(item["index"], encode_fn(item["image"].unsqueeze(0)), 0)
                    except Exception as exc:  # noqa: BLE001
                        record(item["index"], f"encode: {type(exc).__name__}: {exc}")
        progress.update(len(batch))
        progress.set_postfix(fail=len(failures), refresh=False)
    progress.close()
    return failures
