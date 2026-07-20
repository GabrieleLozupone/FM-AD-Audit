from __future__ import annotations

import os
import random
import subprocess
from typing import Any

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a project dependency, but keep scripts robust.
    tqdm = None


def list_gpus_pci_order() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        pci_bus_id, cuda_index, name, memory_total_mb = parts
        rows.append(
            {
                "pci_bus_id": pci_bus_id,
                "cuda_index": int(cuda_index),
                "name": name,
                "memory_total_mb": int(float(memory_total_mb)),
                "memory_mb": int(float(memory_total_mb)),
            }
        )
    return sorted(rows, key=lambda row: row["pci_bus_id"])


def print_gpus() -> None:
    rows = list_gpus_pci_order()
    if not rows:
        print("No GPUs reported by nvidia-smi.")
        return
    for ordered_index, row in enumerate(rows):
        print(
            f"{ordered_index}: CUDA_VISIBLE_DEVICES={row['cuda_index']} "
            f"pci_bus_id={row['pci_bus_id']} name={row['name']} memory_mb={row['memory_mb']}"
        )


def set_gpu_visibility(gpu_index: int | None) -> None:
    if gpu_index is None:
        return
    rows = list_gpus_pci_order()
    if not rows:
        raise RuntimeError("`--gpu-index` was provided, but nvidia-smi did not report any GPUs.")
    if gpu_index < 0 or gpu_index >= len(rows):
        raise ValueError(f"`--gpu-index` must be in [0, {len(rows) - 1}], got {gpu_index}.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rows[gpu_index]["cuda_index"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None, gpu_index: int | None = None) -> torch.device:
    if gpu_index is not None:
        set_gpu_visibility(gpu_index)
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-index was provided, but CUDA is not available.")
        return torch.device("cuda:0")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_device_name(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_logging(verbosity: int) -> None:
    import logging

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def epoch_iterator(max_epochs: int, description: str, enabled: bool):
    iterator = range(1, max_epochs + 1)
    if enabled and tqdm is not None:
        return tqdm(iterator, desc=description, unit="epoch")
    return iterator
