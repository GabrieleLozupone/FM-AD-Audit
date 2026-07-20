from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        token_column: str,
        covariates: np.ndarray | None,
        labels: np.ndarray,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.token_column = token_column
        self.covariates = covariates
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        token_path = Path(str(self.rows.at[index, self.token_column]))
        tokens = np.load(token_path).astype(np.float32, copy=False)
        item = {
            "tokens": torch.from_numpy(tokens),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
        }
        if self.covariates is not None:
            item["covariates"] = torch.from_numpy(self.covariates[index].astype(np.float32, copy=False))
        return item


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        y_cls: np.ndarray,
        cls_mask: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32, copy=False))
        self.reg_mask = torch.from_numpy(reg_mask.astype(np.float32, copy=False))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32, copy=False))
        self.cls_mask = torch.from_numpy(cls_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y_reg[index], self.reg_mask[index], self.y_cls[index], self.cls_mask[index]


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        observed: np.ndarray,
        delta_t: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        y_cls: np.ndarray,
        cls_mask: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.observed = torch.from_numpy(observed.astype(bool, copy=False))
        self.delta_t = torch.from_numpy(delta_t.astype(np.float32, copy=False))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32, copy=False))
        self.reg_mask = torch.from_numpy(reg_mask.astype(np.float32, copy=False))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32, copy=False))
        self.cls_mask = torch.from_numpy(cls_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.observed[index],
            self.delta_t[index],
            self.y_reg[index],
            self.reg_mask[index],
            self.y_cls[index],
            self.cls_mask[index],
        )


class JepaWorldDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        y_cls: np.ndarray,
        cls_mask: np.ndarray,
        current_z: np.ndarray,
        delta_z: np.ndarray,
        z_mask: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32, copy=False))
        self.reg_mask = torch.from_numpy(reg_mask.astype(np.float32, copy=False))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32, copy=False))
        self.cls_mask = torch.from_numpy(cls_mask.astype(np.float32, copy=False))
        self.current_z = torch.from_numpy(current_z.astype(np.float32, copy=False))
        self.delta_z = torch.from_numpy(delta_z.astype(np.float32, copy=False))
        self.z_mask = torch.from_numpy(z_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.y_reg[index],
            self.reg_mask[index],
            self.y_cls[index],
            self.cls_mask[index],
            self.current_z[index],
            self.delta_z[index],
            self.z_mask[index],
        )


class SequenceJepaWorldDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        observed: np.ndarray,
        delta_t: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        y_cls: np.ndarray,
        cls_mask: np.ndarray,
        current_z: np.ndarray,
        delta_z: np.ndarray,
        z_mask: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.observed = torch.from_numpy(observed.astype(bool, copy=False))
        self.delta_t = torch.from_numpy(delta_t.astype(np.float32, copy=False))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32, copy=False))
        self.reg_mask = torch.from_numpy(reg_mask.astype(np.float32, copy=False))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32, copy=False))
        self.cls_mask = torch.from_numpy(cls_mask.astype(np.float32, copy=False))
        self.current_z = torch.from_numpy(current_z.astype(np.float32, copy=False))
        self.delta_z = torch.from_numpy(delta_z.astype(np.float32, copy=False))
        self.z_mask = torch.from_numpy(z_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.observed[index],
            self.delta_t[index],
            self.y_reg[index],
            self.reg_mask[index],
            self.y_cls[index],
            self.cls_mask[index],
            self.current_z[index],
            self.delta_z[index],
            self.z_mask[index],
        )


class BrainWMStateDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        observed: np.ndarray,
        delta_t: np.ndarray,
        y_reg: np.ndarray,
        reg_mask: np.ndarray,
        y_cls: np.ndarray,
        cls_mask: np.ndarray,
        current_z: np.ndarray,
        z_target: np.ndarray,
        z_mask: np.ndarray,
        y_align: np.ndarray,
        align_mask: np.ndarray,
    ) -> None:
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.observed = torch.from_numpy(observed.astype(bool, copy=False))
        self.delta_t = torch.from_numpy(delta_t.astype(np.float32, copy=False))
        self.y_reg = torch.from_numpy(y_reg.astype(np.float32, copy=False))
        self.reg_mask = torch.from_numpy(reg_mask.astype(np.float32, copy=False))
        self.y_cls = torch.from_numpy(y_cls.astype(np.float32, copy=False))
        self.cls_mask = torch.from_numpy(cls_mask.astype(np.float32, copy=False))
        self.current_z = torch.from_numpy(current_z.astype(np.float32, copy=False))
        self.z_target = torch.from_numpy(z_target.astype(np.float32, copy=False))
        self.z_mask = torch.from_numpy(z_mask.astype(np.float32, copy=False))
        self.y_align = torch.from_numpy(y_align.astype(np.float32, copy=False))
        self.align_mask = torch.from_numpy(align_mask.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[index],
            self.observed[index],
            self.delta_t[index],
            self.y_reg[index],
            self.reg_mask[index],
            self.y_cls[index],
            self.cls_mask[index],
            self.current_z[index],
            self.z_target[index],
            self.z_mask[index],
            self.y_align[index],
            self.align_mask[index],
        )
