"""DINOv3 ViT-B/16 applied tri-planarly to MNI-space T1 volumes (protocol 0.6.0)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
N_REGISTER_TOKENS = 4
INPUT_SIZE = 224
PLANES = ("axial", "coronal", "sagittal")
# Volume array axes are (x, y, z) in FSL MNI152 1 mm LAS space.
PLANE_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}
MNI_SHAPE = (182, 218, 182)
MNI_AFFINE = np.array([
    [-1.0, 0.0, 0.0, 90.0],
    [0.0, 1.0, 0.0, -126.0],
    [0.0, 0.0, 1.0, -72.0],
    [0.0, 0.0, 0.0, 1.0],
])
# All planes share one square canvas so every slice has the same mm per pixel.
CANVAS_SIZE = max(MNI_SHAPE)
SLICE_AREA_FRACTION = 0.10
# Inclusive ranges from MNI152_T1_1mm_brain_mask.nii.gz: slices whose brain area
# is at least 10% of that plane's largest slice (443 slices per volume).
SLICE_RANGES = {"axial": (13, 148), "coronal": (26, 195), "sagittal": (21, 157)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def slice_ranges_from_mask(mask: np.ndarray, fraction: float = SLICE_AREA_FRACTION) -> dict[str, tuple[int, int]]:
    """Fixed inclusive slice range per plane from a binary template brain mask."""
    brain = np.asarray(mask) > 0
    ranges = {}
    for plane in PLANES:
        axis = PLANE_AXIS[plane]
        area = brain.sum(axis=tuple(i for i in range(3) if i != axis))
        selected = np.flatnonzero(area >= fraction * area.max())
        if selected.size == 0 or np.any(np.diff(selected) != 1):
            raise ValueError(f"The {plane} slice range is empty or not contiguous.")
        ranges[plane] = (int(selected[0]), int(selected[-1]))
    return ranges


def n_slices(ranges: dict[str, tuple[int, int]] = SLICE_RANGES) -> int:
    return sum(stop - start + 1 for start, stop in ranges.values())


def require_mni_grid(shape: tuple[int, ...], affine: np.ndarray) -> None:
    """The fixed slice indices are only valid on the FSL MNI152 1 mm grid."""
    if tuple(shape) != MNI_SHAPE or not np.allclose(affine, MNI_AFFINE, atol=1e-3):
        raise ValueError(f"Image is not on the MNI152 1 mm grid: shape={tuple(shape)}")


def scale_volume(volume: np.ndarray) -> np.ndarray:
    """Clip to the 0.5/99.5 percentiles of nonzero voxels and scale to [0, 1]; background stays 0."""
    data = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    foreground = data[data != 0]
    if foreground.size == 0:
        raise ValueError("Volume has no nonzero voxels.")
    low, high = np.percentile(foreground, [0.5, 99.5])
    if not high > low:
        raise ValueError("Degenerate intensity range.")
    scaled = (np.clip(data, low, high) - low) / (high - low)
    scaled[data == 0] = 0.0
    return scaled.astype(np.float32)


def plane_slices(volume: torch.Tensor, plane: str, ranges: dict[str, tuple[int, int]] = SLICE_RANGES) -> torch.Tensor:
    """Upright 2D slices (N, H, W) of an (x, y, z) LAS volume: anterior or superior at the top."""
    start, stop = ranges[plane]
    if plane == "axial":
        slices = volume[:, :, start:stop + 1].permute(2, 1, 0)  # (z, y, x)
    elif plane == "coronal":
        slices = volume[:, start:stop + 1, :].permute(1, 2, 0)  # (y, z, x)
    elif plane == "sagittal":
        slices = volume[start:stop + 1, :, :].permute(0, 2, 1)  # (x, z, y)
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return torch.flip(slices, dims=(1,))


def to_model_input(slices: torch.Tensor) -> torch.Tensor:
    """(N, H, W) in [0, 1] -> (N, 3, 224, 224): centred zero-pad to the canvas, bilinear resize, ImageNet normalization."""
    n, height, width = slices.shape
    top, left = (CANVAS_SIZE - height) // 2, (CANVAS_SIZE - width) // 2
    canvas = slices.new_zeros((n, 1, CANVAS_SIZE, CANVAS_SIZE))
    canvas[:, 0, top:top + height, left:left + width] = slices
    resized = F.interpolate(canvas, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, dtype=resized.dtype, device=resized.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=resized.dtype, device=resized.device).view(1, 3, 1, 1)
    return (resized.expand(-1, 3, -1, -1) - mean) / std


def load_model(revision: str, device: str) -> tuple[torch.nn.Module, Path]:
    """Frozen DINOv3 at the pinned revision; also returns the local weights path for hashing."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    model = AutoModel.from_pretrained(MODEL_ID, revision=revision, dtype=torch.float32)
    if int(model.config.num_register_tokens) != N_REGISTER_TOKENS:
        raise ValueError("Unexpected number of DINOv3 register tokens.")
    weights_path = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=revision))
    return model.to(device).eval(), weights_path


@torch.inference_mode()
def encode_volume(
    model: torch.nn.Module,
    volume: torch.Tensor,
    *,
    slice_batch_size: int = 128,
    ranges: dict[str, tuple[int, int]] = SLICE_RANGES,
) -> dict[str, torch.Tensor]:
    """Per-plane mean CLS and mean patch-token vectors for one scaled volume on the model device."""
    plane_cls, plane_patch = [], []
    for plane in PLANES:
        slices = plane_slices(volume, plane, ranges)
        cls_sum = torch.zeros(model.config.hidden_size, dtype=torch.float64, device=volume.device)
        patch_sum = torch.zeros_like(cls_sum)
        for start in range(0, slices.shape[0], slice_batch_size):
            batch = to_model_input(slices[start:start + slice_batch_size])
            output = model(pixel_values=batch)
            cls_sum += output.pooler_output.double().sum(dim=0)
            patch_sum += output.last_hidden_state[:, 1 + N_REGISTER_TOKENS:].double().mean(dim=1).sum(dim=0)
        plane_cls.append(cls_sum / slices.shape[0])
        plane_patch.append(patch_sum / slices.shape[0])
    return pool_planes(torch.stack(plane_cls), torch.stack(plane_patch))


def pool_planes(plane_cls: torch.Tensor, plane_patch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Equal-weight mean over the three planes (rows ordered as PLANES)."""
    if plane_cls.shape[0] != len(PLANES) or plane_patch.shape[0] != len(PLANES):
        raise ValueError("Expected one vector per plane.")
    return {
        "dinov3": plane_cls.mean(dim=0),  # primary: CLS, slice mean per plane, equal-weight plane mean
        "dinov3_planes_cls": plane_cls,  # saved, not analysed: (3, 768) in PLANES order
        "dinov3_patchmean": plane_patch.mean(dim=0),  # saved, not analysed
    }
