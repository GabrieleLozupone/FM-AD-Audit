"""BrainIAC (MONAI 3D ViT-B, SimCLR) frozen representation, upstream commit ba60f45."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


IMAGE_SIZE = (96, 96, 96)
N_PATCH_TOKENS = 216
# Background residue removal (protocol 0.6.0 background_residue_handling): voxels below this fraction of the
# volume's 99th-percentile nonzero intensity are set to exactly 0.
RESIDUE_FRACTION = 1e-3
RESIDUE_PERCENTILE = 99.0
# Output suffixes and the representation each one feeds (protocol 0.6.0).
REPRESENTATIONS = {
    "brainiac": "brainiac_patchmean",  # primary: mean of the 216 patch tokens
    "brainiac_token0": "brainiac_token0",  # secondary: upstream features[0][:, 0]
}


def zero_background_residue(volume: Any) -> Any:
    """Set near-zero floating-point registration residue to exactly 0.

    The shared preprocessed images carry about 1e-45..1 residue outside the brain,
    which BrainIAC's NormalizeIntensityd(nonzero=True) would count as foreground.
    Upstream inputs (HD-BET after registration) have an exactly zero background.
    """
    data = volume if isinstance(volume, torch.Tensor) else torch.as_tensor(np.asarray(volume))
    positive = data[data > 0]
    if positive.numel() == 0:
        raise ValueError("Volume has no positive voxels.")
    threshold = RESIDUE_FRACTION * torch.quantile(positive.double().flatten(), RESIDUE_PERCENTILE / 100.0).item()
    cleaned = data.clone()
    cleaned[cleaned < threshold] = 0
    return cleaned


def build_transform(*, remove_background_residue: bool = True) -> Any:
    """Upstream src/dataset.py get_validation_transform, without augmentation.

    With `remove_background_residue` (the protocol 0.6.0 setting), residue is zeroed
    on the native-resolution volume before upstream resizing and normalization.
    """
    from monai.transforms import Compose, EnsureChannelFirstd, Lambdad, LoadImaged, NormalizeIntensityd, Resized, ToTensord

    cleaning = [Lambdad(keys=["image"], func=zero_background_residue)] if remove_background_residue else []
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        *cleaning,
        Resized(keys=["image"], spatial_size=IMAGE_SIZE, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def load_backbone(checkpoint_path: Path) -> tuple[torch.nn.Module, int]:
    """Upstream ViTBackboneNet backbone; returns the model and the count of absent tensors.

    MONAI >= 1.4 creates per-block cross-attention modules that run only when
    with_cross_attention=True; they are the only tensors allowed to be absent.
    """
    from monai.networks.nets import ViT

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone = {key[len("backbone."):]: value for key, value in state_dict.items() if key.startswith("backbone.")}
    if not backbone:
        raise ValueError("No `backbone.*` tensors found; not the BrainIAC SimCLR checkpoint.")
    model = ViT(
        in_channels=1, img_size=IMAGE_SIZE, patch_size=(16, 16, 16), hidden_size=768,
        mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True,
    )
    result = model.load_state_dict(backbone, strict=False)
    unexpected_missing = [key for key in result.missing_keys if "cross_attn." not in key]
    if result.unexpected_keys or unexpected_missing:
        raise ValueError(f"Checkpoint mismatch: unexpected={result.unexpected_keys} missing={unexpected_missing}")
    if any(block.with_cross_attention for block in model.blocks):
        raise ValueError("BrainIAC cross-attention must be disabled.")
    return model.eval(), len(result.missing_keys)


def pool_tokens(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Both prespecified BrainIAC representations from final-layer tokens (B, 216, 768).

    MONAI ViT(classification=False) has no CLS token, so token 0 is the first patch.
    """
    if tokens.ndim != 3 or tokens.shape[1] != N_PATCH_TOKENS:
        raise ValueError(f"Expected (batch, {N_PATCH_TOKENS}, 768) tokens, got {tuple(tokens.shape)}.")
    return {
        "brainiac": tokens.mean(dim=1),
        "brainiac_token0": tokens[:, 0],
    }
