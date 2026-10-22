#!/usr/bin/env python3
"""Quantify how background floating-point residue affects each encoder's model input.

For each scan of the 25-scan preprocessing sample, builds the Neuro-JEPA,
DINOv3 and BrainIAC model inputs from processed_t1_path as-is and after the
BrainIAC residue rule (voxels below 0.1% of the 99th-percentile nonzero
intensity set to 0), and reports the relative change. Image-only: no model,
GPU, embedding or outcome is used. Supports the BrainIAC-only residue rule
(protocol 0.6.0 background_residue_handling).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-manifest-path", type=Path,
                        default=PROJECT_ROOT / "output/preprocessing_volume_check_25/sample_manifest.csv")
    parser.add_argument("--image-path-column", default="processed_t1_path")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/fm_encoders/qc/background_residue")
    return parser.parse_args(argv)


def neurojepa_transform():
    """scripts/extract_neurojepa_embeddings.py build_preprocess_transform, array form.

    The pinned neurojepa remove_nan is replaced by torch.nan_to_num (no NaNs occur
    in these images); every other operation and parameter is identical.
    """
    import monai.transforms as T
    import torch

    return T.Compose([
        T.EnsureChannelFirst(channel_dim="no_channel"),
        T.Lambda(lambda x: torch.nan_to_num(torch.as_tensor(x))),
        T.Orientation(axcodes="RAS"),
        T.Spacing(pixdim=(1.0, 1.0, 1.0), mode=5),
        T.CropForeground(select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
        T.ResizeWithPadOrCrop(spatial_size=[180, 216, 180], mode="edge"),
        T.Resize(spatial_size=[100, 120, 100]),
        T.CenterSpatialCrop(roi_size=[96, 108, 96]),
        T.ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
    ])


def crop_box(volume, margin: int = 4) -> tuple[list[int], list[float]]:
    """Size and centre (voxels) of the CropForeground box (intensity > 0, margin 4, clipped)."""
    import numpy as np

    index = np.nonzero(volume > 0)
    low = [max(0, int(axis.min()) - margin) for axis in index]
    high = [min(size, int(axis.max()) + margin + 1) for axis, size in zip(index, volume.shape)]
    return [b - a for a, b in zip(low, high)], [(a + b) / 2 for a, b in zip(low, high)]


def relative_change(cleaned, as_is) -> float:
    import numpy as np

    cleaned, as_is = np.asarray(cleaned, dtype=np.float64), np.asarray(as_is, dtype=np.float64)
    return float(np.linalg.norm(cleaned - as_is) / np.linalg.norm(as_is))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import monai
    import monai.transforms as T
    import nibabel as nib
    import numpy as np
    import pandas as pd
    import torch

    from neurojepa_adni_audit.encoders import brainiac, dinov3

    brainiac_after_cleaning = T.Compose([
        T.EnsureChannelFirst(channel_dim="no_channel"),
        T.Resize(spatial_size=brainiac.IMAGE_SIZE, mode="trilinear"),
        T.NormalizeIntensity(nonzero=True, channel_wise=True),
    ])
    neurojepa = neurojepa_transform()
    sample = pd.read_csv(args.sample_manifest_path, low_memory=False)
    rows = []
    for record in sample.itertuples(index=False):
        path = getattr(record, args.image_path_column)
        image = nib.load(path)
        as_is = image.get_fdata(dtype="float32")
        cleaned = brainiac.zero_background_residue(torch.from_numpy(as_is)).numpy()
        positive = as_is[as_is > 0]
        threshold = brainiac.RESIDUE_FRACTION * float(np.percentile(positive, brainiac.RESIDUE_PERCENTILE))

        b_clean, b_as_is = brainiac_after_cleaning(cleaned), brainiac_after_cleaning(as_is)
        resized_as_is = T.Resize(spatial_size=brainiac.IMAGE_SIZE, mode="trilinear")(torch.from_numpy(as_is)[None])

        def njin(volume):
            return neurojepa(monai.data.MetaTensor(torch.from_numpy(volume), affine=torch.as_tensor(image.affine)))

        n_clean, n_as_is = njin(cleaned), njin(as_is)
        box_as_is, centre_as_is = crop_box(as_is)
        box_clean, centre_clean = crop_box(cleaned)

        dinov3.require_mni_grid(image.shape, image.affine)
        d_clean = torch.from_numpy(dinov3.scale_volume(cleaned))
        d_as_is = torch.from_numpy(dinov3.scale_volume(as_is))
        inputs_clean = torch.cat([dinov3.to_model_input(dinov3.plane_slices(d_clean, plane)) for plane in dinov3.PLANES])
        inputs_as_is = torch.cat([dinov3.to_model_input(dinov3.plane_slices(d_as_is, plane)) for plane in dinov3.PLANES])

        rows.append({
            "image_id": record.image_id,
            "nonzero_voxels": int((as_is != 0).sum()),
            "residue_voxels_below_threshold": int(((as_is != 0) & (as_is < threshold)).sum()),
            "threshold": threshold,
            "p99_nonzero_intensity": float(np.percentile(positive, brainiac.RESIDUE_PERCENTILE)),
            "brainiac_foreground_fraction_as_is": float((resized_as_is != 0).float().mean()),
            "brainiac_foreground_fraction_cleaned": float((b_clean != 0).float().mean()),
            "brainiac_input_relative_change": relative_change(b_clean, b_as_is),
            "dinov3_input_relative_change": relative_change(inputs_clean, inputs_as_is),
            "dinov3_input_max_abs_change": float((inputs_clean - inputs_as_is).abs().max()),
            "neurojepa_input_relative_change": relative_change(n_clean, n_as_is),
            "neurojepa_crop_box_as_is": box_as_is,
            "neurojepa_crop_box_cleaned": box_clean,
            "neurojepa_crop_box_is_whole_volume_as_is": box_as_is == list(as_is.shape),
            # Array axes are x, y, z of the LAS MNI grid at 1 mm.
            "neurojepa_crop_centre_shift_x_mm": centre_clean[0] - centre_as_is[0],
            "neurojepa_crop_centre_shift_y_mm": centre_clean[1] - centre_as_is[1],
            "neurojepa_crop_centre_shift_z_mm": centre_clean[2] - centre_as_is[2],
        })
        print(f"{len(rows):2d}/{len(sample)} {record.image_id}", flush=True)

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "per_scan.csv", index=False)

    def describe(column: str) -> dict:
        values = table[column].astype(float)
        return {"median": float(values.median()), "min": float(values.min()), "max": float(values.max()),
                "mean": float(values.mean()), "sd": float(values.std())}

    summary = {
        "n_scans": int(len(table)),
        "rule": f"voxels < {brainiac.RESIDUE_FRACTION:g} x {brainiac.RESIDUE_PERCENTILE:g}th-percentile nonzero intensity set to 0",
        "residue_fraction_of_nonzero_voxels": describe_ratio(table),
        "brainiac_foreground_fraction_as_is": describe("brainiac_foreground_fraction_as_is"),
        "brainiac_foreground_fraction_cleaned": describe("brainiac_foreground_fraction_cleaned"),
        "brainiac_input_relative_change": describe("brainiac_input_relative_change"),
        "dinov3_input_relative_change": describe("dinov3_input_relative_change"),
        "dinov3_input_max_abs_change": describe("dinov3_input_max_abs_change"),
        "neurojepa_input_relative_change": describe("neurojepa_input_relative_change"),
        "neurojepa_crop_whole_volume_as_is": int(table["neurojepa_crop_box_is_whole_volume_as_is"].sum()),
        "neurojepa_crop_centre_shift_mm": {axis: describe(f"neurojepa_crop_centre_shift_{axis}_mm") for axis in "xyz"},
        "notes": [
            "Relative change = ||input(cleaned) - input(as-is)|| / ||input(as-is)|| of the exact model input.",
            "For Neuro-JEPA the change is geometric (field-of-view position), not an intensity change.",
            "Neuro-JEPA transform reproduced with MONAI; neurojepa.remove_nan replaced by nan_to_num.",
            "No model, embedding or outcome was used.",
        ],
        "runtime": {"monai": monai.__version__, "torch": torch.__version__, "nibabel": nib.__version__},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"notes", "runtime"}}, indent=2))
    return 0


def describe_ratio(table) -> dict:
    ratio = table["residue_voxels_below_threshold"] / table["nonzero_voxels"]
    return {"median": float(ratio.median()), "min": float(ratio.min()), "max": float(ratio.max())}


if __name__ == "__main__":
    raise SystemExit(main())
