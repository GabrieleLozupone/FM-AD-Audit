#!/usr/bin/env python3
"""Outcome-blind input and representation QC for the protocol 0.6.0 encoders.

Uses the 25-scan preprocessing sample plus each participant's next session from
the locked image list. Writes montages of the exact model inputs, input
statistics, and representation checks (within- versus between-participant
similarity, nearest-neighbour session matching). No outcome is read, and
nothing is written to the production embedding directories.
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

from neurojepa_adni_audit.encoders.common import (  # noqa: E402
    KEY_COLUMNS,
    configure_gpu,
    load_encoder_spec,
    require_sha256,
    runtime_metadata,
    set_full_precision,
)

LOCKED = PROJECT_ROOT / "results/paper_neurojepa_feature_audit/locked_inputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", choices=["brainiac", "dinov3"], required=True)
    parser.add_argument("--sample-manifest-path", type=Path,
                        default=PROJECT_ROOT / "output/preprocessing_volume_check_25/sample_manifest.csv")
    parser.add_argument("--locked-inputs-dir", type=Path, default=LOCKED,
                        help="Locked nested-CV inputs (embedding_index.csv, manifest.csv); read only.")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "downloads/brainiac/BrainIAC.ckpt")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: results/fm_encoders/qc/<encoder>_inputs")
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (PCI bus order). Default: 0.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--montage-images", type=int, default=6)
    return parser.parse_args(argv)


def qc_sample(sample_manifest_path: Path, locked_dir: Path = LOCKED):
    """Sample sessions plus each sampled participant's next locked session (outcome-blind)."""
    import pandas as pd

    sample = pd.read_csv(sample_manifest_path, dtype={c: "string" for c in KEY_COLUMNS}, low_memory=False)
    images = pd.read_csv(locked_dir / "embedding_index.csv", dtype={c: "string" for c in KEY_COLUMNS}, low_memory=False)
    months = pd.read_csv(locked_dir / "manifest.csv", usecols=["image_id", "session_month"], dtype={"image_id": "string"})
    images = images[[*KEY_COLUMNS, "source_image_path"]].merge(months, on="image_id", how="left")
    chosen = images[images["image_id"].isin(sample["image_id"])].copy()
    chosen["qc_role"] = "sample"
    later = []
    for row in chosen.itertuples(index=False):
        candidates = images[(images["participant_id"] == row.participant_id) & (images["session_month"] > row.session_month)]
        if not candidates.empty:
            later.append(candidates.sort_values("session_month").iloc[[0]])
    if later:
        followups = pd.concat(later)
        followups["qc_role"] = "next_session"
        chosen = pd.concat([chosen, followups])
    if len(chosen[chosen["qc_role"] == "sample"]) != len(sample):
        raise ValueError("Some sample images are not in the locked image list.")
    return chosen.sort_values(["participant_id", "session_month"]).reset_index(drop=True)


def similarity_report(vectors, participants) -> dict:
    """Raw and sample-centred cosine similarity; nearest-neighbour participant matching."""
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float64)
    participants = np.asarray(participants)

    def cosine_matrix(x):
        normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        return normalized @ normalized.T

    same = participants[:, None] == participants[None, :]
    off_diagonal = ~np.eye(len(vectors), dtype=bool)
    report = {}
    for label, matrix in (("raw", cosine_matrix(vectors)), ("centred", cosine_matrix(vectors - vectors.mean(0)))):
        report[f"{label}_mean_within_participant_cosine"] = float(matrix[same & off_diagonal].mean()) if (same & off_diagonal).any() else None
        report[f"{label}_mean_between_participant_cosine"] = float(matrix[~same].mean())
    centred = cosine_matrix(vectors - vectors.mean(0))
    np.fill_diagonal(centred, -np.inf)
    has_pair = (same & off_diagonal).any(axis=1)
    nearest_same = same[np.arange(len(vectors)), centred.argmax(axis=1)]
    report["nearest_neighbour_same_participant_rate"] = float(nearest_same[has_pair].mean()) if has_pair.any() else None
    report["n_images_with_same_participant_pair"] = int(has_pair.sum())
    std = vectors.std(axis=0)
    report["n_dimensions"] = int(vectors.shape[1])
    report["n_near_constant_dimensions"] = int((std < 1e-6 * max(std.max(), 1e-12)).sum())
    report["all_finite"] = bool(np.isfinite(vectors).all())
    return report


def save_montage(panels: list[tuple[str, list]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = max(len(images) for _, images in panels)
    figure, axes = plt.subplots(len(panels), columns, figsize=(2.2 * columns, 2.3 * len(panels)), squeeze=False)
    for row, (label, images) in enumerate(panels):
        for column in range(columns):
            axis = axes[row][column]
            axis.axis("off")
            if column < len(images):
                axis.imshow(images[column], cmap="gray", vmin=0, vmax=1)
        axes[row][0].set_title(label, fontsize=7, loc="left")
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def run_brainiac(args, rows, spec) -> tuple[dict, list, list]:
    import numpy as np
    import torch

    from neurojepa_adni_audit.encoders.brainiac import build_transform, load_backbone, pool_tokens

    require_sha256(args.checkpoint_path, spec.get("checkpoint_sha256"), "BrainIAC checkpoint")
    model, _ = load_backbone(args.checkpoint_path)
    model = model.to(args.device)
    transform = build_transform()
    vectors = {"brainiac": [], "brainiac_token0": []}
    stats, panels = [], []
    with torch.inference_mode():
        for position, row in enumerate(rows.itertuples(index=False)):
            image = torch.as_tensor(transform({"image": row.source_image_path})["image"], dtype=torch.float32)
            volume = image[0].numpy()
            stats.append({"image_id": row.image_id, "shape": list(image.shape), "min": float(volume.min()),
                          "max": float(volume.max()), "mean": float(volume.mean()), "std": float(volume.std()),
                          "fraction_nonzero": float((volume != 0).mean())})
            for name, vector in pool_tokens(model(image.unsqueeze(0).to(args.device))[0]).items():
                vectors[name].append(vector[0].double().cpu().numpy())
            if position < args.montage_images:
                # Display only: model input as given to BrainIAC (no reorientation), mid slices.
                shown = (volume - volume.min()) / max(volume.max() - volume.min(), 1e-6)
                mid = [s // 2 for s in shown.shape]
                panels.append((f"{row.image_id} (BrainIAC 96^3 input; display flip only)", [
                    np.flipud(shown[:, :, mid[2]].T), np.flipud(shown[:, mid[1], :].T), np.flipud(shown[mid[0], :, :].T),
                ]))
    return vectors, stats, panels


def run_dinov3(args, rows, spec) -> tuple[dict, list, list]:
    import nibabel as nib
    import numpy as np
    import torch

    from neurojepa_adni_audit.encoders import dinov3

    model, weights_path = dinov3.load_model(spec["hf_revision"], args.device)
    require_sha256(weights_path, spec.get("checkpoint_sha256"), "DINOv3 weights")
    mean = torch.tensor(dinov3.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(dinov3.IMAGENET_STD).view(3, 1, 1)
    vectors = {"dinov3": [], "dinov3_patchmean": []}
    stats, panels = [], []
    for position, row in enumerate(rows.itertuples(index=False)):
        image = nib.load(row.source_image_path)
        dinov3.require_mni_grid(image.shape, image.affine)
        volume = torch.from_numpy(dinov3.scale_volume(image.get_fdata(dtype="float32")))
        stats.append({"image_id": row.image_id, "shape": list(volume.shape), "min": float(volume.min()),
                      "max": float(volume.max()), "mean": float(volume.mean()), "std": float(volume.std()),
                      "fraction_nonzero": float((volume != 0).float().mean())})
        outputs = dinov3.encode_volume(model, volume.to(args.device))
        for name in vectors:
            vectors[name].append(outputs[name].double().cpu().numpy())
        if position < args.montage_images:
            for plane in dinov3.PLANES:
                slices = dinov3.plane_slices(volume, plane)
                picks = np.linspace(0, slices.shape[0] - 1, 6).round().astype(int)
                model_inputs = dinov3.to_model_input(slices[picks])
                shown = [(model_input * std + mean)[0].clamp(0, 1).numpy() for model_input in model_inputs]
                start = dinov3.SLICE_RANGES[plane][0]
                panels.append((f"{row.image_id} {plane} slices {[int(start + p) for p in picks]}", shown))
    return vectors, stats, panels


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_gpu(args.gpu if args.device.startswith("cuda") else None)

    import pandas as pd
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    set_full_precision()
    output_dir = args.output_dir or PROJECT_ROOT / "results/fm_encoders/qc" / f"{args.encoder}_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_encoder_spec(args.protocol_path, args.encoder)
    rows = qc_sample(args.sample_manifest_path, args.locked_inputs_dir)
    print(f"QC {args.encoder}: {len(rows)} images ({(rows['qc_role'] == 'next_session').sum()} next sessions)", flush=True)

    runner = run_brainiac if args.encoder == "brainiac" else run_dinov3
    vectors, stats, panels = runner(args, rows, spec)

    pd.DataFrame(stats).to_csv(output_dir / "input_statistics.csv", index=False)
    rows.to_csv(output_dir / "qc_images.csv", index=False)
    for chunk in range(0, len(panels), 3 if args.encoder == "dinov3" else 6):
        part = panels[chunk:chunk + (3 if args.encoder == "dinov3" else 6)]
        save_montage(part, output_dir / f"montage_{chunk // (3 if args.encoder == 'dinov3' else 6):02d}.png",
                     f"{args.encoder} model inputs (anterior/superior at top)")
    summary = {
        "encoder": args.encoder,
        "n_images": int(len(rows)),
        "n_participants": int(rows["participant_id"].nunique()),
        "representations": {name: similarity_report(values, rows["participant_id"]) for name, values in vectors.items()},
        "outcomes_used": False,
        "runtime": runtime_metadata(),
    }
    (output_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["representations"], indent=2))
    print(f"QC outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
