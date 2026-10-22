#!/usr/bin/env python3
"""Extract frozen tri-planar DINOv3 ViT-B/16 representations for the locked image list (protocol 0.6.0).

Per image: CLS token of every slice in the fixed MNI brain range of each plane,
mean over slices within a plane, equal-weight mean over axial, coronal and
sagittal (primary, 768-d). Also saved but not analysed: the per-plane CLS means
(3 x 768) and the patch-token mean with the same aggregation (768-d).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.encoders.common import (  # noqa: E402
    build_index,
    configure_gpu,
    load_encoder_spec,
    load_image_list,
    require_sha256,
    runtime_metadata,
    set_full_precision,
    write_summary,
)

SUFFIXES = {
    "dinov3": "dinov3_pooled",
    "dinov3_planes_cls": "dinov3_planes_cls",
    "dinov3_patchmean": "dinov3_patchmean",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image-list-path",
        type=Path,
        default=PROJECT_ROOT / "results/paper_neurojepa_feature_audit/locked_inputs/embedding_index.csv",
        help="Locked image list; read only (keys and source_image_path).",
    )
    parser.add_argument("--image-path-column", default="source_image_path")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/fm_encoders")
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (PCI bus order). Default: 0.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slice-batch-size", type=int, default=128, help="Slices per forward pass.")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending images (testing).")
    return parser.parse_args(argv)


def load_volume(path: str):
    import nibabel as nib
    import torch

    from neurojepa_adni_audit.encoders.dinov3 import require_mni_grid, scale_volume

    image = nib.load(path)
    require_mni_grid(image.shape, image.affine)
    return torch.from_numpy(scale_volume(image.get_fdata(dtype="float32")))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_gpu(args.gpu if args.device.startswith("cuda") else None)

    import torch

    from neurojepa_adni_audit.encoders import dinov3
    from neurojepa_adni_audit.encoders.extraction import pending_rows, run_extraction

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    set_full_precision()
    spec = load_encoder_spec(args.protocol_path, "dinov3")
    if spec.get("hf_model_id") != dinov3.MODEL_ID or not spec.get("hf_revision"):
        raise SystemExit("The protocol must pin the DINOv3 model id and revision.")
    recorded_ranges = spec.get("slice_rule", {}).get("slice_indices")
    if recorded_ranges is not None and {k: tuple(v) for k, v in recorded_ranges.items()} != dinov3.SLICE_RANGES:
        raise SystemExit("Slice ranges in the protocol differ from the code constants.")

    rows = load_image_list(args.image_list_path, args.image_path_column)
    embeddings_dir = args.output_root / "dinov3" / "embeddings"
    todo = pending_rows(rows, embeddings_dir, SUFFIXES)
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(
        f"DINOv3: {len(rows)} images in list, {len(todo)} to extract into {embeddings_dir}; "
        f"{dinov3.n_slices()} slices per volume {dinov3.SLICE_RANGES}",
        flush=True,
    )

    model, weights_path = dinov3.load_model(spec["hf_revision"], args.device)
    weights_sha256 = require_sha256(weights_path, spec.get("checkpoint_sha256"), "DINOv3 weights")

    def encode_fn(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [dinov3.encode_volume(model, volume.to(args.device), slice_batch_size=args.slice_batch_size)
                   for volume in batch]
        return {name: torch.stack([output[name] for output in outputs]) for name in SUFFIXES}

    failures = run_extraction(
        rows=todo,
        output_dir=embeddings_dir,
        suffixes=SUFFIXES,
        load_fn=load_volume,
        encode_fn=encode_fn,
        batch_size=1,
        num_workers=args.num_workers,
        progress_label="DINOv3",
    )

    remaining = pending_rows(rows, embeddings_dir, SUFFIXES)
    complete = rows[~rows["image_id"].isin(remaining["image_id"])]
    index = build_index(
        complete, output_dir=embeddings_dir, pooled_suffix=SUFFIXES["dinov3"], encoder="dinov3",
        representation="dinov3: per-slice CLS, slice mean per plane, equal-weight plane mean",
        extra_suffixes={
            "planes_cls_path": SUFFIXES["dinov3_planes_cls"],
            "patchmean_embedding_path": SUFFIXES["dinov3_patchmean"],
        },
    )
    index_path = args.output_root / "dinov3" / "embedding_index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(index_path, index=False)
    write_summary(index_path.with_name("embedding_summary.json"), {
        "encoder": "dinov3",
        "representation": "dinov3",
        "status": "complete" if remaining.empty else "incomplete",
        "n_images_in_list": int(len(rows)),
        "n_embeddings_indexed": int(len(index)),
        "n_missing": int(len(remaining)),
        "failures_this_run": failures,
        "image_list_path": str(args.image_list_path.resolve()),
        "embeddings_dir": str(embeddings_dir.resolve()),
        "hf_model_id": dinov3.MODEL_ID,
        "hf_revision": spec["hf_revision"],
        "weights_path": str(weights_path),
        "weights_sha256": weights_sha256,
        "slice_ranges_inclusive": dinov3.SLICE_RANGES,
        "slices_per_volume": dinov3.n_slices(),
        "canvas_size": dinov3.CANVAS_SIZE,
        "input_size": dinov3.INPUT_SIZE,
        "plane_order_in_planes_cls": list(dinov3.PLANES),
        "preprocessing": spec.get("preprocessing_operations_in_order"),
        "protocol_path": str(args.protocol_path.resolve()),
        "runtime": runtime_metadata(),
        "args": {key: str(value) for key, value in vars(args).items()},
    })
    print(f"Wrote {index_path} ({len(index)}/{len(rows)} images)", flush=True)
    if failures:
        print(f"{len(failures)} failure(s); see embedding_summary.json. Fix and rerun; nothing is dropped.", flush=True)
    return 0 if remaining.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
