#!/usr/bin/env python3
"""Extract frozen BrainIAC representations for the locked image list (protocol 0.6.0).

One pass saves both prespecified representations per image:
  brainiac         mean of the 216 patch tokens (primary)
  brainiac_token0  upstream features[0][:, 0], the first patch token (secondary)
and writes one embedding index per representation.
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--image-list-path",
        type=Path,
        default=PROJECT_ROOT / "results/paper_neurojepa_feature_audit/locked_inputs/embedding_index.csv",
        help="Locked image list; read only (keys and source_image_path).",
    )
    parser.add_argument("--image-path-column", default="source_image_path")
    parser.add_argument("--checkpoint-path", type=Path, default=PROJECT_ROOT / "downloads/brainiac/BrainIAC.ckpt")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/fm_encoders")
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi index (PCI bus order). Default: 0.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N pending images (testing).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_gpu(args.gpu if args.device.startswith("cuda") else None)

    import torch

    from neurojepa_adni_audit.encoders.brainiac import REPRESENTATIONS, build_transform, load_backbone, pool_tokens
    from neurojepa_adni_audit.encoders.extraction import pending_rows, run_extraction

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    set_full_precision()
    spec = load_encoder_spec(args.protocol_path, "brainiac")
    checkpoint_sha256 = require_sha256(args.checkpoint_path, spec.get("checkpoint_sha256"), "BrainIAC checkpoint")

    rows = load_image_list(args.image_list_path, args.image_path_column)
    embeddings_dir = args.output_root / "brainiac" / "embeddings"
    suffixes = dict(REPRESENTATIONS)
    todo = pending_rows(rows, embeddings_dir, suffixes)
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"BrainIAC: {len(rows)} images in list, {len(todo)} to extract into {embeddings_dir}", flush=True)

    model, absent_tensors = load_backbone(args.checkpoint_path)
    model = model.to(args.device)
    transform = build_transform()

    def load_fn(path: str) -> torch.Tensor:
        return torch.as_tensor(transform({"image": path})["image"], dtype=torch.float32)

    @torch.inference_mode()
    def encode_fn(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return pool_tokens(model(batch.to(args.device))[0])

    failures = run_extraction(
        rows=todo,
        output_dir=embeddings_dir,
        suffixes=suffixes,
        load_fn=load_fn,
        encode_fn=encode_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress_label="BrainIAC",
    )

    remaining = pending_rows(rows, embeddings_dir, suffixes)
    complete = rows[~rows["image_id"].isin(remaining["image_id"])]
    for name, suffix in suffixes.items():
        role = spec["representation_runs"][name]
        index = build_index(
            complete, output_dir=embeddings_dir, pooled_suffix=suffix, encoder="brainiac",
            representation=f"{name}: {role['pooling']}",
        )
        index_path = args.output_root / name / "embedding_index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_path, index=False)
        write_summary(index_path.with_name("embedding_summary.json"), {
            "encoder": "brainiac",
            "representation": name,
            "role": role["role"],
            "pooling": role["pooling"],
            "status": "complete" if remaining.empty else "incomplete",
            "n_images_in_list": int(len(rows)),
            "n_embeddings_indexed": int(len(index)),
            "n_missing": int(len(remaining)),
            "failures_this_run": failures,
            "image_list_path": str(args.image_list_path.resolve()),
            "embeddings_dir": str(embeddings_dir.resolve()),
            "checkpoint_path": str(args.checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "code_commit_upstream": spec.get("code_commit"),
            "absent_unused_cross_attention_tensors": absent_tensors,
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
