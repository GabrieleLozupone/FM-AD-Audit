#!/usr/bin/env python3
"""Build the paired-report specification for the multi-encoder extension (protocol 0.6.0).

Writes configs/nested_paired_report_multi_encoder.json for the UNCHANGED
scripts/report_nested_paired_results.py. Every comparison id and BH family is the one
prespecified in configs/protocol_multi_encoder.json (fdr_families); the builder refuses
to write a spec whose ids or families differ from the protocol.

Comparisons, all on identical participants, rows and folds:
  * per encoder (dinov3, brainiac, brainiac_token0), linear (11) and MLP (10):
    embedding + conventional_full minus conventional_full. Same estimand as the
    Neuro-JEPA report, pointed at results/fm_encoders/<rep>/{experiments,nonlinear_experiments},
    whose conventional_full directories are the read-only Neuro-JEPA symlinks.
  * cross-encoder standalone, linear (33): embedding-only A minus embedding-only B for
    neurojepa-dinov3, brainiac-dinov3, brainiac-neurojepa.
  * BrainIAC pooling, linear (11, descriptive): brainiac minus brainiac_token0, embedding-only.

The reporting code compares two feature directories under one predictions root
("combined" minus "conventional"). For the standalone contrasts, the builder creates
results/fm_encoders/paired_report_inputs/standalone/<task>/<target>/<encoder> as symlinks to
each encoder's neurojepa_only directory, so A and B sit under one root and no existing
source file needs changing. The Neuro-JEPA paired report is not recomputed: its effects
stay as published in results/paper_neurojepa_feature_audit/paired_report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = PROJECT_ROOT / "configs/nested_paired_report.json"
PROTOCOL = PROJECT_ROOT / "configs/protocol_multi_encoder.json"
OUTPUT = PROJECT_ROOT / "configs/nested_paired_report_multi_encoder.json"
FM = "results/fm_encoders"
NJ = "results/paper_neurojepa_feature_audit"
REPRESENTATIONS = ("dinov3", "brainiac", "brainiac_token0")
STANDALONE_ROOT = f"{FM}/paired_report_inputs/standalone"
EMBEDDING_ONLY = {
    "neurojepa": f"{NJ}/experiments",
    "dinov3": f"{FM}/dinov3/experiments",
    "brainiac": f"{FM}/brainiac/experiments",
    "brainiac_token0": f"{FM}/brainiac_token0/experiments",
}
STANDALONE_CONTRASTS = (("neurojepa", "dinov3"), ("brainiac", "dinov3"), ("brainiac", "neurojepa"))
POOLING_CONTRAST = ("brainiac", "brainiac_token0")


def build(base: list[dict]) -> list[dict]:
    linear = [c for c in base if c["decoder"] == "linear"]
    mlp = [c for c in base if c["decoder"] == "mlp"]
    comparisons: list[dict] = []
    for rep in REPRESENTATIONS:
        for c in linear:
            comparisons.append({
                **c, "id": f"{rep}_{c['id']}", "base_endpoint_id": c["id"], "encoder": rep,
                "family": f"{rep}_global_core_endpoints", "predictions_root": f"{FM}/{rep}/experiments",
            })
        for c in mlp:
            comparisons.append({
                **c, "id": f"{rep}_{c['id']}", "base_endpoint_id": c["base_endpoint_id"], "encoder": rep,
                "family": f"{rep}_nonlinear_decoder_core_endpoints", "predictions_root": f"{FM}/{rep}/nonlinear_experiments",
            })
    for (first, second), family, prefix, analysis_class in (
        *[(pair, "cross_encoder_standalone_core_endpoints", "standalone", "secondary_cross_encoder") for pair in STANDALONE_CONTRASTS],
        (POOLING_CONTRAST, "brainiac_pooling_descriptive", "pooling", "descriptive"),
    ):
        for c in linear:
            comparisons.append({
                **c, "id": f"{prefix}_{first}_minus_{second}_{c['id']}", "base_endpoint_id": c["id"],
                "encoder": f"{first}_minus_{second}", "family": family, "analysis_class": analysis_class,
                "predictions_root": STANDALONE_ROOT, "combined_feature": first, "conventional_feature": second,
            })
    return comparisons


def check_against_protocol(comparisons: list[dict]) -> None:
    families = json.loads(PROTOCOL.read_text(encoding="utf-8"))["fdr_families"]
    built: dict[str, list[str]] = {}
    for c in comparisons:
        built.setdefault(c["family"], []).append(c["id"])
    for family, ids in built.items():
        if family == "brainiac_pooling_descriptive":
            continue  # descriptive contrast; the protocol defines it without a BH family
        if family not in families:
            raise ValueError(f"family {family} is not in the protocol")
        if sorted(ids) != sorted(families[family]):
            raise ValueError(f"ids of {family} differ from the protocol: {sorted(set(ids) ^ set(families[family]))}")


def link_standalone_inputs(linear: list[dict], require_complete: bool) -> int:
    created = 0
    for c in linear:
        for encoder, root in EMBEDDING_ONLY.items():
            source = (PROJECT_ROOT / root / c["task"] / c["target"] / "neurojepa_only").resolve()
            if require_complete and not any(source.glob("*/outer_*/outer_test_predictions.csv")):
                raise FileNotFoundError(f"no predictions under {source}")
            link = PROJECT_ROOT / STANDALONE_ROOT / c["task"] / c["target"] / encoder
            if link.is_symlink():
                if link.resolve() != source:
                    raise ValueError(f"{link} points to {link.resolve()}, expected {source}")
                continue
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source, target_is_directory=True)
            created += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-incomplete", action="store_true", help="link even if some predictions are missing")
    args = parser.parse_args()
    base = json.loads(BASE_SPEC.read_text(encoding="utf-8"))["comparisons"]
    comparisons = build(base)
    check_against_protocol(comparisons)
    created = link_standalone_inputs([c for c in base if c["decoder"] == "linear"], not args.allow_incomplete)
    if args.output.exists() and json.loads(args.output.read_text(encoding="utf-8"))["comparisons"] != comparisons:
        raise FileExistsError(f"{args.output} exists with different content; refusing to overwrite")
    args.output.write_text(json.dumps({"comparisons": comparisons}, indent=1) + "\n", encoding="utf-8")
    families: dict[str, int] = {}
    for c in comparisons:
        families[c["family"]] = families.get(c["family"], 0) + 1
    print(f"Wrote {len(comparisons)} comparisons to {args.output}; {created} standalone symlink(s) created")
    for family, count in families.items():
        print(f"  {family:50s} {count}")


if __name__ == "__main__":
    main()
