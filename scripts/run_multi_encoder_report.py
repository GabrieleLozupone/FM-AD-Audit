#!/usr/bin/env python3
"""Run the multi-encoder paired report (protocol 0.6.0) with one process per BH family.

Splits configs/nested_paired_report_multi_encoder.json by family and runs the UNCHANGED
scripts/report_nested_paired_results.py once per family, in parallel, into
<output-dir>/families/<family>/. BH correction is applied within each family by that
script, so splitting by family gives the same q-values as a single run. Then merges:

  <output-dir>/nested_paired_effects.csv   all comparisons, plus
      sensitivity_all_encoders_linear_q: one BH over the 33 linear incremental core effects
      (published Neuro-JEPA 11 + dinov3 11 + brainiac 11; protocol sensitivity_fdr_families).
      The Neuro-JEPA rows are read from the published report, not recomputed.
  <output-dir>/outer_fold_effects.csv      fold-level dispersion of all comparisons

Resumable: a family whose output already exists is skipped. Each process uses 1 BLAS thread.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurojepa_adni_audit.evaluation.paired import benjamini_hochberg  # noqa: E402

NJ_EFFECTS = PROJECT_ROOT / "results/paper_neurojepa_feature_audit/paired_report/nested_paired_effects.csv"


def run_family(family: str, comparisons: list[dict], output_dir: Path, protocol: Path) -> tuple[str, int]:
    family_dir = output_dir / "families" / family
    if (family_dir / "nested_paired_effects.csv").exists():
        return family, 0
    family_dir.mkdir(parents=True, exist_ok=True)
    spec = family_dir / "spec.json"
    spec.write_text(json.dumps({"comparisons": comparisons}, indent=1) + "\n", encoding="utf-8")
    environment = {**os.environ, **{name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    with (family_dir / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/report_nested_paired_results.py"), "--spec-path", str(spec),
             "--protocol-path", str(protocol), "--allow-draft-protocol", "--output-dir", str(family_dir), "--no-progress"],
            stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=environment,
        ).returncode
    print(f"{'DONE' if code == 0 else f'FAILED(exit={code})'} {family} ({len(comparisons)} comparisons)", flush=True)
    return family, code


def merge(output_dir: Path, families: list[str], protocol: Path) -> None:
    effects = pd.concat([pd.read_csv(output_dir / "families" / f / "nested_paired_effects.csv") for f in families], ignore_index=True)
    folds = pd.concat([pd.read_csv(output_dir / "families" / f / "outer_fold_effects.csv") for f in families], ignore_index=True)
    sensitivity = json.loads(protocol.read_text(encoding="utf-8"))["sensitivity_fdr_families"]["all_encoders_linear_core_endpoints"]
    nj = pd.read_csv(NJ_EFFECTS)
    pool = pd.concat([nj[nj["id"].isin(sensitivity)], effects[effects["id"].isin(sensitivity)]], ignore_index=True)
    if sorted(pool["id"]) != sorted(sensitivity):
        raise ValueError(f"sensitivity family incomplete: {sorted(set(sensitivity) - set(pool['id']))}")
    q = dict(zip(pool["id"], benjamini_hochberg(pool["bootstrap_sign_p_value"].tolist())))
    effects["sensitivity_all_encoders_linear_q"] = effects["id"].map(q)
    effects.to_csv(output_dir / "nested_paired_effects.csv", index=False)
    folds.to_csv(output_dir / "outer_fold_effects.csv", index=False)
    pd.DataFrame({"id": list(q), "sensitivity_all_encoders_linear_q": list(q.values())}).to_csv(
        output_dir / "sensitivity_all_encoders_linear_q.csv", index=False)
    print(f"Merged {len(effects)} comparisons into {output_dir / 'nested_paired_effects.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec-path", type=Path, default=PROJECT_ROOT / "configs/nested_paired_report_multi_encoder.json")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/fm_encoders/paired_report")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    by_family: dict[str, list[dict]] = {}
    for comparison in json.loads(args.spec_path.read_text(encoding="utf-8"))["comparisons"]:
        by_family.setdefault(comparison["family"], []).append(comparison)
    # Longest families first: survival and same-visit dominate, and the standalone family has 33 comparisons.
    order = sorted(by_family, key=lambda family: -len(by_family[family]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda f: run_family(f, by_family[f], args.output_dir, args.protocol_path), order))
    failed = [family for family, code in results if code]
    if failed:
        sys.exit(f"{len(failed)} family run(s) failed: {failed}; see <output-dir>/families/<family>/run.log, delete that folder, rerun.")
    merge(args.output_dir, list(by_family), args.protocol_path)


if __name__ == "__main__":
    main()
