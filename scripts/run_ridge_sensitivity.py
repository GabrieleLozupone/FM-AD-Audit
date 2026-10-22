#!/usr/bin/env python3
"""Post hoc ridge regularization sensitivity analysis (decided 2026-10-21): extended alpha grid.

In the primary linear matrix the inner CV selected alpha = 1000, the largest value of the
locked grid, in most embedding-model fits of the regression endpoints (table S9). This
refits the 8 regression core endpoints with the grid extended to 1e6 and nothing else
changed: same unchanged runner (run_nested_tabular_probe.py), locked inputs, folds,
repeats, targets and feature groups.

Scope (only what the reported claims need):
  * encoders: Neuro-JEPA, DINOv3, BrainIAC (mean pooling); the token-0 variant is descriptive.
  * endpoints: same-visit ADAS-Cog13/ADAS-Cog11/MMSE, 12-month ADAS-Cog13/ADAS-Cog11/MMSE,
    12-month hippocampus/ICV and lateral ventricles/ICV. AD vs CN, conversion and survival
    use logistic/Cox models and are not affected.
  * groups: conventional_full (reference), neurojepa_conventional (reference + embedding,
    incremental contrasts) and neurojepa_only (standalone cross-encoder contrasts).
    conventional_full is fitted once, in the Neuro-JEPA process together with the embedding
    groups (as in the original matrix, same rows), and symlinked for DINOv3 and BrainIAC.

Everything is written under results/fm_encoders/ridge_sensitivity/; the primary results
are only read. Each (encoder, endpoint, repeat, outer fold) is one process with one BLAS
thread. Default is a read-only plan; --execute runs it (resumable: rerun the same command
after deleting any INCOMPLETE run dir shown by --status); --report then builds:

  paired_report/nested_paired_effects.csv  the 8 refitted contrasts per family, with BH
      recomputed over the original family (11 core endpoints per encoder, 33 standalone
      contrasts), reusing the unchanged p-values of the logistic/Cox endpoints, next to the
      primary estimate, interval and q-value;
  alpha_selection.csv / alpha_summary.csv   selected alpha per run, before and after;
  identity_check.csv  runs whose selected alpha is unchanged must reproduce the primary
      outer-test predictions exactly (integrity check of the refit).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neurojepa_adni_audit.evaluation.paired import benjamini_hochberg  # noqa: E402
from neurojepa_adni_audit.protocol import load_protocol  # noqa: E402

FM = PROJECT_ROOT / "results/fm_encoders"
NJ_ROOT = PROJECT_ROOT / "results/paper_neurojepa_feature_audit"
ROOT = FM / "ridge_sensitivity"
LOG_ROOT = PROJECT_ROOT / "logs/fm_encoders/ridge_sensitivity"
BASE_PROTOCOL = PROJECT_ROOT / "configs/protocol_multi_encoder.json"
PROTOCOL = PROJECT_ROOT / "configs/protocol_ridge_sensitivity.json"
BASE_SPEC = PROJECT_ROOT / "configs/nested_paired_report_multi_encoder.json"
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1.0e4, 1.0e5, 1.0e6]
PRIMARY_MAX_ALPHA = 1000.0

# encoder -> (locked inputs, primary experiments root)
ENCODERS = {
    "neurojepa": (NJ_ROOT / "locked_inputs", NJ_ROOT / "experiments"),
    "dinov3": (FM / "dinov3/locked_inputs", FM / "dinov3/experiments"),
    "brainiac": (FM / "brainiac/locked_inputs", FM / "brainiac/experiments"),
}
GROUPS = {
    "neurojepa": ("conventional_full", "neurojepa_conventional", "neurojepa_only"),
    "dinov3": ("neurojepa_conventional", "neurojepa_only"),
    "brainiac": ("neurojepa_conventional", "neurojepa_only"),
}
# (task, --target, --horizon-months, endpoint directory)
ENDPOINTS = (
    *[("same_visit_cognition", f"{t}_at_session", None, f"same_visit_cognition/{t}_at_session") for t in ("adas13", "adas11", "mmse")],
    *[("future_cognition", f"{t}_at_session", 12, f"future_cognition/{t}_at_session_12m") for t in ("adas13", "adas11", "mmse")],
    *[("future_morphometry", t, 12, f"future_morphometry/{t}_12m") for t in ("hippocampus_total_icv", "lateral_ventricles_total_icv")],
)
COMPLETE = ("summary.json", "outer_test_predictions.csv", "selected_inner_oof_predictions.csv", "outer_test_calibration_smooth.csv")
# Primary paired reports (read only) and the families whose BH is recomputed.
PRIMARY_EFFECTS = {
    "neurojepa": NJ_ROOT / "paired_report/nested_paired_effects.csv",
    "fm": FM / "paired_report/nested_paired_effects.csv",
}
STANDALONE_PAIRS = ("neurojepa_minus_dinov3", "brainiac_minus_dinov3", "brainiac_minus_neurojepa")


@dataclass(frozen=True)
class Job:
    encoder: str
    task: str
    target: str
    horizon: int | None
    endpoint_dir: str
    repeat: str
    outer_fold: int

    def run_dirs(self) -> list[Path]:
        base = ROOT / self.encoder / "experiments" / self.endpoint_dir
        return [base / group / self.repeat / f"outer_{self.outer_fold}" for group in GROUPS[self.encoder]]

    def state(self) -> str:
        dirs = self.run_dirs()
        if all(all((d / name).exists() for name in COMPLETE) for d in dirs):
            return "complete"
        if any(d.exists() and not all((d / name).exists() for name in COMPLETE) for d in dirs):
            return "incomplete"
        return "pending"

    def command(self) -> list[str]:
        inputs, _ = ENCODERS[self.encoder]
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts/run_nested_tabular_probe.py"),
            "--protocol-path", str(PROTOCOL), "--input-dir", str(inputs),
            "--output-dir", str(ROOT / self.encoder / "experiments"), "--resume", "--allow-draft-protocol",
            "--task", self.task, "--target", self.target,
        ]
        if self.horizon is not None:
            command += ["--horizon-months", str(self.horizon)]
        for group in GROUPS[self.encoder]:
            command += ["--feature-group", group]
        return command + ["--repeat", self.repeat, "--outer-fold", str(self.outer_fold)]

    @property
    def label(self) -> str:
        return f"{self.encoder}/ridge/{self.endpoint_dir}/{self.repeat}/outer_{self.outer_fold}"


def write_protocol() -> None:
    """configs/protocol_ridge_sensitivity.json = the multi-encoder protocol with only ridge_alphas extended."""
    payload = json.loads(BASE_PROTOCOL.read_text(encoding="utf-8"))
    payload["protocol_version"] = f"{payload['protocol_version']}-ridge-sensitivity"
    payload["status"] = "draft"
    payload["ridge_alphas"] = RIDGE_ALPHAS
    payload["ridge_sensitivity"] = {
        "derived_from": str(BASE_PROTOCOL.relative_to(PROJECT_ROOT)),
        "changed_keys": ["protocol_version", "status", "ridge_alphas"],
        "reason": "Inner CV selected the largest locked alpha (1000) in most embedding-model regression fits; "
                  "declared sensitivity, primary results unchanged (decision 2026-10-21).",
    }
    text = json.dumps(payload, indent=2) + "\n"
    if PROTOCOL.exists():
        if PROTOCOL.read_text(encoding="utf-8") != text:
            raise ValueError(f"{PROTOCOL} differs from its derivation from {BASE_PROTOCOL}; refusing to continue.")
        return
    PROTOCOL.write_text(text, encoding="utf-8")
    print(f"Wrote {PROTOCOL}")


def build_jobs() -> list[Job]:
    protocol = load_protocol(BASE_PROTOCOL, require_frozen=False)
    repeats = [replication.name for replication in protocol.replications]
    folds = range(int(protocol.payload["partitioning"]["outer_folds"]))
    return [
        Job(encoder, task, target, horizon, endpoint_dir, repeat, fold)
        for encoder in ENCODERS
        for task, target, horizon, endpoint_dir in ENDPOINTS
        for repeat in repeats
        for fold in folds
    ]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    return environment


def _run(job: Job) -> tuple[Job, int, float]:
    log_path = LOG_ROOT / job.encoder / (job.label.split("/", 2)[2].replace("/", "__") + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(job.command())}\n")
        log.flush()
        code = subprocess.run(job.command(), stdout=log, stderr=subprocess.STDOUT, env=_environment(), cwd=PROJECT_ROOT).returncode
    return job, code, time.monotonic() - started


def execute(jobs: list[Job], workers: int) -> list[Job]:
    # Same-visit endpoints have the most rows and Neuro-JEPA jobs fit three groups: start them first.
    pending = sorted((job for job in jobs if job.state() == "pending"),
                     key=lambda job: (job.task != "same_visit_cognition", job.encoder != "neurojepa"))
    incomplete = [job for job in jobs if job.state() == "incomplete"]
    if incomplete:
        print(f"WARNING {len(incomplete)} job(s) have incomplete run dirs and will fail until removed; see --status.", flush=True)
    # Keep these line formats stable: external progress monitors parse them.
    print(f"Launching {len(pending)} runs on {workers} workers", flush=True)
    failed: list[Job] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, job) for job in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            job, code, seconds = future.result()
            status = "DONE" if code == 0 and job.state() == "complete" else f"FAILED(exit={code})"
            if status != "DONE":
                failed.append(job)
            elapsed = (time.monotonic() - started) / 3600
            print(f"[{done}/{len(pending)} {elapsed:5.1f}h] {status} {job.label} ({seconds / 60:.1f} min)", flush=True)
    return failed


def print_status(jobs: list[Job]) -> None:
    for encoder in ENCODERS:
        group = [job for job in jobs if job.encoder == encoder]
        states = [job.state() for job in group]
        print(f"{encoder:<10} {states.count('complete'):>3}/{len(group)} complete", flush=True)
        for job, state in zip(group, states):
            if state == "incomplete":
                for run_dir in job.run_dirs():
                    if run_dir.exists() and not all((run_dir / name).exists() for name in COMPLETE):
                        print(f"    INCOMPLETE {run_dir}  (delete it, then rerun)")


# ----------------------------------------------------------------------------- report


def _link(link: Path, source: Path) -> None:
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ValueError(f"{link} points elsewhere: {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink.")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(source.resolve(), target_is_directory=True)


def link_inputs() -> None:
    for _task, _target, _horizon, endpoint_dir in ENDPOINTS:
        shared = ROOT / "neurojepa/experiments" / endpoint_dir / "conventional_full"
        for encoder in ("dinov3", "brainiac"):
            _link(ROOT / encoder / "experiments" / endpoint_dir / "conventional_full", shared)
        for encoder in ENCODERS:
            _link(ROOT / "paired_report_inputs/standalone" / endpoint_dir / encoder,
                  ROOT / encoder / "experiments" / endpoint_dir / "neurojepa_only")


def build_spec() -> list[dict]:
    """The primary comparisons of the refitted endpoints, pointed at the sensitivity runs."""
    targets = {endpoint_dir.split("/", 1)[1] for *_, endpoint_dir in ENDPOINTS}
    comparisons = []
    for comparison in json.loads(BASE_SPEC.read_text(encoding="utf-8"))["comparisons"]:
        if comparison["decoder"] != "linear" or comparison["target"] not in targets:
            continue
        encoder = comparison["encoder"]
        if comparison["family"] == "dinov3_global_core_endpoints":
            # Neuro-JEPA comparisons have the same definition, with the base ids of the primary report.
            nj = {**comparison, "id": comparison["base_endpoint_id"], "encoder": "neurojepa",
                  "family": "global_core_endpoints", "predictions_root": str(ROOT / "neurojepa/experiments")}
            comparisons.append(nj)
        if encoder in ("dinov3", "brainiac"):
            comparisons.append({**comparison, "predictions_root": str(ROOT / encoder / "experiments")})
        elif encoder in STANDALONE_PAIRS:
            comparisons.append({**comparison, "predictions_root": str(ROOT / "paired_report_inputs/standalone")})
    return comparisons


def run_family(family: str, comparisons: list[dict]) -> int:
    family_dir = ROOT / "paired_report/families" / family
    if (family_dir / "nested_paired_effects.csv").exists():
        return 0
    family_dir.mkdir(parents=True, exist_ok=True)
    spec = family_dir / "spec.json"
    spec.write_text(json.dumps({"comparisons": comparisons}, indent=1) + "\n", encoding="utf-8")
    with (family_dir / "run.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts/report_nested_paired_results.py"), "--spec-path", str(spec),
             "--protocol-path", str(PROTOCOL), "--allow-draft-protocol", "--output-dir", str(family_dir), "--no-progress"],
            stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=_environment(),
        ).returncode
    print(f"{'DONE' if code == 0 else f'FAILED(exit={code})'} paired report {family} ({len(comparisons)} comparisons)", flush=True)
    return code


def merge_effects(families: list[str]) -> pd.DataFrame:
    """BH over each original family: refitted p-values replace the regression ones, the
    logistic/Cox p-values of that family are reused unchanged."""
    new = pd.concat([pd.read_csv(ROOT / "paired_report/families" / f / "nested_paired_effects.csv") for f in families],
                    ignore_index=True)
    primary = pd.concat([pd.read_csv(path) for path in PRIMARY_EFFECTS.values()], ignore_index=True)
    primary = primary.drop_duplicates("id")
    q = {}
    for family in families:
        original = primary[primary["family"] == family]
        refitted = new[new["family"] == family]
        missing = set(refitted["id"]) - set(original["id"])
        if missing:
            raise ValueError(f"{family}: ids not in the primary report: {sorted(missing)}")
        pool = pd.concat([refitted[["id", "bootstrap_sign_p_value"]],
                          original.loc[~original["id"].isin(refitted["id"]), ["id", "bootstrap_sign_p_value"]]])
        q.update(zip(pool["id"], benjamini_hochberg(pool["bootstrap_sign_p_value"].tolist())))
        print(f"{family}: BH over {len(pool)} comparisons ({len(refitted)} refitted, {len(pool) - len(refitted)} reused)")
    new["q_full_family"] = new["id"].map(q)
    keep = ["id", "estimate", "ci95_low", "ci95_high", "bootstrap_sign_p_value", "fdr_adjusted_p_value"]
    merged = new.merge(primary[keep].rename(columns={c: f"primary_{c}" for c in keep[1:]}), on="id", how="left")
    merged.to_csv(ROOT / "paired_report/nested_paired_effects.csv", index=False)
    return merged


def alpha_tables(jobs: list[Job]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for job in jobs:
        _, primary_root = ENCODERS[job.encoder]
        for group, run_dir in zip(GROUPS[job.encoder], job.run_dirs()):
            sub = Path(job.endpoint_dir) / group / job.repeat / f"outer_{job.outer_fold}"
            new = json.loads((ROOT / job.encoder / "experiments" / sub / "summary.json").read_text(encoding="utf-8"))
            old = json.loads((primary_root / sub / "summary.json").read_text(encoding="utf-8"))
            rows.append({
                "encoder": "shared" if group == "conventional_full" else job.encoder,
                "endpoint": job.endpoint_dir, "feature_group": group, "repeat": job.repeat, "outer_fold": job.outer_fold,
                "alpha_primary": old["selected_hyperparameter"], "alpha_extended": new["selected_hyperparameter"],
                "mae_primary": old["outer_test_metric_value"], "mae_extended": new["outer_test_metric_value"],
            })
    alphas = pd.DataFrame(rows)
    alphas.to_csv(ROOT / "alpha_selection.csv", index=False)
    summary = alphas.groupby(["encoder", "feature_group"]).agg(
        runs=("alpha_extended", "size"),
        primary_at_1000=("alpha_primary", lambda a: float(np.mean(a == PRIMARY_MAX_ALPHA))),
        extended_above_1000=("alpha_extended", lambda a: float(np.mean(a > PRIMARY_MAX_ALPHA))),
        extended_at_max=("alpha_extended", lambda a: float(np.mean(a == max(RIDGE_ALPHAS)))),
        median_alpha_extended=("alpha_extended", "median"),
    ).reset_index()
    summary.to_csv(ROOT / "alpha_summary.csv", index=False)
    return alphas, summary


def identity_check(alphas: pd.DataFrame) -> pd.DataFrame:
    """Same selected alpha => the refit must give the primary outer-test predictions."""
    rows = []
    same = alphas[alphas["alpha_primary"] == alphas["alpha_extended"]]
    for row in same.itertuples(index=False):
        encoder = "neurojepa" if row.encoder == "shared" else row.encoder
        sub = Path(row.endpoint) / row.feature_group / row.repeat / f"outer_{row.outer_fold}" / "outer_test_predictions.csv"
        new = pd.read_csv(ROOT / encoder / "experiments" / sub)
        old = pd.read_csv(ENCODERS[encoder][1] / sub)
        identical_rows = len(new) == len(old) and (new.iloc[:, :3].astype(str).values == old.iloc[:, :3].astype(str).values).all()
        diff = float(np.max(np.abs(new["prediction"].to_numpy() - old["prediction"].to_numpy()))) if identical_rows else float("nan")
        rows.append({**row._asdict(), "same_rows": identical_rows, "max_abs_prediction_diff": diff})
    check = pd.DataFrame(rows)
    check.to_csv(ROOT / "identity_check.csv", index=False)
    return check


def report(jobs: list[Job], workers: int) -> None:
    not_done = [job for job in jobs if job.state() != "complete"]
    if not_done:
        sys.exit(f"{len(not_done)} job(s) not complete; run --execute first.")
    link_inputs()
    by_family: dict[str, list[dict]] = {}
    for comparison in build_spec():
        by_family.setdefault(comparison["family"], []).append(comparison)
    print("Comparisons per family: " + ", ".join(f"{f} {len(c)}" for f, c in by_family.items()), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        codes = list(pool.map(lambda family: run_family(family, by_family[family]), by_family))
    if any(codes):
        sys.exit("Paired report failed; see results/fm_encoders/ridge_sensitivity/paired_report/families/*/run.log, "
                 "delete that family folder and rerun --report.")
    effects = merge_effects(list(by_family))
    alphas, summary = alpha_tables(jobs)
    check = identity_check(alphas)
    pd.set_option("display.width", 200)
    print("\nSelected alpha (fraction of the 96 runs per encoder and group):")
    print(summary.to_string(index=False))
    # 1e-4: the primary DINOv3/BrainIAC linear runs used numpy 2.4.6 on a second server (reruns differed by up to 5.5e-6).
    exact = int((check["same_rows"] & (check["max_abs_prediction_diff"] < 1e-4)).sum())
    print(f"\nIdentity check: {exact}/{len(check)} runs with unchanged alpha reproduce the primary predictions "
          f"(max diff {check['max_abs_prediction_diff'].max():.2e}).")
    columns = ["id", "primary_estimate", "estimate", "ci95_low", "ci95_high", "primary_fdr_adjusted_p_value", "q_full_family"]
    print("\nPaired effects, primary vs extended grid:")
    print(effects[columns].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nWritten under {ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Run the refits; default is a read-only plan.")
    parser.add_argument("--report", action="store_true", help="Paired report, alpha tables and identity check.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    jobs = build_jobs()
    if args.report:
        report(jobs, args.workers)
        return
    if args.status or not args.execute:
        print_status(jobs)
        if not args.execute:
            if not args.status:
                pending = sum(job.state() != "complete" for job in jobs)
                print(f"Plan: {len(jobs)} jobs ({pending} to do), {len(ENDPOINTS)} endpoints x 12 folds x "
                      f"{len(ENCODERS)} encoders; alpha grid {RIDGE_ALPHAS}. Add --execute to start.")
            return
    write_protocol()
    failed = execute(jobs, args.workers)
    if failed:
        print(f"{len(failed)} run(s) failed; logs under {LOG_ROOT}. Rerun the same command to retry.")
        sys.exit(1)
    print("All runs complete.", flush=True)
    print_status(jobs)


if __name__ == "__main__":
    main()
