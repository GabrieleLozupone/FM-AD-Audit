#!/usr/bin/env python3
"""Run the locked Neuro-JEPA experiment matrices for an additional encoder (protocol 0.6.0).

The unchanged runners (run_nested_tabular_probe.py, run_nested_survival_probe.py,
run_nested_nonlinear_probe.py) are called with the same commands that
run_locked_experiment_matrix.py / run_locked_nonlinear_matrix.py build, pointed at
results/fm_encoders/<representation>/locked_inputs. Two differences, both
computational only:

  * Only the embedding feature groups are run. The encoder-independent conventional
    groups are reused from the Neuro-JEPA root through read-only symlinks
    (--link-baselines; protocol conventional_feature_groups_reused) and are never passed to a runner, so no
    runner can ever write into the Neuro-JEPA results.
  * Each (endpoint, feature group, repeat, outer fold) run is its own process, so runs
    execute in parallel (--workers). Longest jobs start first. Each process uses one
    BLAS/OpenMP thread. A final sequential --resume pass per endpoint (all runs skip)
    rewrites each endpoint's complete run index.

Default is a read-only plan; add --execute to run. Safe to interrupt and rerun.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.features.groups import FEATURE_GROUPS, uses_neurojepa  # noqa: E402
from neurojepa_adni_audit.protocol import load_protocol  # noqa: E402


REPRESENTATIONS = ("dinov3", "brainiac", "brainiac_token0")
NJ_ROOT = PROJECT_ROOT / "results/paper_neurojepa_feature_audit"
EMBEDDING_GROUPS = tuple(group for group in FEATURE_GROUPS if uses_neurojepa(group))
CONVENTIONAL_GROUPS = tuple(group for group in FEATURE_GROUPS if not uses_neurojepa(group))
TABULAR_COMPLETE = ("summary.json", "outer_test_predictions.csv", "selected_inner_oof_predictions.csv", "outer_test_calibration_smooth.csv")
SURVIVAL_COMPLETE = (
    "summary.json",
    "outer_test_predictions.csv",
    "inner_model_selection.csv",
    "outer_test_calibration.csv",
    "development_scaled_schoenfeld_residuals.csv",
)
# Mean hours per embedding-group run, measured on the Neuro-JEPA matrices (2026-07).
# Used only to start the longest jobs first and to print estimates.
HOURS_PER_RUN = {
    ("linear", "first_mci_survival"): 4.8,
    ("linear", "same_visit_cognition"): 0.14,
    ("linear", "ad_vs_cn"): 0.04,
    ("linear", "future_cognition"): 0.010,
    ("linear", "future_morphometry"): 0.011,
    ("linear", "mci_conversion_36m"): 0.002,
    ("mlp", "same_visit_cognition"): 0.055,
    ("mlp", "future_cognition"): 0.024,
    ("mlp", "future_morphometry"): 0.025,
    ("mlp", "ad_vs_cn"): 0.008,
    ("mlp", "mci_conversion_36m"): 0.004,
}


@dataclass(frozen=True)
class Job:
    matrix: str  # "linear" or "mlp"
    representation: str
    task: str
    endpoint_dir: str  # <task>/<endpoint_target> under the output root
    feature_group: str
    repeat: str
    outer_fold: int
    command: tuple[str, ...]
    output_root: Path

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.endpoint_dir / self.feature_group / self.repeat / f"outer_{self.outer_fold}"

    @property
    def complete_files(self) -> tuple[str, ...]:
        return SURVIVAL_COMPLETE if self.task == "first_mci_survival" else TABULAR_COMPLETE

    def state(self) -> str:
        if not self.run_dir.exists():
            return "pending"
        return "complete" if all((self.run_dir / name).exists() for name in self.complete_files) else "incomplete"

    @property
    def hours(self) -> float:
        return HOURS_PER_RUN.get((self.matrix, self.task), 0.05)

    @property
    def label(self) -> str:
        return f"{self.representation}/{self.matrix}/{self.endpoint_dir}/{self.feature_group}/{self.repeat}/outer_{self.outer_fold}"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_feature_groups(command: list[str]) -> list[str]:
    stripped, skip = [], False
    for argument in command:
        if skip:
            skip = False
            continue
        if argument == "--feature-group":
            skip = True
            continue
        stripped.append(argument)
    return stripped


def _option(command: list[str], name: str) -> str | None:
    return command[command.index(name) + 1] if name in command else None


def _tabular_endpoint_dir(command: list[str]) -> tuple[str, str]:
    if "run_nested_survival_probe.py" in command[1]:
        return "first_mci_survival", "first_mci_survival/time_to_ad"
    task = _option(command, "--task")
    target = _option(command, "--target") or "label"
    horizon = _option(command, "--horizon-months")
    if horizon is not None:
        target = f"{target}_{float(horizon):g}m".replace(".", "p")
    return task, f"{task}/{target}"


def _mlp_endpoint_dir(command: list[str]) -> tuple[str, str]:
    task = _option(command, "--task")
    target = _option(command, "--target") or "label"
    horizon = _option(command, "--horizon-months")
    if horizon is not None:
        target = f"{target}_{float(horizon):g}m".replace(".0m", "m")
    return task, f"{task}/{target}"


def paths(fm_root: Path, representation: str) -> dict[str, Path]:
    root = fm_root / representation
    return {
        "input": root / "locked_inputs",
        "linear": root / "experiments",
        "mlp": root / "nonlinear_experiments",
        "logs": PROJECT_ROOT / "logs/fm_encoders" / representation,
    }


def endpoint_commands(matrix: str, protocol_path: Path, input_dir: Path, output_dir: Path, device: str) -> list[list[str]]:
    if matrix == "linear":
        commands = _load_script("run_locked_experiment_matrix").build_commands(
            protocol_path=protocol_path, input_dir=input_dir, output_dir=output_dir, allow_draft=True
        )
    else:
        commands = _load_script("run_locked_nonlinear_matrix").build_commands(
            protocol_path=protocol_path, input_dir=input_dir, output_dir=output_dir,
            device=device, torch_threads=1, allow_draft=True,
        )
    return [_strip_feature_groups(command) for command in commands]


def build_jobs(matrix: str, representation: str, protocol_path: Path, fm_root: Path, device: str = "cuda") -> list[Job]:
    protocol = load_protocol(protocol_path, require_frozen=False)
    repeats = [replication.name for replication in protocol.replications]
    folds = range(int(protocol.payload["partitioning"]["outer_folds"]))
    if matrix == "linear":
        groups = EMBEDDING_GROUPS
    else:
        nonlinear = protocol.payload["nonlinear_decoder_analysis"]
        groups = tuple(
            group
            for group in (*nonlinear["required_feature_groups"], *nonlinear["descriptive_feature_groups"])
            if uses_neurojepa(group)
        )
    where = paths(fm_root, representation)
    jobs = []
    for command in endpoint_commands(matrix, protocol_path, where["input"], where[matrix], device):
        task, endpoint_dir = (_tabular_endpoint_dir if matrix == "linear" else _mlp_endpoint_dir)(command)
        for group in groups:
            for repeat in repeats:
                for fold in folds:
                    jobs.append(
                        Job(
                            matrix, representation, task, endpoint_dir, group, repeat, fold,
                            tuple([*command, "--feature-group", group, "--repeat", repeat, "--outer-fold", str(fold)]),
                            where[matrix],
                        )
                    )
    return jobs


def link_baselines(representation: str, fm_root: Path, protocol_path: Path) -> int:
    """Symlink the encoder-independent conventional runs from the Neuro-JEPA root."""
    created = 0
    where = paths(fm_root, representation)
    protocol = load_protocol(protocol_path, require_frozen=False)
    nonlinear = protocol.payload["nonlinear_decoder_analysis"]
    mlp_conventional = [
        group
        for group in (*nonlinear["required_feature_groups"], *nonlinear["descriptive_feature_groups"])
        if not uses_neurojepa(group)
    ]
    for matrix, groups, nj_dir in (
        ("linear", CONVENTIONAL_GROUPS, NJ_ROOT / "experiments"),
        ("mlp", mlp_conventional, NJ_ROOT / "nonlinear_experiments"),
    ):
        endpoints = {job.endpoint_dir for job in build_jobs(matrix, representation, protocol_path, fm_root)}
        for endpoint_dir in sorted(endpoints):
            for group in groups:
                source = nj_dir / endpoint_dir / group
                link = where[matrix] / endpoint_dir / group
                if not source.is_dir():
                    raise FileNotFoundError(f"Neuro-JEPA baseline missing: {source}")
                if link.is_symlink():
                    if link.resolve() != source.resolve():
                        raise ValueError(f"{link} points elsewhere: {link.resolve()}")
                    continue
                if link.exists():
                    raise FileExistsError(f"{link} exists and is not a symlink; refusing to replace it.")
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source.resolve(), target_is_directory=True)
                created += 1
    return created


def _environment(gpu: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    if gpu is not None:
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = gpu
    return environment


def _run(job: Job, log_dir: Path, gpu: str | None) -> tuple[Job, int, float]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (job.label.split("/", 2)[2].replace("/", "__") + ".log")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(job.command)}\n")
        log.flush()
        code = subprocess.run(job.command, stdout=log, stderr=subprocess.STDOUT, env=_environment(gpu), cwd=PROJECT_ROOT).returncode
    return job, code, time.monotonic() - started


def execute(jobs: list[Job], workers: int, gpus: list[str], fm_root: Path) -> list[Job]:
    pending = sorted((job for job in jobs if job.state() == "pending"), key=lambda job: -job.hours)
    incomplete = [job for job in jobs if job.state() == "incomplete"]
    if incomplete:
        print(f"WARNING {len(incomplete)} incomplete run dir(s) will fail until removed; see --status.", flush=True)
    print(f"Launching {len(pending)} runs on {workers} workers", flush=True)
    failed: list[Job] = []
    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for index, job in enumerate(pending):
            gpu = gpus[index % len(gpus)] if job.matrix == "mlp" and gpus else None
            futures.append(pool.submit(_run, job, paths(fm_root, job.representation)["logs"] / job.matrix, gpu))
        for future in as_completed(futures):
            job, code, seconds = future.result()
            done += 1
            status = "DONE" if code == 0 and job.state() == "complete" else f"FAILED(exit={code})"
            if status != "DONE":
                failed.append(job)
            elapsed = (time.monotonic() - started) / 3600
            print(f"[{done}/{len(pending)} {elapsed:5.1f}h] {status} {job.label} ({seconds / 60:.1f} min)", flush=True)
    return failed


def finalize_run_indexes(jobs: list[Job]) -> None:
    """One --resume pass per endpoint with all embedding groups: every run skips, and
    the runner rewrites that endpoint's run index with all runs (survival has none)."""
    seen = set()
    for job in jobs:
        key = (job.representation, job.matrix, job.endpoint_dir)
        if key in seen or job.task == "first_mci_survival":
            continue
        seen.add(key)
        base = list(job.command[: job.command.index("--feature-group")])
        groups = sorted({other.feature_group for other in jobs if (other.representation, other.matrix, other.endpoint_dir) == key})
        command = [*base, *[argument for group in groups for argument in ("--feature-group", group)]]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=_environment(None), cwd=PROJECT_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Run-index pass failed for {key}: {result.stderr[-2000:]}")


def print_status(jobs: list[Job]) -> None:
    by_key: dict[tuple[str, str], list[Job]] = {}
    for job in jobs:
        by_key.setdefault((job.representation, job.matrix), []).append(job)
    for (representation, matrix), group in sorted(by_key.items()):
        states = [job.state() for job in group]
        complete = states.count("complete")
        remaining_hours = sum(job.hours for job, state in zip(group, states) if state != "complete")
        print(
            f"{representation:<16}{matrix:<7} {complete:>4}/{len(group)} complete "
            f"({100 * complete / len(group):5.1f}%)  remaining ~{remaining_hours:6.1f} CPU/GPU-h",
            flush=True,
        )
        for job, state in zip(group, states):
            if state == "incomplete":
                print(f"    INCOMPLETE {job.run_dir}  (partial output from an interrupted run: delete it, then rerun)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--representation", action="append", choices=[*REPRESENTATIONS, "all"], required=True)
    parser.add_argument("--linear", action="store_true", help="Linear + survival matrix (CPU).")
    parser.add_argument("--mlp", action="store_true", help="MLP matrix (GPU).")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--link-baselines", action="store_true", help="Create the conventional-group symlinks and exit.")
    parser.add_argument("--execute", action="store_true", help="Run; default is a read-only plan.")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--gpus", default="0", help="Comma-separated nvidia-smi indices for MLP runs (round-robin).")
    parser.add_argument("--fm-root", type=Path, default=PROJECT_ROOT / "results/fm_encoders")
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol_multi_encoder.json")
    args = parser.parse_args(argv)
    representations = list(REPRESENTATIONS) if "all" in args.representation else list(dict.fromkeys(args.representation))
    matrices = [name for name, wanted in (("linear", args.linear), ("mlp", args.mlp)) if wanted] or ["linear", "mlp"]

    for representation in representations:
        if not (paths(args.fm_root, representation)["input"] / "provenance.json").exists():
            raise FileNotFoundError(f"No locked inputs for {representation}; run prepare_encoder_locked_inputs.py first.")

    if args.link_baselines:
        for representation in representations:
            print(f"{representation}: {link_baselines(representation, args.fm_root, args.protocol_path)} symlink(s) created")
        return

    jobs = [
        job
        for representation in representations
        for matrix in matrices
        for job in build_jobs(matrix, representation, args.protocol_path, args.fm_root)
    ]
    if args.status or not args.execute:
        print_status(jobs)
        if not args.execute:
            if not args.status:
                pending = [job for job in jobs if job.state() != "complete"]
                cpu = sum(job.hours for job in pending if job.matrix == "linear")
                gpu = sum(job.hours for job in pending if job.matrix == "mlp")
                longest = max((job.hours for job in pending), default=0.0)
                print(
                    f"Plan: {len(pending)} runs to do. Estimate from Neuro-JEPA timings: linear ~{cpu:.0f} CPU-h "
                    f"(~{max(cpu / args.workers, longest):.0f} h wall on {args.workers} workers, at least the longest single run), "
                    f"MLP ~{gpu:.0f} GPU-h. Add --execute to start."
                )
            return

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    failed = execute(jobs, args.workers, gpus, args.fm_root)
    if failed:
        print(f"{len(failed)} run(s) failed; logs under logs/fm_encoders/<representation>/. Rerun the same command to retry.")
        sys.exit(1)
    print("All runs complete; rewriting endpoint run indexes (every run skips)...", flush=True)
    finalize_run_indexes(jobs)
    print_status(jobs)


if __name__ == "__main__":
    main()
