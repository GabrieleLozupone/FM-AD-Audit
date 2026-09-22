#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurojepa_adni_audit.protocol import load_protocol


AUDIT_ROOT = PROJECT_ROOT / "results" / "paper_neurojepa_feature_audit"
DEFAULT_INPUT_DIR = AUDIT_ROOT / "locked_inputs"
DEFAULT_OUTPUT_DIR = AUDIT_ROOT / "nonlinear_experiments"
COMPLETE_FILES = {
    "summary.json",
    "outer_test_predictions.csv",
    "selected_inner_oof_predictions.csv",
    "outer_test_calibration_smooth.csv",
}


def build_commands(
    *,
    protocol_path: Path,
    input_dir: Path,
    output_dir: Path,
    device: str,
    torch_threads: int,
    allow_draft: bool,
) -> list[list[str]]:
    protocol = load_protocol(protocol_path, require_frozen=not allow_draft)
    common = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_nested_nonlinear_probe.py"),
        "--protocol-path",
        str(protocol_path),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--torch-threads",
        str(torch_threads),
        "--resume",
    ]
    if allow_draft:
        common.append("--allow-draft-protocol")
    feature_groups = [
        *protocol.payload["nonlinear_decoder_analysis"]["required_feature_groups"],
        *protocol.payload["nonlinear_decoder_analysis"]["descriptive_feature_groups"],
    ]
    feature_args = [
        argument
        for feature in feature_groups
        for argument in ("--feature-group", feature)
    ]
    commands = []
    for endpoint in protocol.payload["nonlinear_decoder_analysis"]["endpoints"]:
        task = str(endpoint["task"])
        target = str(endpoint["target"])
        command = [*common, *feature_args, "--task", task]
        if task in {"same_visit_cognition", "future_cognition", "future_morphometry"}:
            if task in {"future_cognition", "future_morphometry"}:
                target, suffix = target.rsplit("_", 1)
                command.extend(["--horizon-months", suffix.removesuffix("m")])
            command.extend(["--target", target])
        commands.append(command)
    return commands


def _status(protocol: object, output_dir: Path) -> dict:
    nonlinear = protocol.payload["nonlinear_decoder_analysis"]
    features = [
        *nonlinear["required_feature_groups"],
        *nonlinear["descriptive_feature_groups"],
    ]
    expected_folds = len(protocol.replications) * int(
        protocol.payload["partitioning"]["outer_folds"]
    )
    endpoint_rows = []
    complete_total = 0
    expected_total = 0
    for endpoint in nonlinear["endpoints"]:
        expected = expected_folds * len(features)
        complete = 0
        for feature in features:
            root = output_dir / endpoint["task"] / endpoint["target"] / feature
            for summary in root.glob("outer_repeat_*/outer_*/summary.json"):
                run_dir = summary.parent
                if COMPLETE_FILES.issubset({path.name for path in run_dir.iterdir()}):
                    complete += 1
        endpoint_rows.append(
            {
                "id": endpoint["id"],
                "complete_runs": complete,
                "expected_runs": expected,
            }
        )
        complete_total += complete
        expected_total += expected
    return {
        "protocol_version": protocol.payload["protocol_version"],
        "protocol_sha256": protocol.sha256,
        "complete_runs": complete_total,
        "expected_runs": expected_total,
        "percent_complete": (
            100.0 * complete_total / expected_total if expected_total else 100.0
        ),
        "endpoints": endpoint_rows,
    }


def _write_status(protocol: object, output_dir: Path) -> dict:
    status = _status(protocol, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nonlinear_run_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan, execute, resume, and track the locked ten-endpoint MLP matrix."
    )
    parser.add_argument("--protocol-path", type=Path, default=PROJECT_ROOT / "configs/protocol.json")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent endpoint processes for CPU execution; use one worker on GPU.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-draft-protocol", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.torch_threads < 1:
        raise ValueError("--workers and --torch-threads must be positive.")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("Concurrent workers require --device cpu; use one worker for auto/CUDA.")
    protocol = load_protocol(
        args.protocol_path,
        require_frozen=not args.allow_draft_protocol,
    )
    status = _write_status(protocol, args.output_dir)
    print(
        f"MLP matrix {status['complete_runs']}/{status['expected_runs']} runs "
        f"({status['percent_complete']:.1f}%)",
        flush=True,
    )
    if args.status:
        for endpoint in status["endpoints"]:
            print(
                f"  {endpoint['id']}: "
                f"{endpoint['complete_runs']}/{endpoint['expected_runs']}",
                flush=True,
            )
        if not args.execute:
            return
    commands = build_commands(
        protocol_path=args.protocol_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        torch_threads=args.torch_threads,
        allow_draft=args.allow_draft_protocol,
    )
    if not args.execute:
        print(
            f"Plan: {len(commands)} endpoints; "
            f"{status['expected_runs']} outer feature-group runs. "
            "Add --execute to start.",
            flush=True,
        )
        for index, command in enumerate(commands, start=1):
            print(f"[{index:02d}/{len(commands):02d}] {shlex.join(command)}")
        return

    started = time.monotonic()
    endpoints = [
        str(endpoint["id"])
        for endpoint in protocol.payload["nonlinear_decoder_analysis"]["endpoints"]
    ]
    if args.workers == 1:
        for index, (endpoint_name, command) in enumerate(
            zip(endpoints, commands), start=1
        ):
            print(f"=== [{index:02d}/{len(commands):02d}] {endpoint_name} ===", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            status = _write_status(protocol, args.output_dir)
            print(
                f"PROGRESS {status['complete_runs']}/{status['expected_runs']} "
                f"elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
    else:
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        def run_endpoint(item: tuple[int, str, list[str]]) -> tuple[int, str, float, Path]:
            index, endpoint_name, command = item
            log_path = log_dir / f"{endpoint_name}.log"
            endpoint_started = time.monotonic()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"COMMAND {shlex.join(command)}\n")
                handle.flush()
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            return index, endpoint_name, time.monotonic() - endpoint_started, log_path

        items = [
            (index, endpoint_name, command)
            for index, (endpoint_name, command) in enumerate(
                zip(endpoints, commands), start=1
            )
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for item in items:
                print(
                    f"QUEUE [{item[0]:02d}/{len(items):02d}] {item[1]}",
                    flush=True,
                )
                futures[executor.submit(run_endpoint, item)] = item
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    index, endpoint_name, elapsed, log_path = future.result()
                except Exception:
                    print(
                        f"FAILED [{item[0]:02d}/{len(items):02d}] {item[1]} "
                        f"log={log_dir / (item[1] + '.log')}",
                        flush=True,
                    )
                    raise
                status = _write_status(protocol, args.output_dir)
                print(
                    f"DONE [{index:02d}/{len(items):02d}] {endpoint_name} "
                    f"endpoint_elapsed={elapsed:.0f}s "
                    f"progress={status['complete_runs']}/{status['expected_runs']} "
                    f"log={log_path}",
                    flush=True,
                )
    print(
        f"COMPLETE runs={status['complete_runs']} elapsed={time.monotonic() - started:.0f}s "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
