from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from neurojepa_adni_audit.protocol import Protocol, ReplicationSpec


SPLIT_COLUMNS = (
    "subject_split",
    "subject_split_stratum",
    "subject_split_baseline_diagnosis",
    "subject_split_any_diagnosis_change",
    "subject_split_seed",
    "subject_split_strategy",
)
SPLIT_STRATEGY = "subject_level_stratified_by_baseline_diagnosis_and_any_diagnosis_change"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_non_null(series: pd.Series) -> str:
    values = series.dropna()
    return "NA" if values.empty else str(values.iloc[0])


def _allocate_counts(n_subjects: int, ratios: dict[str, float]) -> dict[str, int]:
    if n_subjects <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if n_subjects < 4:
        small = {1: (1, 0, 0), 2: (1, 1, 0), 3: (2, 1, 0)}[n_subjects]
        return dict(zip(("train", "val", "test"), small))

    target = {split: n_subjects * ratios[split] for split in ("train", "val", "test")}
    counts = {split: max(1, int(target[split])) for split in target}
    while sum(counts.values()) > n_subjects:
        candidates = [split for split in counts if counts[split] > 1]
        counts[max(candidates, key=lambda split: (counts[split] - target[split], counts[split]))] -= 1
    while sum(counts.values()) < n_subjects:
        counts[max(counts, key=lambda split: (target[split] - counts[split], target[split]))] += 1
    return counts


def _participant_hash(participant_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{participant_id}".encode()).hexdigest()


def build_subject_table(manifest: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    required = {"participant_id", "session_id", "session_month", "diagnosis_at_session"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing split columns: {', '.join(missing)}")
    ordered = manifest.copy()
    ordered["participant_id"] = ordered["participant_id"].astype(str)
    ordered["_session_month"] = pd.to_numeric(ordered["session_month"], errors="coerce").fillna(float("inf"))
    ordered = ordered.sort_values(["participant_id", "_session_month", "session_id"])
    subjects = (
        ordered.groupby("participant_id", as_index=False)
        .agg(
            baseline_diagnosis=("diagnosis_at_session", _first_non_null),
            n_sessions=("session_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    if pairs is not None and not pairs.empty and "diagnosis_changed" in pairs:
        work = pairs.copy()
        work["participant_id"] = work["participant_id"].astype(str)
        changed = work.groupby("participant_id", as_index=False).agg(
            any_diagnosis_change=(
                "diagnosis_changed",
                lambda values: bool(values.dropna().astype(bool).any()),
            )
        )
    else:
        changed = ordered.groupby("participant_id", as_index=False).agg(
            any_diagnosis_change=(
                "diagnosis_at_session",
                lambda values: len(pd.unique(values.dropna().astype(str))) > 1,
            )
        )
    subjects = subjects.merge(changed, on="participant_id", how="left")
    subjects["any_diagnosis_change"] = subjects["any_diagnosis_change"].map(
        lambda value: False if pd.isna(value) else bool(value)
    )
    subjects["split_stratum"] = (
        subjects["baseline_diagnosis"].fillna("NA").astype(str)
        + "__chg_"
        + subjects["any_diagnosis_change"].astype(str)
    )
    return subjects


def assign_subject_splits(subjects: pd.DataFrame, ratios: dict[str, float], seed: int) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for _, group in subjects.groupby("split_stratum", sort=True):
        group = group.copy()
        counts = _allocate_counts(len(group), ratios)
        ordered_ids = sorted(group["participant_id"].astype(str), key=lambda value: _participant_hash(value, seed))
        lookup: dict[str, str] = {}
        start = 0
        for split in ("train", "val", "test"):
            stop = start + counts[split]
            lookup.update({participant_id: split for participant_id in ordered_ids[start:stop]})
            start = stop
        group["subject_split"] = group["participant_id"].astype(str).map(lookup)
        groups.append(group)
    assigned = pd.concat(groups, ignore_index=True)
    assigned["subject_split_stratum"] = assigned["split_stratum"]
    assigned["subject_split_baseline_diagnosis"] = assigned["baseline_diagnosis"]
    assigned["subject_split_any_diagnosis_change"] = assigned["any_diagnosis_change"]
    assigned["subject_split_seed"] = int(seed)
    assigned["subject_split_strategy"] = SPLIT_STRATEGY
    return assigned.drop(columns=["baseline_diagnosis", "any_diagnosis_change", "split_stratum"])


def annotate_table(table: pd.DataFrame, subject_splits: pd.DataFrame) -> pd.DataFrame:
    work = table.drop(columns=list(SPLIT_COLUMNS), errors="ignore").copy()
    work["participant_id"] = work["participant_id"].astype(str)
    annotated = work.merge(subject_splits[["participant_id", *SPLIT_COLUMNS]], on="participant_id", how="left")
    if annotated["subject_split"].isna().any():
        missing = int(annotated["subject_split"].isna().sum())
        raise ValueError(f"{missing} rows could not be assigned to a participant split.")
    return annotated


def assert_disjoint_splits(table: pd.DataFrame) -> None:
    sets = {
        split: set(table.loc[table["subject_split"] == split, "participant_id"].astype(str))
        for split in ("train", "val", "test")
    }
    overlaps = {
        f"{left}-{right}": sorted(sets[left] & sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        if sets[left] & sets[right]
    }
    if overlaps:
        raise ValueError(f"Participant overlap detected: {overlaps}")


def _git_value(project_root: Path, arguments: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=project_root, check=True, capture_output=True, text=True
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(project_root: Path) -> dict[str, Any]:
    packages = {}
    for name in ("neurojepa-adni-audit", "neurojepa", "torch", "monai", "huggingface-hub", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_status_porcelain": _git_value(project_root, ["status", "--short"]),
    }


def _table_summary(table: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(table)),
        "participants": int(table["participant_id"].nunique()),
        "by_split": {
            split: {
                "rows": int(len(group)),
                "participants": int(group["participant_id"].nunique()),
            }
            for split, group in table.groupby("subject_split", sort=True)
        },
    }


def prepare_replication(
    *,
    protocol: Protocol,
    replication: ReplicationSpec,
    manifest_path: Path,
    embedding_index_path: Path,
    progression_pairs_path: Path,
    horizon_pairs_path: Path | None,
    output_root: Path,
    project_root: Path,
    allow_dirty_worktree: bool = False,
) -> Path:
    final_dir = output_root / f"{replication.name}_seed_{replication.split_seed}"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable replication directory: {final_dir}")
    inputs = [manifest_path, embedding_index_path, progression_pairs_path]
    if horizon_pairs_path is not None:
        inputs.append(horizon_pairs_path)
    missing_inputs = [str(path) for path in inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing replication inputs: {', '.join(missing_inputs)}")
    runtime = runtime_provenance(project_root)
    if not allow_dirty_worktree:
        if runtime["git_commit"] is None:
            raise ValueError("Confirmatory replication preparation requires a committed Git revision.")
        if runtime["git_status_porcelain"]:
            raise ValueError("Confirmatory replication preparation requires a clean Git worktree.")

    manifest = pd.read_csv(manifest_path, dtype={"participant_id": "string"}, low_memory=False)
    progression = pd.read_csv(progression_pairs_path, dtype={"participant_id": "string"}, low_memory=False)
    horizon = (
        pd.read_csv(horizon_pairs_path, dtype={"participant_id": "string"}, low_memory=False)
        if horizon_pairs_path is not None
        else None
    )
    subjects = build_subject_table(manifest, progression)
    subject_splits = assign_subject_splits(subjects, protocol.split_ratios, replication.split_seed)
    annotated_manifest = annotate_table(manifest, subject_splits)
    annotated_progression = annotate_table(progression, subject_splits)
    annotated_horizon = annotate_table(horizon, subject_splits) if horizon is not None else None
    for table in (subject_splits, annotated_manifest, annotated_progression, annotated_horizon):
        if table is not None:
            assert_disjoint_splits(table)

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{replication.name}_", dir=output_root))
    try:
        files = {
            "subject_splits.csv": subject_splits,
            "manifest.csv": annotated_manifest,
            "progression_pairs.csv": annotated_progression,
        }
        if annotated_horizon is not None:
            files["horizon_matched_pairs.csv"] = annotated_horizon
        for filename, table in files.items():
            table.to_csv(temporary / filename, index=False)
        shutil.copy2(embedding_index_path, temporary / "embedding_index.csv")

        source_paths = {
            "manifest": manifest_path,
            "embedding_index": embedding_index_path,
            "progression_pairs": progression_pairs_path,
        }
        if horizon_pairs_path is not None:
            source_paths["horizon_matched_pairs"] = horizon_pairs_path
        provenance = {
            "protocol": {
                "path": str(protocol.path),
                "sha256": protocol.sha256,
                "version": protocol.payload["protocol_version"],
                "status": protocol.status,
            },
            "encoder": protocol.payload["encoder"],
            "replication": {
                "name": replication.name,
                "split_seed": replication.split_seed,
                "analysis_role": replication.analysis_role,
                "split_strategy": SPLIT_STRATEGY,
                "split_ratios": protocol.split_ratios,
            },
            "sources": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in source_paths.items()
            },
            "tables": {
                "manifest": _table_summary(annotated_manifest),
                "progression_pairs": _table_summary(annotated_progression),
                **({"horizon_matched_pairs": _table_summary(annotated_horizon)} if annotated_horizon is not None else {}),
            },
            "runtime": runtime,
        }
        provenance["outputs"] = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.glob("*.csv"))
        }
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        temporary.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_dir
