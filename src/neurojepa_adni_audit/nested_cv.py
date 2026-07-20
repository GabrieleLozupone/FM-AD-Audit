from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.protocol import Protocol
from neurojepa_adni_audit.replications import build_subject_table, runtime_provenance, sha256_file
from neurojepa_adni_audit.tables.pairs import enrich_horizon_pairs


NESTED_STRATEGY = "participant_level_stratified_repeated_nested_cross_validation"


@dataclass(frozen=True)
class NestedFoldSpec:
    repeat: str
    outer_fold: int
    inner_fold: int | None = None


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _hash_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_stratified_folds(
    subjects: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    stratum_column: str = "split_stratum",
) -> pd.Series:
    required = {"participant_id", stratum_column}
    missing = sorted(required - set(subjects.columns))
    if missing:
        raise ValueError(f"Subject table is missing fold columns: {', '.join(missing)}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if subjects["participant_id"].astype(str).duplicated().any():
        raise ValueError("Subject table contains duplicate participant_id values.")
    if len(subjects) < n_splits:
        raise ValueError(f"At least {n_splits} participants are required for {n_splits} folds.")

    assignments = pd.Series(index=subjects.index, dtype="int64")
    for stratum, group in subjects.groupby(stratum_column, sort=True, dropna=False):
        ordered = sorted(
            ((index, str(participant)) for index, participant in zip(group.index, group["participant_id"])),
            key=lambda item: _hash_order(item[1], seed),
        )
        offset = _derived_seed(seed, f"stratum|{stratum}") % n_splits
        for position, (index, _participant) in enumerate(ordered):
            assignments.at[index] = int((position + offset) % n_splits)
    return assignments.astype(int)


def build_nested_fold_assignments(
    subjects: pd.DataFrame,
    protocol: Protocol,
) -> pd.DataFrame:
    partitioning = protocol.payload.get("partitioning", {})
    outer_folds = int(partitioning.get("outer_folds", 0))
    inner_folds = int(partitioning.get("inner_folds", 0))
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Protocol partitioning must define at least two outer and inner folds.")
    required = {"participant_id", "split_stratum"}
    missing = sorted(required - set(subjects.columns))
    if missing:
        raise ValueError(f"Subject table is missing nested-CV columns: {', '.join(missing)}")

    base = subjects.copy()
    base["participant_id"] = base["participant_id"].astype(str)
    outputs: list[pd.DataFrame] = []
    for repeat_index, replication in enumerate(protocol.replications, start=1):
        repeated = base.copy()
        repeated["outer_fold"] = assign_stratified_folds(
            repeated, n_splits=outer_folds, seed=replication.split_seed
        )
        for outer_fold in range(outer_folds):
            fold = repeated.copy()
            fold["repeat"] = replication.name
            fold["repeat_index"] = repeat_index
            fold["repeat_seed"] = int(replication.split_seed)
            fold["outer_role"] = "development"
            fold.loc[fold["outer_fold"] == outer_fold, "outer_role"] = "test"
            fold["selected_outer_fold"] = outer_fold
            fold["inner_validation_fold"] = pd.Series(pd.NA, index=fold.index, dtype="Int64")
            development = fold[fold["outer_role"] == "development"]
            inner_seed = _derived_seed(replication.split_seed, f"outer-fold-{outer_fold}")
            fold.loc[development.index, "inner_validation_fold"] = assign_stratified_folds(
                development, n_splits=inner_folds, seed=inner_seed
            ).astype("Int64")
            fold["inner_seed"] = inner_seed
            fold["outer_folds"] = outer_folds
            fold["inner_folds"] = inner_folds
            fold["nested_cv_strategy"] = NESTED_STRATEGY
            outputs.append(fold)
    assignments = pd.concat(outputs, ignore_index=True)
    validate_nested_fold_assignments(assignments, protocol)
    return assignments.sort_values(
        ["repeat_index", "selected_outer_fold", "participant_id"]
    ).reset_index(drop=True)


def materialize_split(
    assignments: pd.DataFrame,
    spec: NestedFoldSpec,
) -> pd.DataFrame:
    selected = assignments[
        (assignments["repeat"] == spec.repeat)
        & (assignments["selected_outer_fold"] == spec.outer_fold)
    ].copy()
    if selected.empty:
        raise ValueError(f"Unknown repeat/outer fold: {spec.repeat}/{spec.outer_fold}.")
    if spec.inner_fold is None:
        selected["subject_split"] = selected["outer_role"].map(
            {"development": "train", "test": "test"}
        )
    else:
        n_inner = int(selected["inner_folds"].iloc[0])
        if spec.inner_fold < 0 or spec.inner_fold >= n_inner:
            raise ValueError(f"inner_fold must be in [0, {n_inner - 1}].")
        if (selected["outer_role"] == "test").any():
            selected = selected[selected["outer_role"] == "development"].copy()
        selected["subject_split"] = "train"
        selected.loc[selected["inner_validation_fold"] == spec.inner_fold, "subject_split"] = "val"
    return selected.reset_index(drop=True)


def validate_nested_fold_assignments(assignments: pd.DataFrame, protocol: Protocol) -> None:
    outer_folds = int(protocol.payload["partitioning"]["outer_folds"])
    inner_folds = int(protocol.payload["partitioning"]["inner_folds"])
    participant_count = assignments["participant_id"].nunique()
    expected_rows = len(protocol.replications) * outer_folds * participant_count
    if len(assignments) != expected_rows:
        raise ValueError(f"Nested assignments contain {len(assignments)} rows; expected {expected_rows}.")
    for replication in protocol.replications:
        repeat = assignments[assignments["repeat"] == replication.name]
        test_counts = repeat.groupby("participant_id")["outer_role"].apply(lambda x: int((x == "test").sum()))
        if not (test_counts == 1).all():
            raise ValueError(f"Every participant must be outer-test once in repeat {replication.name}.")
        for outer_fold in range(outer_folds):
            fold = repeat[repeat["selected_outer_fold"] == outer_fold]
            test = set(fold.loc[fold["outer_role"] == "test", "participant_id"])
            development = set(fold.loc[fold["outer_role"] == "development", "participant_id"])
            if not test or not development or test & development:
                raise ValueError(f"Invalid outer partition in {replication.name}, fold {outer_fold}.")
            inner_values = fold.loc[fold["outer_role"] == "development", "inner_validation_fold"]
            if inner_values.isna().any() or set(inner_values.astype(int)) != set(range(inner_folds)):
                raise ValueError(f"Invalid inner folds in {replication.name}, outer fold {outer_fold}.")
            if fold.loc[fold["outer_role"] == "test", "inner_validation_fold"].notna().any():
                raise ValueError("Outer-test participants must not receive inner-fold assignments.")


def prepare_nested_cv_artifacts(
    *,
    protocol: Protocol,
    manifest_path: Path,
    embedding_index_path: Path,
    embedding_integrity_dir: Path,
    progression_pairs_path: Path,
    horizon_pairs_path: Path | None,
    output_dir: Path,
    project_root: Path,
    allow_dirty_worktree: bool = False,
) -> Path:
    """Create immutable, provenance-bound inputs for all nested-CV repeats."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite nested-CV artifact directory: {output_dir}")
    sources = {
        "manifest": manifest_path,
        "embedding_index": embedding_index_path,
        "progression_pairs": progression_pairs_path,
    }
    if horizon_pairs_path is not None:
        sources["horizon_matched_pairs"] = horizon_pairs_path
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing nested-CV inputs: {', '.join(missing)}")

    integrity_summary_path = embedding_integrity_dir / "summary.json"
    integrity_records_path = embedding_integrity_dir / "pooled_embeddings.csv"
    if not integrity_summary_path.exists() or not integrity_records_path.exists():
        raise FileNotFoundError(
            f"Complete embedding integrity manifest not found in {embedding_integrity_dir}."
        )
    integrity = json.loads(integrity_summary_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "valid":
        raise ValueError("Embedding integrity manifest does not have status `valid`.")
    if integrity.get("embedding_index", {}).get("sha256") != sha256_file(embedding_index_path):
        raise ValueError("Embedding index hash does not match the integrity manifest.")
    if integrity.get("pooled_file_manifest", {}).get("sha256") != sha256_file(integrity_records_path):
        raise ValueError("Pooled embedding file manifest hash mismatch.")

    runtime = runtime_provenance(project_root)
    if not allow_dirty_worktree:
        if runtime["git_commit"] is None:
            raise ValueError("Nested-CV preparation requires a committed Git revision.")
        if runtime["git_status_porcelain"]:
            raise ValueError("Nested-CV preparation requires a clean Git worktree.")

    manifest = pd.read_csv(manifest_path, dtype={"participant_id": "string"}, low_memory=False)
    progression = pd.read_csv(
        progression_pairs_path, dtype={"participant_id": "string"}, low_memory=False
    )
    horizon = (
        pd.read_csv(horizon_pairs_path, dtype={"participant_id": "string"}, low_memory=False)
        if horizon_pairs_path is not None
        else None
    )
    enriched_horizon = enrich_horizon_pairs(horizon, manifest) if horizon is not None else None
    subjects = build_subject_table(manifest, progression)
    assignments = build_nested_fold_assignments(subjects, protocol)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    try:
        subjects_path = temporary / "subjects.csv"
        assignments_path = temporary / "nested_fold_assignments.csv"
        subjects.to_csv(subjects_path, index=False)
        assignments.to_csv(assignments_path, index=False)
        for name, path in sources.items():
            if name != "horizon_matched_pairs":
                shutil.copy2(path, temporary / f"{name}.csv")
        if enriched_horizon is not None:
            enriched_horizon.to_csv(temporary / "horizon_matched_pairs.csv", index=False)
        shutil.copy2(integrity_summary_path, temporary / "embedding_integrity_summary.json")
        shutil.copy2(integrity_records_path, temporary / "pooled_embeddings_integrity.csv")

        provenance = {
            "schema_version": "1.0",
            "protocol": {
                "path": str(protocol.path.resolve()),
                "sha256": protocol.sha256,
                "version": protocol.payload["protocol_version"],
                "status": protocol.status,
            },
            "design": {
                "repeats": len(protocol.replications),
                "outer_folds": int(protocol.payload["partitioning"]["outer_folds"]),
                "inner_folds": int(protocol.payload["partitioning"]["inner_folds"]),
                "unit": "participant",
                "strategy": NESTED_STRATEGY,
            },
            "sources": {
                name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in sources.items()
            },
            "embedding_integrity": integrity,
            "counts": {
                "participants": int(subjects["participant_id"].nunique()),
                "assignment_rows": int(len(assignments)),
                "horizon_pair_rows": int(len(enriched_horizon)) if enriched_horizon is not None else None,
            },
            "runtime": runtime,
        }
        provenance["outputs"] = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "provenance.json"
        }
        (temporary / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir
