"""Encoder-specific copies of the 0.4.0 locked nested-CV inputs (protocol 0.6.0).

The task design (manifest, pairs, subjects, folds) is copied byte for byte. Only the
encoder-describing columns of `embedding_index.csv` and `horizon_matched_pairs.csv`
are rewritten, at the CSV-text level so that every other cell keeps its exact bytes.
`verify_task_tables_equivalent` then loads every locked endpoint through the runners'
own loaders from both directories and requires identical rows.
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from neurojepa_adni_audit.encoders.common import KEY_COLUMNS, POOLED_DIMENSION
from neurojepa_adni_audit.replications import runtime_provenance, sha256_file


BYTE_IDENTICAL_FILES = (
    "manifest.csv",
    "progression_pairs.csv",
    "subjects.csv",
    "nested_fold_assignments.csv",
)
REWRITTEN_FILES = ("embedding_index.csv", "horizon_matched_pairs.csv")
INTEGRITY_FILES = {
    "embedding_integrity_summary.json": "summary.json",
    "pooled_embeddings_integrity.csv": "pooled_embeddings.csv",
}
# Columns that describe the embedding rather than the scan or the task. Everything else
# must stay byte-identical. Token columns are blanked so no code can reach Neuro-JEPA
# token files through an encoder input directory.
INDEX_ENCODER_COLUMNS = (
    "pooled_embedding_path",
    "token_embedding_path",
    "model_name_or_path",
    "checkpoint_filename",
    "feature_kind",
    "save_dtype",
    "n_tokens",
    "token_dim",
)
PAIR_PREFIXES = ("current", "next")
PAIR_ENCODER_COLUMNS = tuple(
    f"{prefix}_{column}"
    for prefix in PAIR_PREFIXES
    for column in ("pooled_embedding_path", "token_embedding_path", "n_tokens", "token_dim")
)
ENCODER_COLUMN_SUFFIXES = (*INDEX_ENCODER_COLUMNS, "pooled_dim")
REPRESENTATIONS = ("dinov3", "brainiac", "brainiac_token0")

# The 21 tabular endpoint loads of run_locked_experiment_matrix.py plus survival; the
# nonlinear matrix uses a subset of the same loads.
LOCKED_TABLE_LOADS: tuple[tuple[str, str | None, int | None], ...] = (
    ("ad_vs_cn", None, None),
    ("mci_conversion_36m", None, None),
    *(("same_visit_cognition", t, None) for t in ("adas13_at_session", "adas11_at_session", "mmse_at_session")),
    *(
        ("future_cognition", t, h)
        for t in ("adas13_at_session", "adas11_at_session", "mmse_at_session")
        for h in (12, 24, 36)
    ),
    *(
        ("future_morphometry", t, h)
        for t in ("hippocampus_total_icv", "lateral_ventricles_total_icv")
        for h in (12, 24, 36)
    ),
    ("survival", None, None),
)


def _read_csv_text(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header, rows


def _csv_bytes(header: list[str], rows: Iterator[list[str]] | list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def rewrite_csv_columns(
    path: Path,
    columns: tuple[str, ...],
    update: Callable[[dict[str, str]], dict[str, str]],
) -> bytes:
    """Return `path` with only `columns` changed by `update`; every other cell keeps its text.

    Refuses to proceed unless rewriting the file with no change reproduces it byte for
    byte, which proves the CSV dialect round-trips exactly.
    """
    original = path.read_bytes()
    header, rows = _read_csv_text(path)
    if _csv_bytes(header, rows) != original:
        raise ValueError(f"{path} does not round-trip through the CSV writer; refusing a text-level rewrite.")
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"{path} lacks encoder columns: {missing}")
    positions = {column: header.index(column) for column in columns}
    rewritten = []
    for row in rows:
        record = dict(zip(header, row))
        changes = update(record)
        unexpected = set(changes) - set(columns)
        if unexpected:
            raise ValueError(f"Update touched non-encoder columns: {sorted(unexpected)}")
        new_row = list(row)
        for column, value in changes.items():
            new_row[positions[column]] = value
        rewritten.append(new_row)
    return _csv_bytes(header, rewritten)


def assert_only_columns_changed(before: bytes, after: bytes, columns: tuple[str, ...], label: str) -> int:
    """Cell-by-cell check that `after` differs from `before` only inside `columns`."""
    old = list(csv.reader(io.StringIO(before.decode("utf-8"))))
    new = list(csv.reader(io.StringIO(after.decode("utf-8"))))
    if old[0] != new[0] or len(old) != len(new):
        raise ValueError(f"{label}: header or row count changed.")
    allowed = {old[0].index(column) for column in columns}
    for line, (a, b) in enumerate(zip(old[1:], new[1:]), start=2):
        if len(a) != len(b):
            raise ValueError(f"{label}: line {line} changed width.")
        for position, (x, y) in enumerate(zip(a, b)):
            if x != y and position not in allowed:
                raise ValueError(f"{label}: line {line} column `{old[0][position]}` changed outside the encoder columns.")
    return len(old) - 1


def _key(record: dict[str, str], prefix: str = "") -> tuple[str, str, str]:
    if not prefix:
        return tuple(record[column] for column in KEY_COLUMNS)  # type: ignore[return-value]
    return (record["participant_id"], record[f"{prefix}_session_id"], record[f"{prefix}_image_id"])


def load_encoder_index(encoder_root: Path, representation: str) -> pd.DataFrame:
    index = pd.read_csv(encoder_root / "embedding_index.csv", dtype=str, keep_default_na=False)
    if set(index["representation"].str.split(":").str[0]) != {representation}:
        raise ValueError(f"{encoder_root} index does not hold representation `{representation}` only.")
    if index.duplicated(list(KEY_COLUMNS)).any() or index["pooled_embedding_path"].duplicated().any():
        raise ValueError(f"{encoder_root} index has duplicate keys or paths.")
    return index


def make_pair_update(
    old_path_key: dict[str, tuple[str, str, str]], new_path: dict[tuple[str, str, str], str]
) -> Callable[[dict[str, str]], dict[str, str]]:
    """Map each pair side from its Neuro-JEPA file to the encoder file of the same scan."""

    def update(record: dict[str, str]) -> dict[str, str]:
        changes: dict[str, str] = {}
        for prefix in PAIR_PREFIXES:
            old = record[f"{prefix}_pooled_embedding_path"]
            if old == "":
                changes[f"{prefix}_pooled_embedding_path"] = ""
            else:
                key = old_path_key.get(old)
                if key is None or key != _key(record, prefix):
                    raise ValueError(f"Pair row `{prefix}` embedding does not match its own scan key: {old}")
                changes[f"{prefix}_pooled_embedding_path"] = new_path[key]
            if record[f"{prefix}_pooled_dim"] not in ("", str(POOLED_DIMENSION)):
                raise ValueError(f"Pair row `{prefix}_pooled_dim` is not 768.")
            changes[f"{prefix}_token_embedding_path"] = ""
            changes[f"{prefix}_n_tokens"] = ""
            changes[f"{prefix}_token_dim"] = ""
        return changes

    return update


def _validated_integrity(encoder_root: Path, encoder_index_path: Path, parent_dir: Path) -> dict[str, Any]:
    integrity_dir = encoder_root / "embedding_integrity"
    summary = json.loads((integrity_dir / "summary.json").read_text(encoding="utf-8"))
    checks = {
        "status_valid": summary.get("status") == "valid",
        "all_checks_true": bool(summary.get("checks")) and all(summary["checks"].values()),
        "expected_shape": summary.get("expected_pooled_shape") == [POOLED_DIMENSION],
        "records_hash": summary.get("pooled_file_manifest", {}).get("sha256")
        == sha256_file(integrity_dir / "pooled_embeddings.csv"),
        "index_unchanged_since_integrity": summary.get("embedding_index", {}).get("sha256")
        == sha256_file(encoder_index_path),
        "source_manifest_is_locked_manifest": summary.get("source_manifest", {}).get("sha256")
        == sha256_file(parent_dir / "manifest.csv"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Embedding integrity for {encoder_root.name} failed: {failed}")
    return summary


def _verify_parent(parent_dir: Path) -> dict[str, Any]:
    provenance = json.loads((parent_dir / "provenance.json").read_text(encoding="utf-8"))
    for name, record in provenance["outputs"].items():
        if sha256_file(parent_dir / name) != record["sha256"]:
            raise ValueError(f"Parent locked input {name} no longer matches its provenance hash.")
    if provenance.get("protocol", {}).get("status") != "frozen":
        raise ValueError("Parent locked inputs were not generated under a frozen protocol.")
    return provenance


def _encoder_metadata(encoder_root: Path, integrity_dir: Path) -> dict[str, str]:
    summary = json.loads((encoder_root / "embedding_summary.json").read_text(encoding="utf-8"))
    weights = summary.get("weights_path") or summary.get("checkpoint_path")
    dtypes = set(pd.read_csv(integrity_dir / "pooled_embeddings.csv", dtype=str)["pooled_embedding_dtype"])
    if len(dtypes) != 1:
        raise ValueError(f"{encoder_root.name}: mixed embedding dtypes {sorted(dtypes)}")
    return {
        "model_name_or_path": summary.get("hf_model_id") or str(weights),
        "checkpoint_filename": Path(str(weights)).name,
        "feature_kind": "pooled",
        "save_dtype": dtypes.pop(),
        "n_tokens": "",
        "token_dim": "",
    }


def build_encoder_locked_inputs(
    *,
    representation: str,
    parent_dir: Path,
    encoder_root: Path,
    protocol_path: Path,
    project_root: Path,
    output_dir: Path | None = None,
) -> Path:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation `{representation}`; expected one of {REPRESENTATIONS}.")
    output_dir = output_dir or encoder_root / "locked_inputs"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite encoder locked inputs: {output_dir}")
    parent_provenance = _verify_parent(parent_dir)
    encoder_index_path = encoder_root / "embedding_index.csv"
    integrity = _validated_integrity(encoder_root, encoder_index_path, parent_dir)
    encoder_index = load_encoder_index(encoder_root, representation)
    metadata = _encoder_metadata(encoder_root, encoder_root / "embedding_integrity")

    parent_index = pd.read_csv(parent_dir / "embedding_index.csv", dtype=str, keep_default_na=False)
    parent_keys = parent_index[list(KEY_COLUMNS)].apply(tuple, axis=1)
    encoder_keys = encoder_index[list(KEY_COLUMNS)].apply(tuple, axis=1)
    if parent_keys.duplicated().any() or set(parent_keys) != set(encoder_keys) or len(parent_keys) != len(encoder_keys):
        raise ValueError("Encoder index keys do not match the locked embedding index one to one.")
    new_path = dict(zip(encoder_keys, encoder_index["pooled_embedding_path"]))
    new_source = dict(zip(encoder_keys, encoder_index["source_image_path"]))
    if any(new_source[key] != source for key, source in zip(parent_keys, parent_index["source_image_path"])):
        raise ValueError("Encoder embeddings were extracted from different image files than the locked index.")
    old_path_key = dict(zip(parent_index["pooled_embedding_path"], parent_keys))
    if len(old_path_key) != len(parent_index):
        raise ValueError("Locked index has duplicate pooled paths.")
    if set(parent_index["pooled_dim"]) != {str(POOLED_DIMENSION)}:
        raise ValueError("Locked index pooled_dim is not uniformly 768.")

    def update_index(record: dict[str, str]) -> dict[str, str]:
        return {
            "pooled_embedding_path": new_path[_key(record)],
            "token_embedding_path": "",
            **metadata,
        }

    rewrites = {
        "embedding_index.csv": (INDEX_ENCODER_COLUMNS, update_index),
        "horizon_matched_pairs.csv": (PAIR_ENCODER_COLUMNS, make_pair_update(old_path_key, new_path)),
    }
    rewrite_report: dict[str, Any] = {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent))
    try:
        for name in BYTE_IDENTICAL_FILES:
            shutil.copy2(parent_dir / name, temporary / name)
            if sha256_file(temporary / name) != parent_provenance["outputs"][name]["sha256"]:
                raise ValueError(f"Byte-identical copy of {name} failed.")
        for name, (columns, update) in rewrites.items():
            before = (parent_dir / name).read_bytes()
            after = rewrite_csv_columns(parent_dir / name, columns, update)
            rows = assert_only_columns_changed(before, after, columns, name)
            (temporary / name).write_bytes(after)
            rewrite_report[name] = {"rows": rows, "rewritten_columns": list(columns), "other_cells_identical": True}
        for target, source in INTEGRITY_FILES.items():
            shutil.copy2(encoder_root / "embedding_integrity" / source, temporary / target)

        equivalence = verify_task_tables_equivalent(parent_dir, temporary, set(encoder_index["pooled_embedding_path"]))
        amendment = json.loads(protocol_path.read_text(encoding="utf-8"))
        provenance = {
            "schema_version": parent_provenance["schema_version"],
            # The folds and task tables were generated under 0.4.0 and are reused unchanged,
            # so the parent protocol block is carried over verbatim (accepted by 0.6.0 for
            # `nested_cv_inputs`). The amendment in force at build time is recorded below.
            "protocol": parent_provenance["protocol"],
            "design": parent_provenance["design"],
            "counts": parent_provenance["counts"],
            "sources": parent_provenance["sources"],
            "embedding_integrity": integrity,
            "encoder_inputs": {
                "representation": representation,
                "encoder_root": str(encoder_root.resolve()),
                "embedding_index": {"path": str(encoder_index_path.resolve()), "sha256": sha256_file(encoder_index_path)},
                "embedding_summary": {
                    "path": str((encoder_root / "embedding_summary.json").resolve()),
                    "sha256": sha256_file(encoder_root / "embedding_summary.json"),
                },
                "encoder_metadata_written": metadata,
                "parent_locked_inputs": {
                    "path": str(parent_dir.resolve()),
                    "provenance_sha256": sha256_file(parent_dir / "provenance.json"),
                },
                "amendment_protocol_at_build": {
                    "path": str(protocol_path.resolve()),
                    "sha256": sha256_file(protocol_path),
                    "version": amendment.get("protocol_version"),
                    "status": amendment.get("status"),
                },
                "byte_identical_files": list(BYTE_IDENTICAL_FILES),
                "rewritten_files": rewrite_report,
                "table_load_equivalence": equivalence,
            },
            "runtime": runtime_provenance(project_root),
        }
        provenance["outputs"] = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def load_locked_table(input_dir: Path, task: str, target: str | None, horizon: int | None) -> pd.DataFrame:
    """Load one endpoint table exactly as the locked runners do, embedding filter included."""
    from neurojepa_adni_audit.features.groups import add_morphometry_ratios
    from neurojepa_adni_audit.probes.cognitive import load_cognitive_task_table, load_future_change_task_table
    from neurojepa_adni_audit.probes.downstream import load_task_table, validate_feature_paths
    from neurojepa_adni_audit.survival.probe import load_survival_task_table
    from neurojepa_adni_audit.tables.pairs import enrich_horizon_pairs
    from neurojepa_adni_audit.transitions.future import make_regression_values

    manifest_path = input_dir / "manifest.csv"
    index_path = input_dir / "embedding_index.csv"
    if task == "survival":
        table = load_survival_task_table(manifest_path, index_path, mci_anchor_policy="first")
        return validate_feature_paths(table, "pooled")
    if task == "same_visit_cognition":
        table = load_cognitive_task_table(manifest_path, index_path, target, row_policy="all_sessions")
    elif task == "future_cognition":
        table = load_future_change_task_table(
            manifest_path, index_path, target, row_policy="all_sessions",
            horizon_months=horizon, tolerance_months=3.0,
        )
    elif task == "future_morphometry":
        manifest = pd.read_csv(manifest_path, low_memory=False)
        pairs = enrich_horizon_pairs(pd.read_csv(input_dir / "horizon_matched_pairs.csv", low_memory=False), manifest)
        pairs = pairs[pd.to_numeric(pairs["requested_horizon_months"], errors="coerce") == float(horizon)].copy()
        table = add_morphometry_ratios(make_regression_values(pairs, target), prefix="current_")
        return table[
            table["current_pooled_embedding_path"].notna()
            & table["current_pooled_embedding_path"].map(lambda value: Path(str(value)).is_file())
        ].reset_index(drop=True)
    else:
        table = load_task_table(
            manifest_path, index_path, task, mci_anchor_policy="first" if task == "mci_conversion_36m" else "all"
        )
    return validate_feature_paths(add_morphometry_ratios(table), "pooled")


def _is_encoder_column(column: str) -> bool:
    return any(column == suffix or column.endswith(f"_{suffix}") for suffix in ENCODER_COLUMN_SUFFIXES)


def verify_task_tables_equivalent(parent_dir: Path, encoder_dir: Path, encoder_paths: set[str]) -> dict[str, Any]:
    """Every locked endpoint must load the same rows, in the same order, with the same values.

    Only encoder-describing columns may differ, and each pooled path must move from the
    Neuro-JEPA file of a scan to the encoder file of the same scan.
    """
    parent_index = pd.read_csv(parent_dir / "embedding_index.csv", dtype=str, keep_default_na=False)
    encoder_index = pd.read_csv(encoder_dir / "embedding_index.csv", dtype=str, keep_default_na=False)
    old_key = dict(zip(parent_index["pooled_embedding_path"], parent_index[list(KEY_COLUMNS)].apply(tuple, axis=1)))
    new_key = dict(zip(encoder_index["pooled_embedding_path"], encoder_index[list(KEY_COLUMNS)].apply(tuple, axis=1)))
    report: dict[str, Any] = {}
    for task, target, horizon in LOCKED_TABLE_LOADS:
        label = "/".join(str(part) for part in (task, target, horizon) if part is not None)
        before = load_locked_table(parent_dir, task, target, horizon)
        after = load_locked_table(encoder_dir, task, target, horizon)
        if list(before.columns) != list(after.columns) or len(before) != len(after):
            raise ValueError(f"{label}: table shape or columns differ ({before.shape} vs {after.shape}).")
        shared = [column for column in before.columns if not _is_encoder_column(column)]
        pd.testing.assert_frame_equal(
            before[shared].reset_index(drop=True), after[shared].reset_index(drop=True), check_exact=True, obj=label
        )
        path_columns = [column for column in before.columns if column.endswith("pooled_embedding_path")]
        if not path_columns:
            raise ValueError(f"{label}: no pooled embedding path column in the loaded table.")
        for column in path_columns:
            for old, new in zip(before[column], after[column]):
                if pd.isna(old) and pd.isna(new):
                    continue
                if new not in encoder_paths or old_key.get(old) is None or old_key[old] != new_key.get(new):
                    raise ValueError(f"{label}: `{column}` does not map a scan to its own encoder embedding.")
        report[label] = {
            "rows": int(len(after)),
            "participants": int(after["participant_id"].nunique()),
            "identical_non_embedding_columns": len(shared),
            "embedding_path_columns_checked": path_columns,
        }
    return report
