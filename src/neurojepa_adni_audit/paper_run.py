from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from neurojepa_adni_audit.protocol import Protocol, ReplicationSpec, load_protocol
from neurojepa_adni_audit.replications import sha256_file


@dataclass(frozen=True)
class PaperRunContext:
    protocol: Protocol
    replication: ReplicationSpec
    provenance_path: Path

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_path": str(self.protocol.path),
            "protocol_sha256": self.protocol.sha256,
            "protocol_version": self.protocol.payload["protocol_version"],
            "protocol_status": self.protocol.status,
            "replication": self.replication.name,
            "analysis_role": self.replication.analysis_role,
            "split_seed": self.replication.split_seed,
            "replication_provenance_path": str(self.provenance_path.resolve()),
            "replication_provenance_sha256": sha256_file(self.provenance_path),
        }


def _same_float_grid(observed: list[float], expected: list[float]) -> bool:
    return [float(value) for value in observed] == [float(value) for value in expected]


def require_protocol_grid(observed: list[float], expected: list[float], label: str) -> None:
    if not _same_float_grid(observed, expected):
        raise ValueError(f"{label} does not match the frozen protocol: observed={observed}, expected={expected}")


def bind_paper_run(
    *,
    protocol_path: Path,
    replication_name: str,
    manifest_path: Path,
    embedding_index_path: Path,
    allow_draft_protocol: bool = False,
) -> PaperRunContext:
    protocol = load_protocol(protocol_path, require_frozen=not allow_draft_protocol)
    replication = protocol.replication(replication_name)
    manifest = pd.read_csv(manifest_path, usecols=["subject_split_seed"], low_memory=False)
    seeds = {int(value) for value in manifest["subject_split_seed"].dropna().unique()}
    if seeds != {replication.split_seed}:
        raise ValueError(
            f"Manifest split seed does not match {replication.name}: observed={sorted(seeds)}, "
            f"expected={replication.split_seed}"
        )
    if manifest_path.parent.resolve() != embedding_index_path.parent.resolve():
        raise ValueError("Paper manifest and embedding index must come from the same replication directory.")
    provenance_path = manifest_path.parent / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Replication provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("Replication provenance was generated from a different protocol hash.")
    if provenance.get("replication", {}).get("name") != replication.name:
        raise ValueError("Replication provenance name does not match the requested replication.")
    for path in (manifest_path, embedding_index_path):
        recorded = provenance.get("outputs", {}).get(path.name, {}).get("sha256")
        if recorded is None or recorded != sha256_file(path):
            raise ValueError(f"Replication input hash mismatch for {path.name}.")
    return PaperRunContext(protocol, replication, provenance_path)


def bind_paper_table_run(
    *,
    protocol_path: Path,
    replication_name: str,
    table_path: Path,
    allow_draft_protocol: bool = False,
) -> PaperRunContext:
    protocol = load_protocol(protocol_path, require_frozen=not allow_draft_protocol)
    replication = protocol.replication(replication_name)
    table = pd.read_csv(table_path, usecols=["subject_split_seed"], low_memory=False)
    seeds = {int(value) for value in table["subject_split_seed"].dropna().unique()}
    if seeds != {replication.split_seed}:
        raise ValueError(
            f"Table split seed does not match {replication.name}: observed={sorted(seeds)}, "
            f"expected={replication.split_seed}"
        )
    provenance_path = table_path.parent / "provenance.json"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Replication provenance is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("Replication provenance was generated from a different protocol hash.")
    if provenance.get("replication", {}).get("name") != replication.name:
        raise ValueError("Replication provenance name does not match the requested replication.")
    recorded = provenance.get("outputs", {}).get(table_path.name, {}).get("sha256")
    if recorded is None or recorded != sha256_file(table_path):
        raise ValueError(f"Replication input hash mismatch for {table_path.name}.")
    return PaperRunContext(protocol, replication, provenance_path)
