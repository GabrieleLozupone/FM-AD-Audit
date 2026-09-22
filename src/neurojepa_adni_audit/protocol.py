from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplicationSpec:
    name: str
    split_seed: int
    analysis_role: str


@dataclass(frozen=True)
class Protocol:
    path: Path
    payload: dict[str, Any]
    sha256: str
    replications: tuple[ReplicationSpec, ...]

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def split_ratios(self) -> dict[str, float]:
        ratios = self.payload["split_ratios"]
        return {
            "train": float(ratios["train"]),
            "val": float(ratios["validation"]),
            "test": float(ratios["test"]),
        }

    def replication(self, name: str) -> ReplicationSpec:
        for replication in self.replications:
            if replication.name == name:
                return replication
        raise ValueError(f"Unknown replication `{name}`.")

    def accepts_artifact_protocol(self, sha256: str, artifact: str) -> bool:
        """Return whether an immutable artifact is valid for this protocol release."""
        if sha256 == self.sha256:
            return True
        compatibility = self.payload.get("artifact_compatibility", {})
        for parent in compatibility.get("accepted_parent_protocols", []):
            if str(parent.get("sha256")) == str(sha256) and artifact in parent.get("artifacts", []):
                return True
        return False

    def require_artifact_protocol(self, sha256: str | None, artifact: str) -> None:
        if not sha256 or not self.accepts_artifact_protocol(str(sha256), artifact):
            raise ValueError(
                f"{artifact} protocol hash `{sha256}` is not compatible with "
                f"protocol {self.payload['protocol_version']} ({self.sha256})."
            )


def _require_keys(mapping: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise ValueError(f"{label} is missing required keys: {', '.join(missing)}")


def validate_protocol(payload: dict[str, Any], *, require_frozen: bool = False) -> None:
    _require_keys(
        payload,
        {
            "protocol_version",
            "status",
            "replications",
            "split_ratios",
            "bootstrap",
            "ridge_alphas",
            "logistic_c_values",
            "cox_penalizers",
            "cox_l1_ratios",
            "feature_groups",
            "confidence_interval",
            "model_selection",
            "classification_reporting",
            "encoder",
            "horizons",
            "anchor_policies",
            "fdr_families",
            "replication_synthesis",
            "partitioning",
        },
        "Protocol",
    )
    if payload["status"] not in {"draft", "frozen"}:
        raise ValueError("Protocol status must be `draft` or `frozen`.")
    if require_frozen and payload["status"] != "frozen":
        raise ValueError("Confirmatory artifact generation requires a frozen protocol.")

    replications = payload["replications"]
    if not isinstance(replications, list) or not replications:
        raise ValueError("Protocol must define at least one locked analysis run.")
    names: list[str] = []
    seeds: list[int] = []
    for index, replication in enumerate(replications):
        if not isinstance(replication, dict):
            raise ValueError(f"Replication {index} must be an object.")
        _require_keys(replication, {"name", "split_seed", "analysis_role"}, f"Replication {index}")
        names.append(str(replication["name"]))
        seeds.append(int(replication["split_seed"]))
    if len(names) != len(set(names)):
        raise ValueError("Replication names must be unique.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Replication split seeds must be unique.")

    partitioning = payload["partitioning"]
    _require_keys(partitioning, {"outer_folds", "inner_folds"}, "partitioning")
    if int(partitioning["outer_folds"]) < 2 or int(partitioning["inner_folds"]) < 2:
        raise ValueError("Nested cross-validation requires at least two outer and inner folds.")

    ratios = payload["split_ratios"]
    _require_keys(ratios, {"train", "validation", "test"}, "split_ratios")
    ratio_values = [float(ratios[key]) for key in ("train", "validation", "test")]
    if min(ratio_values) <= 0 or abs(sum(ratio_values) - 1.0) > 1e-8:
        raise ValueError("Split ratios must be positive and sum to 1.0.")

    bootstrap = payload["bootstrap"]
    _require_keys(bootstrap, {"unit", "replicates"}, "bootstrap")
    if bootstrap["unit"] != "participant" or int(bootstrap["replicates"]) < 1000:
        raise ValueError("Bootstrap must use participants and at least 1000 replicates.")

    for grid_name in ("ridge_alphas", "logistic_c_values", "cox_penalizers"):
        grid = payload[grid_name]
        if not isinstance(grid, list) or not grid or any(float(value) <= 0 for value in grid):
            raise ValueError(f"{grid_name} must be a non-empty positive grid.")
    l1_ratios = payload["cox_l1_ratios"]
    if not isinstance(l1_ratios, list) or not l1_ratios or any(not 0 <= float(value) <= 1 for value in l1_ratios):
        raise ValueError("cox_l1_ratios must be a non-empty grid within [0, 1].")

    if "endpoints" in payload and "core" in payload["endpoints"]:
        endpoints = payload["endpoints"]
        _require_keys(endpoints, {"framework", "comparison", "core"}, "endpoints")
        if not isinstance(endpoints["core"], list) or not endpoints["core"]:
            raise ValueError("endpoints.core must contain at least one prespecified endpoint.")
        for index, endpoint in enumerate(endpoints["core"]):
            _require_keys(
                endpoint,
                {"domain", "task", "target", "metric", "favorable_direction"},
                f"endpoints.core[{index}]",
            )
    elif "primary_endpoint" in payload:
        endpoint = payload["primary_endpoint"]
        _require_keys(endpoint, {"task", "target", "metric", "comparison"}, "primary_endpoint")
    else:
        raise ValueError("Protocol must define either `endpoints.core` or `primary_endpoint`.")
    required_groups = {"conventional_full", "neurojepa_conventional"}
    if not required_groups.issubset(set(payload["feature_groups"])):
        raise ValueError("Primary conventional and Neuro-JEPA feature groups are required.")

    nonlinear = payload.get("nonlinear_decoder_analysis")
    if nonlinear is not None:
        _require_keys(
            nonlinear,
            {
                "status",
                "decoder",
                "required_feature_groups",
                "descriptive_feature_groups",
                "endpoints",
                "architecture",
                "optimization",
                "selection",
                "reporting",
            },
            "nonlinear_decoder_analysis",
        )
        endpoints = nonlinear["endpoints"]
        if not isinstance(endpoints, list) or len(endpoints) != 10:
            raise ValueError("nonlinear_decoder_analysis must define exactly 10 tabular endpoints.")
        nonlinear_ids = []
        for index, endpoint in enumerate(endpoints):
            _require_keys(
                endpoint,
                {
                    "id",
                    "base_endpoint_id",
                    "task",
                    "target",
                    "metric",
                    "favorable_direction",
                },
                f"nonlinear_decoder_analysis.endpoints[{index}]",
            )
            if endpoint["task"] == "first_mci_survival":
                raise ValueError("The locked nonlinear MLP analysis must exclude survival.")
            nonlinear_ids.append(str(endpoint["id"]))
        if len(nonlinear_ids) != len(set(nonlinear_ids)):
            raise ValueError("Nonlinear endpoint IDs must be unique.")
        nonlinear_groups = set(nonlinear["required_feature_groups"])
        if nonlinear_groups != required_groups:
            raise ValueError(
                "The nonlinear paired comparison must use conventional_full and "
                "neurojepa_conventional."
            )
        all_nonlinear_groups = nonlinear_groups | set(nonlinear["descriptive_feature_groups"])
        if not all_nonlinear_groups.issubset(set(payload["feature_groups"])):
            raise ValueError("Nonlinear feature groups must be protocol feature groups.")
        optimization = nonlinear["optimization"]
        _require_keys(
            optimization,
            {
                "learning_rate",
                "weight_decay_grid",
                "batch_size",
                "epochs",
                "gradient_clip_norm",
                "early_stopping",
            },
            "nonlinear_decoder_analysis.optimization",
        )
        for name in ("learning_rate", "batch_size", "epochs", "gradient_clip_norm"):
            if float(optimization[name]) <= 0:
                raise ValueError(f"Nonlinear optimization `{name}` must be positive.")
        grid = optimization["weight_decay_grid"]
        if not isinstance(grid, list) or not grid or any(float(value) <= 0 for value in grid):
            raise ValueError("Nonlinear weight_decay_grid must be non-empty and positive.")
        family = str(nonlinear["reporting"]["family"])
        if set(payload.get("fdr_families", {}).get(family, [])) != set(nonlinear_ids):
            raise ValueError("The nonlinear endpoint list must exactly match its FDR family.")

    compatibility = payload.get("artifact_compatibility")
    if compatibility is not None:
        _require_keys(
            compatibility,
            {"policy", "accepted_parent_protocols"},
            "artifact_compatibility",
        )
        for index, parent in enumerate(compatibility["accepted_parent_protocols"]):
            _require_keys(
                parent,
                {"version", "sha256", "artifacts"},
                f"artifact_compatibility.accepted_parent_protocols[{index}]",
            )
            if len(str(parent["sha256"])) != 64:
                raise ValueError("Compatible parent protocols require a SHA-256 digest.")
            if not parent["artifacts"]:
                raise ValueError("Compatible parent protocols must name reusable artifacts.")

    if payload["status"] == "frozen":
        if not payload.get("frozen_at"):
            raise ValueError("A frozen protocol must record `frozen_at`.")
        if not payload.get("approved_by"):
            raise ValueError("A frozen protocol must record at least one approver.")
        encoder = payload["encoder"]
        _require_keys(
            encoder,
            {
                "hf_model_id",
                "hf_revision",
                "checkpoint_filename",
                "checkpoint_sha256",
                "code_commit",
                "frozen",
                "representation",
            },
            "encoder",
        )
        if not encoder["hf_revision"]:
            raise ValueError("A frozen protocol must pin `encoder.hf_revision`.")
        if len(str(encoder["checkpoint_sha256"])) != 64:
            raise ValueError("A frozen protocol must pin a SHA-256 checkpoint digest.")
        if encoder["frozen"] is not True:
            raise ValueError("The confirmatory protocol requires a frozen encoder.")
        if not payload["fdr_families"]:
            raise ValueError("A frozen protocol must define FDR families.")


def load_protocol(path: Path, *, require_frozen: bool = False) -> Protocol:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Protocol root must be a JSON object.")
    validate_protocol(payload, require_frozen=require_frozen)
    replications = tuple(
        ReplicationSpec(
            name=str(item["name"]),
            split_seed=int(item["split_seed"]),
            analysis_role=str(item["analysis_role"]),
        )
        for item in payload["replications"]
    )
    return Protocol(
        path=path.resolve(),
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
        replications=replications,
    )
