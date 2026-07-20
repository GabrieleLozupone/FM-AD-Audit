import json
from pathlib import Path

import pytest

from neurojepa_adni_audit.protocol import load_protocol, validate_protocol


def protocol_payload(status: str = "draft") -> dict:
    return {
        "protocol_version": "test",
        "status": status,
        "frozen_at": "2026-07-20T00:00:00Z" if status == "frozen" else None,
        "approved_by": ["test-author"] if status == "frozen" else [],
        "replications": [
            {"name": "replication_01", "split_seed": 1, "analysis_role": "pilot"},
            {"name": "replication_02", "split_seed": 2, "analysis_role": "confirmation"},
        ],
        "split_ratios": {"train": 0.7, "validation": 0.2, "test": 0.1},
        "bootstrap": {"unit": "participant", "replicates": 1000},
        "confidence_interval": {"method": "percentile", "level": 0.95},
        "ridge_alphas": [0.1, 1.0],
        "logistic_c_values": [0.1, 1.0],
        "cox_penalizers": [0.1, 1.0],
        "cox_l1_ratios": [0.0, 1.0],
        "primary_endpoint": {
            "task": "same_visit_cognition",
            "target": "adas13_at_session",
            "metric": "subject_equal_mae",
            "comparison": "neurojepa_conventional-minus-conventional_full",
        },
        "feature_groups": ["conventional_full", "neurojepa_conventional"],
        "model_selection": {"ridge": "validation_mae"},
        "classification_reporting": {"metrics": ["auroc", "brier_score"]},
        "encoder": {
            "hf_model_id": "test/model",
            "hf_revision": "abc123",
            "checkpoint_filename": "model.safetensors",
            "checkpoint_sha256": "a" * 64,
            "code_commit": "def456",
            "frozen": True,
            "representation": "pooled",
        },
        "horizons": {"primary": 12},
        "anchor_policies": {"mci": "first"},
        "fdr_families": {"primary": ["endpoint"]},
        "replication_synthesis": "separate",
        "partitioning": {"outer_folds": 4, "inner_folds": 5},
    }


def test_draft_protocol_cannot_authorize_confirmatory_generation(tmp_path: Path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol_payload()))
    with pytest.raises(ValueError, match="requires a frozen protocol"):
        load_protocol(path, require_frozen=True)


def test_frozen_protocol_requires_approval_metadata():
    payload = protocol_payload("frozen")
    payload["approved_by"] = []
    with pytest.raises(ValueError, match="approver"):
        validate_protocol(payload)


def test_frozen_protocol_requires_encoder_revision():
    payload = protocol_payload("frozen")
    payload["encoder"]["hf_revision"] = None
    with pytest.raises(ValueError, match="hf_revision"):
        validate_protocol(payload)


def test_current_protocol_is_valid_draft():
    path = Path(__file__).resolve().parents[1] / "configs" / "protocol.json"
    protocol = load_protocol(path)
    assert protocol.status == "draft"
    assert protocol.replication("outer_repeat_01").analysis_role == "locked_repeated_nested_cv"
    assert protocol.payload["development_history"]["excluded_from_final_results"] is True
