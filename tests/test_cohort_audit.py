import json
from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.evaluation.cohort_audit import summarize_nested_cohort, wilson_interval
from neurojepa_adni_audit.nested_cv import build_nested_fold_assignments
from neurojepa_adni_audit.protocol import load_protocol

from test_protocol import protocol_payload


def test_design_only_cohort_audit_has_disjoint_fold_counts_and_no_performance(tmp_path: Path):
    payload = protocol_payload()
    payload["replications"] = [{"name": "r1", "split_seed": 4, "analysis_role": "test"}]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    protocol = load_protocol(path)
    table = pd.DataFrame(
        {
            "participant_id": [f"p{i:02d}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
            "age": [60 + i for i in range(40)],
        }
    )
    subjects = pd.DataFrame(
        {"participant_id": table["participant_id"], "split_stratum": table["label"].astype(str)}
    )
    assignments = build_nested_fold_assignments(subjects, protocol)
    audit = summarize_nested_cohort(
        table=table,
        assignments=assignments,
        task_name="binary",
        outcome_column="label",
        missing_columns=["age"],
    )
    assert audit["comparative_performance_computed"] is False
    assert len(audit["outer_folds"]) == 4
    assert sum(fold["test"]["participants"] for fold in audit["outer_folds"]) == 40
    assert wilson_interval(20, 40)[0] < 0.5 < wilson_interval(20, 40)[1]
