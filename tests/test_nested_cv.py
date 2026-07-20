import json
from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.nested_cv import (
    NestedFoldSpec,
    build_nested_fold_assignments,
    materialize_split,
)
from neurojepa_adni_audit.protocol import load_protocol

from test_protocol import protocol_payload


def test_repeated_nested_cv_is_deterministic_complete_and_participant_disjoint(tmp_path: Path):
    payload = protocol_payload()
    payload["replications"].append(
        {"name": "replication_03", "split_seed": 3, "analysis_role": "confirmation"}
    )
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    protocol = load_protocol(path)
    subjects = pd.DataFrame(
        {
            "participant_id": [f"p{index:03d}" for index in range(120)],
            "split_stratum": [f"diagnosis_{index % 6}" for index in range(120)],
        }
    )

    first = build_nested_fold_assignments(subjects, protocol)
    second = build_nested_fold_assignments(subjects, protocol)
    assert first.equals(second)
    assert len(first) == 3 * 4 * 120

    for repeat in (item.name for item in protocol.replications):
        repeat_rows = first[first["repeat"] == repeat]
        test_counts = repeat_rows.groupby("participant_id")["outer_role"].apply(
            lambda values: int((values == "test").sum())
        )
        assert (test_counts == 1).all()
        for outer in range(4):
            outer_split = materialize_split(first, NestedFoldSpec(repeat, outer))
            assert set(outer_split["subject_split"]) == {"train", "test"}
            assert outer_split["participant_id"].nunique() == 120
            validation_seen = set()
            for inner in range(5):
                inner_split = materialize_split(first, NestedFoldSpec(repeat, outer, inner))
                train_ids = set(inner_split.loc[inner_split["subject_split"] == "train", "participant_id"])
                val_ids = set(inner_split.loc[inner_split["subject_split"] == "val", "participant_id"])
                assert train_ids
                assert val_ids
                assert not train_ids & val_ids
                assert not validation_seen & val_ids
                validation_seen |= val_ids
            expected_development = set(
                outer_split.loc[outer_split["subject_split"] == "train", "participant_id"]
            )
            assert validation_seen == expected_development
