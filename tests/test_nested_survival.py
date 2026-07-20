import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurojepa_adni_audit.nested_cv import build_nested_fold_assignments
from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.survival.nested import run_nested_survival_fold

from test_protocol import protocol_payload


def test_nested_survival_selects_inner_uno_and_refits_outer_development(tmp_path: Path):
    payload = protocol_payload()
    payload["replications"] = [{"name": "repeat_1", "split_seed": 9, "analysis_role": "test"}]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(payload))
    protocol = load_protocol(protocol_path)
    participants = [f"p{index:03d}" for index in range(100)]
    event = np.asarray([index % 3 != 0 for index in range(100)], dtype=int)
    subjects = pd.DataFrame(
        {
            "participant_id": participants,
            "split_stratum": [f"event_{value}" for value in event],
        }
    )
    assignments = build_nested_fold_assignments(subjects, protocol)
    rng = np.random.default_rng(12)
    signal = rng.normal(size=100)
    durations = np.exp(2.5 - 0.5 * signal) + 1.0
    table = pd.DataFrame(
        {
            "participant_id": participants,
            "session_id": "s0",
            "image_id": [f"i{index}" for index in range(100)],
            "duration_months": durations,
            "event": event,
            "survival_status": np.where(event == 1, "event", "censored"),
        }
    )
    pooled = np.column_stack([signal, rng.normal(size=100)]).astype(np.float32)
    summary, predictions, history, calibration, schoenfeld = run_nested_survival_fold(
        table=table,
        assignments=assignments,
        full_pooled_matrix=pooled,
        repeat="repeat_1",
        outer_fold=0,
        feature_group="neurojepa_only",
        penalizers=[0.1, 1.0],
        l1_ratios=[0.0],
        horizons=[5.0, 10.0, 36.0],
        tau=20.0,
    )
    assert summary["final_refit_on_complete_outer_development"] is True
    assert len(history) == 10
    assert summary["outer_test_uno_c_36m"] is not None
    assert len(predictions) == summary["test_participants"]
    assert set(calibration["horizon_months"]) == {5.0, 10.0, 36.0}
    assert summary["ph_global_test"]["method"].startswith("fisher_combination")
    assert not schoenfeld.empty
