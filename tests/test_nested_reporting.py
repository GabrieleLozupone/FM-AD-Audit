import json
from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.evaluation.nested_reporting import run_nested_paired_report
from neurojepa_adni_audit.protocol import load_protocol

from test_protocol import protocol_payload


def _write_predictions(root: Path, feature: str, repeat: str, outer: int, rows: list[dict]) -> None:
    directory = root / "same_visit_cognition" / "adas13_at_session" / feature / repeat / f"outer_{outer}"
    directory.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(directory / "outer_test_predictions.csv", index=False)


def test_nested_report_pairs_outer_oof_and_bootstraps_correlated_repeats(tmp_path: Path):
    protocol_data = protocol_payload()
    protocol_data["bootstrap"]["replicates"] = 1000
    protocol_data["bootstrap"]["seed"] = 123
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol_data))
    protocol = load_protocol(protocol_path)
    root = tmp_path / "predictions"
    repeats = [item["name"] for item in protocol_data["replications"]]
    for repeat in repeats:
        for outer in (0, 1):
            rows_combined = []
            rows_conventional = []
            for offset in range(4):
                index = outer * 4 + offset
                common = {
                    "participant_id": f"p{index}",
                    "session_id": "s0",
                    "image_id": f"i{index}",
                    "repeat": repeat,
                    "outer_fold": outer,
                    "truth": float(index),
                }
                rows_combined.append({**common, "prediction": float(index) + 0.1})
                rows_conventional.append({**common, "prediction": float(index) + 1.0})
            _write_predictions(root, "neurojepa_conventional", repeat, outer, rows_combined)
            _write_predictions(root, "conventional_full", repeat, outer, rows_conventional)
    spec = {
        "comparisons": [
            {
                "id": "same_visit_adas13_subject_equal_mae",
                "family": "global_core_endpoints",
                "predictions_root": str(root),
                "task": "same_visit_cognition",
                "target": "adas13_at_session",
                "metric": "subject_equal_mae",
            }
        ]
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    effects, dispersion = run_nested_paired_report(
        spec_path=spec_path, protocol=protocol, output_dir=tmp_path / "report"
    )
    assert len(effects) == 1
    assert effects.iloc[0]["estimate"] < 0
    assert effects.iloc[0]["n_subjects"] == 8
    assert effects.iloc[0]["repeats"] == 2
    assert len(dispersion) == 4
