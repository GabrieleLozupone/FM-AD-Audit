import json
from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.evaluation.reporting import run_paired_report
from neurojepa_adni_audit.protocol import load_protocol

from test_protocol import protocol_payload


def test_report_pairs_predictions_and_adjusts_within_family(tmp_path: Path):
    conventional_rows = []
    combined_rows = []
    for index in range(20):
        target = float(index % 5)
        base = {
            "participant_id": f"p{index:02d}",
            "session_id": "ses-00",
            "image_id": f"image-{index:02d}",
            "subject_split": "test",
            "target_value": target,
        }
        conventional_rows.append({**base, "prediction": target + 1.0})
        combined_rows.append({**base, "prediction": target + 0.1})
    conventional_path = tmp_path / "conventional.csv"
    combined_path = tmp_path / "combined.csv"
    pd.DataFrame(conventional_rows).to_csv(conventional_path, index=False)
    pd.DataFrame(combined_rows).to_csv(combined_path, index=False)

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol_payload("frozen")))
    protocol = load_protocol(protocol_path, require_frozen=True)
    comparison = {
        "id": "primary_mae",
        "family": "primary",
        "replication": "replication_02",
        "metric": "subject_equal_mae",
        "bootstrap_replicates": 1000,
        "combined_predictions": str(combined_path),
        "conventional_predictions": str(conventional_path),
        "pair_keys": ["participant_id", "session_id", "image_id"],
        "target_column": "target_value",
        "combined_prediction_column": "prediction",
        "conventional_prediction_column": "prediction",
    }
    spec_path = tmp_path / "report.json"
    spec_path.write_text(json.dumps({"comparisons": [comparison]}))
    frame, metadata = run_paired_report(
        spec_path=spec_path, protocol=protocol, output_dir=tmp_path / "report"
    )
    assert frame.loc[0, "estimate"] < 0
    assert frame.loc[0, "ci95_high"] < 0
    assert frame.loc[0, "fdr_adjusted_p_value"] <= 0.01
    assert metadata["protocol_status"] == "frozen"
    assert (tmp_path / "report" / "paired_effects.csv").exists()
