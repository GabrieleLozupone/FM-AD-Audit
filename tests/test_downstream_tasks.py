from pathlib import Path

import pandas as pd

from neurojepa_adni_audit.probes.downstream import load_task_table


def test_ad_vs_cn_uses_first_mri_overall_and_does_not_substitute_later_ad(tmp_path: Path):
    rows = [
        {
            "participant_id": "mci_then_ad",
            "session_id": "s0",
            "image_id": "m0",
            "session_month": 0,
            "diagnosis_at_session": "MCI",
            "subject_split": "train",
        },
        {
            "participant_id": "mci_then_ad",
            "session_id": "s1",
            "image_id": "m1",
            "session_month": 12,
            "diagnosis_at_session": "AD",
            "subject_split": "train",
        },
        {
            "participant_id": "baseline_cn",
            "session_id": "s0",
            "image_id": "c0",
            "session_month": 0,
            "diagnosis_at_session": "CN",
            "subject_split": "test",
        },
    ]
    manifest_path = tmp_path / "manifest.csv"
    index_path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    pd.DataFrame(rows)[["participant_id", "session_id", "image_id"]].to_csv(index_path, index=False)
    table = load_task_table(manifest_path, index_path, "ad_vs_cn")
    assert table["participant_id"].tolist() == ["baseline_cn"]
    assert table["label"].tolist() == [0]
