import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurojepa_adni_audit.nested_cv import build_nested_fold_assignments
from neurojepa_adni_audit.nested_training import participant_equal_weights, run_nested_tabular_fold
from neurojepa_adni_audit.protocol import load_protocol

from test_protocol import protocol_payload


def make_protocol(tmp_path: Path):
    payload = protocol_payload()
    payload["replications"] = [
        {"name": "repeat_1", "split_seed": 17, "analysis_role": "test"}
    ]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload))
    return load_protocol(path)


def test_participant_equal_weights_give_each_participant_equal_total():
    rows = pd.DataFrame({"participant_id": ["a", "a", "a", "b", "c", "c"]})
    weights = participant_equal_weights(rows)
    totals = pd.DataFrame({"participant_id": rows["participant_id"], "weight": weights}).groupby(
        "participant_id"
    )["weight"].sum()
    assert np.allclose(totals, totals.iloc[0])
    assert np.isclose(weights.mean(), 1.0)


def test_nested_regression_refits_outer_development_with_repeated_visit_weights(tmp_path: Path):
    protocol = make_protocol(tmp_path)
    participants = [f"p{index:03d}" for index in range(80)]
    subjects = pd.DataFrame(
        {"participant_id": participants, "split_stratum": [f"s{index % 4}" for index in range(80)]}
    )
    assignments = build_nested_fold_assignments(subjects, protocol)
    rows = []
    embeddings = []
    rng = np.random.default_rng(4)
    for index, participant in enumerate(participants):
        visits = 1 + index % 3
        for visit in range(visits):
            latent = index / 20 + visit / 10
            rows.append(
                {
                    "participant_id": participant,
                    "session_id": f"s{visit}",
                    "image_id": f"{participant}_i{visit}",
                    "current_target_value": 10 + latent,
                    "future_target_value": 10 + 2 * latent + rng.normal(scale=0.05),
                    "target_change": latent + rng.normal(scale=0.05),
                    "age_at_session": 60 + index % 20,
                    "sex_baseline": "F" if index % 2 else "M",
                }
            )
            embeddings.append([latent, latent**2, rng.normal()])
    table = pd.DataFrame(rows)
    pooled = np.asarray(embeddings, dtype=np.float32)
    summary, predictions, oof = run_nested_tabular_fold(
        table=table,
        assignments=assignments,
        full_pooled_matrix=pooled,
        repeat="repeat_1",
        outer_fold=0,
        task="future_cognition",
        target="target_change",
        feature_group="neurojepa_only",
        kind="regression",
        hyperparameters=[0.1, 1.0],
        seed=17,
        repeated_visits=True,
    )
    assert summary["final_refit_on_complete_outer_development"] is True
    assert summary["participant_equal_fit_weights"] is True
    assert len(summary["inner_history"]) == 10
    assert predictions["participant_id"].nunique() == summary["test_participants"]
    assert np.allclose(
        predictions["predicted_future_value"],
        predictions["current_target_value"] + predictions["predicted_change"],
    )
    assert oof["_nested_row_id"].is_unique


def test_nested_classification_uses_all_inner_folds_and_inner_oof_threshold(tmp_path: Path):
    protocol = make_protocol(tmp_path)
    participants = [f"p{index:03d}" for index in range(100)]
    subjects = pd.DataFrame(
        {"participant_id": participants, "split_stratum": [f"dx{index % 4}" for index in range(100)]}
    )
    assignments = build_nested_fold_assignments(subjects, protocol)
    rng = np.random.default_rng(8)
    labels = np.asarray([index % 2 for index in range(100)])
    pooled = np.column_stack([labels + rng.normal(scale=0.3, size=100), rng.normal(size=100)]).astype(
        np.float32
    )
    table = pd.DataFrame(
        {
            "participant_id": participants,
            "session_id": "s0",
            "image_id": [f"i{index}" for index in range(100)],
            "label": labels,
        }
    )
    summary, predictions, oof = run_nested_tabular_fold(
        table=table,
        assignments=assignments,
        full_pooled_matrix=pooled,
        repeat="repeat_1",
        outer_fold=0,
        task="ad_vs_cn",
        target="label",
        feature_group="neurojepa_only",
        kind="classification",
        hyperparameters=[0.1, 1.0],
        seed=17,
        repeated_visits=False,
    )
    assert len(summary["inner_history"]) == 10
    assert 0 <= summary["threshold_inner_oof"] <= 1
    assert set(predictions["predicted_label"].unique()) <= {0, 1}
    assert oof["_nested_row_id"].is_unique
