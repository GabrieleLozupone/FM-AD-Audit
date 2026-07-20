import json
from pathlib import Path

import pandas as pd
import pytest

from neurojepa_adni_audit.protocol import load_protocol
from neurojepa_adni_audit.paper_run import bind_paper_run
from neurojepa_adni_audit.replications import (
    assert_disjoint_splits,
    assign_subject_splits,
    build_subject_table,
    prepare_replication,
)

from test_protocol import protocol_payload


def synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    manifest_rows = []
    pair_rows = []
    embedding_rows = []
    for index in range(30):
        participant = f"p{index:03d}"
        diagnosis = ("CN", "MCI", "AD")[index % 3]
        future_diagnosis = "AD" if index % 5 == 0 else diagnosis
        for month, session, session_diagnosis in ((0, "ses-00", diagnosis), (12, "ses-12", future_diagnosis)):
            image_id = f"{participant}_{session}"
            manifest_rows.append(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "session_month": month,
                    "diagnosis_at_session": session_diagnosis,
                    "image_id": image_id,
                }
            )
            embedding_rows.append({"participant_id": participant, "session_id": session, "image_id": image_id})
        pair_rows.append(
            {
                "pair_id": f"pair_{participant}",
                "participant_id": participant,
                "diagnosis_changed": diagnosis != future_diagnosis,
            }
        )
    paths = tuple(tmp_path / name for name in ("manifest.csv", "embedding.csv", "pairs.csv", "horizon.csv"))
    pd.DataFrame(manifest_rows).to_csv(paths[0], index=False)
    pd.DataFrame(embedding_rows).to_csv(paths[1], index=False)
    pd.DataFrame(pair_rows).to_csv(paths[2], index=False)
    pd.DataFrame(pair_rows).to_csv(paths[3], index=False)
    return paths


def test_subject_split_is_deterministic_and_disjoint(tmp_path: Path):
    manifest_path, _embedding, pairs_path, _horizon = synthetic_inputs(tmp_path)
    manifest = pd.read_csv(manifest_path)
    pairs = pd.read_csv(pairs_path)
    subjects = build_subject_table(manifest, pairs)
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
    first = assign_subject_splits(subjects, ratios, seed=7)
    second = assign_subject_splits(subjects, ratios, seed=7)
    assert first[["participant_id", "subject_split"]].equals(second[["participant_id", "subject_split"]])
    assert_disjoint_splits(first)
    assert set(first["subject_split"]) == {"train", "val", "test"}


def test_replication_directory_is_immutable_and_records_hashes(tmp_path: Path):
    manifest, embedding, pairs, horizon = synthetic_inputs(tmp_path)
    protocol_path = tmp_path / "protocol.json"
    payload = protocol_payload("frozen")
    protocol_path.write_text(json.dumps(payload))
    protocol = load_protocol(protocol_path, require_frozen=True)
    kwargs = {
        "protocol": protocol,
        "replication": protocol.replication("replication_01"),
        "manifest_path": manifest,
        "embedding_index_path": embedding,
        "progression_pairs_path": pairs,
        "horizon_pairs_path": horizon,
        "output_root": tmp_path / "outputs",
        "project_root": Path(__file__).resolve().parents[1],
        "allow_dirty_worktree": True,
    }
    output = prepare_replication(**kwargs)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["protocol"]["status"] == "frozen"
    assert provenance["replication"]["split_seed"] == 1
    assert "manifest.csv" in provenance["outputs"]
    context = bind_paper_run(
        protocol_path=protocol_path,
        replication_name="replication_01",
        manifest_path=output / "manifest.csv",
        embedding_index_path=output / "embedding_index.csv",
    )
    assert context.metadata()["split_seed"] == 1
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_replication(**kwargs)
