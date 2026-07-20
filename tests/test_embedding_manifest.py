import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurojepa_adni_audit.embedding_manifest import build_embedding_integrity_manifest


def synthetic_embedding_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for index in range(3):
        embedding_path = tmp_path / f"embedding_{index}.npy"
        np.save(embedding_path, np.full(4, index, dtype=np.float32))
        rows.append(
            {
                "participant_id": f"p{index}",
                "session_id": "s0",
                "image_id": f"i{index}",
                "pooled_embedding_path": str(embedding_path),
            }
        )
    index_path = tmp_path / "index.csv"
    source_path = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    pd.DataFrame(rows)[["participant_id", "session_id", "image_id"]].to_csv(source_path, index=False)
    return index_path, source_path


def test_build_embedding_integrity_manifest_hashes_and_refuses_overwrite(tmp_path: Path):
    index_path, source_path = synthetic_embedding_inputs(tmp_path)
    output = build_embedding_integrity_manifest(
        embedding_index_path=index_path,
        source_manifest_path=source_path,
        output_dir=tmp_path / "integrity",
        expected_dimension=4,
    )
    summary = json.loads((output / "summary.json").read_text())
    records = pd.read_csv(output / "pooled_embeddings.csv")
    assert summary["status"] == "valid"
    assert summary["rows"] == 3
    assert records["pooled_embedding_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert records["pooled_embedding_finite"].all()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_embedding_integrity_manifest(
            embedding_index_path=index_path,
            source_manifest_path=source_path,
            output_dir=output,
            expected_dimension=4,
        )


@pytest.mark.parametrize("failure", ["non_finite", "wrong_shape", "key_mismatch"])
def test_embedding_integrity_manifest_fails_closed(tmp_path: Path, failure: str):
    index_path, source_path = synthetic_embedding_inputs(tmp_path)
    if failure == "key_mismatch":
        source = pd.read_csv(source_path)
        source.loc[0, "image_id"] = "different"
        source.to_csv(source_path, index=False)
    else:
        index = pd.read_csv(index_path)
        path = Path(index.loc[0, "pooled_embedding_path"])
        array = np.array([0, 1, 2, np.nan]) if failure == "non_finite" else np.zeros(3)
        np.save(path, array)
    with pytest.raises((ValueError, FileNotFoundError)):
        build_embedding_integrity_manifest(
            embedding_index_path=index_path,
            source_manifest_path=source_path,
            output_dir=tmp_path / "integrity",
            expected_dimension=4,
        )
    assert not (tmp_path / "integrity").exists()
