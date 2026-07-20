import pandas as pd

from neurojepa_adni_audit.tables.pairs import PAIR_MORPHOMETRY_COLUMNS, enrich_horizon_pairs


def test_enrich_horizon_pairs_adds_expanded_morphometry_and_excludes_six_months():
    manifest_rows = []
    for session, image, scale in (("s0", "i0", 1.0), ("s1", "i1", 2.0)):
        row = {"participant_id": "p1", "session_id": session, "image_id": image}
        row.update({column: scale for column in PAIR_MORPHOMETRY_COLUMNS})
        manifest_rows.append(row)
    manifest = pd.DataFrame(manifest_rows)
    pairs = pd.DataFrame(
        [
            {
                "participant_id": "p1",
                "current_session_id": "s0",
                "current_image_id": "i0",
                "next_session_id": "s1",
                "next_image_id": "i1",
                "requested_horizon_months": horizon,
            }
            for horizon in (6.0, 12.0, 24.0, 36.0)
        ]
    )
    enriched = enrich_horizon_pairs(pairs, manifest)
    assert set(enriched["requested_horizon_months"]) == {12.0, 24.0, 36.0}
    for prefix, expected in (("current", 1.0), ("next", 2.0)):
        for column in PAIR_MORPHOMETRY_COLUMNS:
            assert (enriched[f"{prefix}_{column}"] == expected).all()
