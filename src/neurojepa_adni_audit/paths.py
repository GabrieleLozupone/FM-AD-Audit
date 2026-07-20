from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINKED_DATA_DIR = PROJECT_ROOT / "data" / "linked"
DEFAULT_MANIFEST_PATH = LINKED_DATA_DIR / "manifest.csv"
DEFAULT_EMBEDDING_INDEX_PATH = LINKED_DATA_DIR / "embedding_index.csv"
DEFAULT_PROGRESSION_PAIRS_PATH = LINKED_DATA_DIR / "progression_pairs.csv"
DEFAULT_HORIZON_PAIRS_PATH = LINKED_DATA_DIR / "horizon_matched_pairs.csv"
