from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

DEFAULT_DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

