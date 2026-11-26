"""Cấu hình hệ thống web Chương 5, nạp mô hình và tham số đã chọn trên tập kiểm định."""
import os
from pathlib import Path

from src.config import ARTIFACTS_DIR, DEFAULT_EMBEDDING_MODEL, PROJECT_ROOT, RESULTS_DIR

WEB_DIR = PROJECT_ROOT / "web"
DEFAULT_DB_PATH = WEB_DIR / "data" / "screening.db"
FRONTEND_DIST_DIR = WEB_DIR / "frontend" / "dist"

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB mỗi tệp
MIN_TEXT_CHARS = 50
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ENCODE_BATCH_SIZE = 64

WEB_MODEL_MODE = os.environ.get("WEB_MODEL_MODE", "m5").lower()
if WEB_MODEL_MODE == "m4":
    MODEL_SOURCE = DEFAULT_EMBEDDING_MODEL
    MODEL_CONFIG_PATH = RESULTS_DIR / "jd" / "seed0" / "e4.json"
    MODEL_VERSION = "M4-MiniLM-L12-v2-seed0"
else:
    MODEL_SOURCE = str(ARTIFACTS_DIR / "models" / "e5_finetuned" / "jd" / "seed4")
    MODEL_CONFIG_PATH = RESULTS_DIR / "jd" / "seed4" / "e5.json"
    MODEL_VERSION = "M5-MiniLM-L12-v2-seed4"

LEVEL_LABELS = {
    0: "Không phù hợp",
    1: "Tiềm năng",
    2: "Phù hợp",
}


def get_db_path() -> Path:
    return Path(os.environ.get("WEB_DB_PATH", str(DEFAULT_DB_PATH)))
