from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
RESULTS_DIR = ARTIFACTS_DIR / "results"

ALL_PAIRS_PATH = PROCESSED_DIR / "all.jsonl"
JD_CACHE_PATH = PROCESSED_DIR / "jd_cache.json"
CV_CACHE_PATH = PROCESSED_DIR / "cv_cache.json"
FAILED_EXTRACTIONS_PATH = PROCESSED_DIR / "failed_extractions.json"

# Domain Constants
DEGREE_LEVELS = ["none", "high_school", "associate", "bachelor", "master", "phd"]

LABEL_MAP = {
    "No Fit": 0,
    "Potential Fit": 1,
    "Good Fit": 2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

MAJOR_GROUPS = [
    "computer_science",
    "information_technology",
    "software_engineering",
    "data_science_ai",
    "electrical_electronics",
    "mechanical_engineering",
    "civil_construction",
    "accounting_finance",
    "business_administration",
    "marketing_communications",
    "human_resources",
    "economics",
    "healthcare_medical",
    "biology_biotechnology",
    "chemistry_chemical_engineering",
    "physics_materials_science",
    "mathematics_statistics",
    "law_legal",
    "languages_linguistics",
    "graphic_design_multimedia",
    "supply_chain_logistics",
    "hospitality_tourism",
    "education_pedagogy",
    "environmental_science",
    "general_engineering",
    "other",
    "unknown",
]

CV_HEADERS = [
    "Professional Summary", "Summary", "Executive Profile", "Profile", "Objective",
    "Highlights", "Skills", "Core Qualifications", "Qualifications", "Technical Skills",
    "Experience", "Work History", "Professional Experience", "Education and Training",
    "Education", "Certifications", "Certification", "Accomplishments", "Additional Information",
    "Languages", "Interests", "Affiliations", "Training",
]

# Defaults & Model Constants
DEFAULT_DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
DEFAULT_VLLM_MODEL = "/tmp/Qwen3-14B-FP8"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

MAX_SEQ_LENGTH = 256
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50  # dùng cho cả CV và JD (JD chunk: giảng viên đã đồng ý)
EMBEDDING_BATCH_SIZE = 64

SKILL_THRESHOLDS = (0.6, 0.7, 0.8)
TAUS = SKILL_THRESHOLDS
WEIGHT_STEP = 0.05
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.50
DEFAULT_THRESHOLDS = (40, 70)

DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_SEEDS = tuple(range(5))
